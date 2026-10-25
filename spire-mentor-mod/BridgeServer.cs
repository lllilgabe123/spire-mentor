// BridgeServer.cs -- loopback TCP transport between SpireMentor and its agent client.
// Originally ported from zhiyue/sts2-rl-agent (STS2BridgeMod); rewritten clean-room.

using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace SpireMentor;

public class BridgeServer
{
    public static readonly BridgeServer Instance = new();

    private const string PongJson = "{\"type\":\"pong\"}";
    private const int WriteTimeoutMs = 5000;

    // Socket state. _clientLock guards _client/_netStream AND serializes every
    // socket write, so lines from different threads never interleave.
    private readonly object _clientLock = new();
    private TcpListener? _tcpListener;
    private TcpClient? _client;
    private NetworkStream? _netStream;
    private volatile bool _running;
    private CancellationTokenSource? _cts;

    // Inbound framing state; only the background read loop touches these
    // (the accept loop resets the remainder per client).
    private readonly byte[] _rxBytes = new byte[8192];
    private string _lineRemainder = "";
    // True once the current connection has delivered its first complete line.
    private bool _sawFirstLine;

    /// <summary>
    /// Longest unterminated inbound line accepted (chars). Agent replies are a
    /// few KB; a client that streams this much without a newline is dropped
    /// instead of growing the buffer without bound.
    /// </summary>
    private const int MaxInboundLineChars = 8 * 1024 * 1024;

    /// <summary>
    /// An HTTP request line ("POST / HTTP/1.1"). A browser page can make the
    /// user's browser connect to this loopback port; such a connection is
    /// dropped before its body could be read as protocol lines.
    /// </summary>
    private static readonly Regex HttpRequestLine =
        new(@"^[A-Z]{3,10} \S+ HTTP/\d", RegexOptions.CultureInvariant);
    private Decoder _decoder = Encoding.UTF8.GetDecoder();

    // Autopilot lane: exactly one pending request at a time. A reply is only
    // delivered when its request_id matches _awaitedRequestId (or, for
    // WaitForActionAsync, when no id is recorded at all).
    private readonly object _awaitLock = new();
    private TaskCompletionSource<string>? _awaitedReply;
    private string? _awaitedRequestId;
    private long _requestCounter;

    // Advice lane (phase 5, PROTOCOL.md v3): non-blocking overlapping
    // requests keyed by their own request_id namespace ("a<N>", disjoint
    // from the autopilot counter's bare numbers). Deliberately a SEPARATE
    // map so advisor traffic can never consume -- or be consumed by -- the
    // autopilot's single _awaitedReply slot: an "advice" reply is routed
    // here by its action field before the pending-action delivery runs.
    private readonly Dictionary<string, Action<JsonElement>> _pendingAdvice = new();
    private readonly object _adviceLock = new();
    private long _adviceCounter;
    // Backstop against a stalled/unresponsive advisor leaking entries;
    // far above any realistic number of in-flight requests.
    private const int MaxPendingAdvice = 32;

    public bool IsClientConnected
    {
        get
        {
            lock (_clientLock)
            {
                return _client != null && _client.Connected;
            }
        }
    }

    private BridgeServer()
    {
    }

    /// <summary>The port actually listened on (0 until Start succeeds, and
    /// again after Stop); the
    /// advisor is launched against this one.</summary>
    public int Port { get; private set; }

    /// <summary>
    /// <see cref="Start"/> on <paramref name="port"/>, else the next free one
    /// of <paramref name="tries"/>: a second game instance on the same machine
    /// (the co-op test lane) reads the same conf, and its advisor must not
    /// end up talking to the first instance.
    /// </summary>
    public void StartWithFallback(int port, int tries)
    {
        for (int i = 0; ; i++)
        {
            try
            {
                Start(port + i);
                return;
            }
            catch (SocketException ex) when (ex.SocketErrorCode == SocketError.AddressAlreadyInUse && i + 1 < tries)
            {
                Logger.Log($"[BridgeServer] Port {port + i} is taken; trying {port + i + 1}.");
            }
        }
    }

    /// <summary>
    /// Starts listening on 127.0.0.1:<paramref name="port"/> and launches the
    /// background accept loop. A bind failure throws to the caller.
    /// </summary>
    public void Start(int port)
    {
        if (_running)
        {
            return;
        }

        _running = true;
        _cts = new CancellationTokenSource();
        var listener = new TcpListener(IPAddress.Loopback, port);
        try
        {
            listener.Start();
        }
        catch
        {
            // CLEANROOM2-Q1: leave the server restartable after a failed bind.
            _running = false;
            throw;
        }
        _tcpListener = listener;
        Port = port;
        Logger.Log($"[BridgeServer] Listening on {IPAddress.Loopback}:{port}");

        CancellationToken token = _cts.Token;
        _ = Task.Run(() => ServeClientsAsync(token));
    }

    /// <summary>Stops the listener and drops the client. Safe when never started.</summary>
    public void Stop()
    {
        _running = false;
        try
        {
            _cts?.Cancel();
        }
        catch (ObjectDisposedException)
        {
        }
        AbandonAwaitedReply("server stopping");
        DropClient();
        try
        {
            _tcpListener?.Stop();
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] Listener stop failed: {ex.Message}");
        }
        _tcpListener = null;
        Port = 0;
        Logger.Log("[BridgeServer] Stopped");
    }

    /// <summary>Fire-and-forget: send one JSON line, ignore whether it went out.</summary>
    public void SendState(string stateJson)
    {
        TryWriteLine(stateJson);
    }

    /// <summary>
    /// Writes <paramref name="stateJson"/> plus a single '\n' as one write and
    /// flushes. Returns false when there is no connected client or the write
    /// failed (the client is then dropped). Never throws.
    /// </summary>
    private bool TryWriteLine(string stateJson)
    {
        lock (_clientLock)
        {
            if (_netStream == null || _client == null || !_client.Connected)
            {
                return false;
            }

            try
            {
                byte[] line = Encoding.UTF8.GetBytes(stateJson + "\n");
                _netStream.Write(line, 0, line.Length);
                _netStream.Flush();
                return true;
            }
            catch (Exception ex)
            {
                Logger.Log($"[BridgeServer] Send failed, dropping client: {ex.Message}");
                // Monitor locks are re-entrant, so this nested lock is fine.
                DropClient();
                return false;
            }
        }
    }

    /// <summary>
    /// Sends a decision message tagged with a fresh autopilot request_id and
    /// waits for the reply carrying the same id. Returns the raw reply line,
    /// or null if the send failed, the wait timed out, was cancelled, or the
    /// client went away.
    /// </summary>
    public async Task<string?> SendStateAndWaitForActionAsync(string stateJson, TimeSpan timeout, CancellationToken ct)
    {
        string requestId = Interlocked.Increment(ref _requestCounter).ToString(System.Globalization.CultureInfo.InvariantCulture);
        TaskCompletionSource<string> tcs = InstallPending(requestId);
        try
        {
            if (!TryWriteLine(WithRequestId(stateJson, requestId)))
            {
                return null;
            }
            return await WaitForPendingAsync(tcs, timeout, ct);
        }
        finally
        {
            ReleasePending(tcs);
        }
    }

    /// <summary>
    /// Replaces the pending slot with a new completion source (cancelling the
    /// one it displaces) and records <paramref name="requestId"/> (null = accept
    /// any reply).
    /// </summary>
    private TaskCompletionSource<string> InstallPending(string? requestId)
    {
        var tcs = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        lock (_awaitLock)
        {
            _awaitedReply?.TrySetCanceled();
            _awaitedReply = tcs;
            _awaitedRequestId = requestId;
        }
        return tcs;
    }

    /// <summary>Clears the pending slot, but only if it still belongs to <paramref name="tcs"/>.</summary>
    private void ReleasePending(TaskCompletionSource<string> tcs)
    {
        lock (_awaitLock)
        {
            if (ReferenceEquals(_awaitedReply, tcs))
            {
                _awaitedReply = null;
                _awaitedRequestId = null;
            }
        }
    }

    /// <summary>
    /// Fire-and-forget advice request (phase 5): sends the payload with an
    /// "a<N>" request_id and registers a callback for the correlated
    /// {"action":"advice"} reply. NON-BLOCKING -- returns immediately; the
    /// callback runs on the SOCKET thread (marshal before touching the
    /// scene tree). Requests may overlap freely; each reply resolves its
    /// own entry. Returns the request_id, or null if the send failed.
    /// Superseded/stale requests can be dropped via CancelAdviceRequest;
    /// replies for unknown ids are logged and discarded.
    /// </summary>
    public string? SendAdviceRequest(string stateJson, Action<JsonElement> onAdvice)
    {
        string requestId = "a" + Interlocked.Increment(ref _adviceCounter);
        lock (_adviceLock)
        {
            if (_pendingAdvice.Count >= MaxPendingAdvice)
            {
                Logger.Log($"[BridgeServer] Advice backlog at {MaxPendingAdvice}; clearing stale entries.");
                _pendingAdvice.Clear();
            }
            _pendingAdvice[requestId] = onAdvice;
        }

        string payload = WithRequestId(stateJson, requestId);
        if (TryWriteLine(payload))
        {
            return requestId;
        }
        lock (_adviceLock)
        {
            _pendingAdvice.Remove(requestId);
        }
        return null;
    }

    /// <summary>Drop an in-flight advice request (human already moved).</summary>
    public void CancelAdviceRequest(string requestId)
    {
        lock (_adviceLock)
        {
            _pendingAdvice.Remove(requestId);
        }
    }

    /// <summary>
    /// Waits for the next inbound line without sending anything and without
    /// request_id correlation (any reply is accepted). Null on timeout,
    /// cancellation or disconnect.
    /// </summary>
    public async Task<string?> WaitForActionAsync(TimeSpan timeout, CancellationToken ct)
    {
        TaskCompletionSource<string> tcs = InstallPending(null);
        try
        {
            return await WaitForPendingAsync(tcs, timeout, ct);
        }
        finally
        {
            ReleasePending(tcs);
        }
    }

    /// <summary>
    /// Awaits <paramref name="tcs"/> until it completes, <paramref name="ct"/>
    /// fires or <paramref name="timeout"/> elapses; the latter two cancel the
    /// source. A timeout too long for a CancellationTokenSource (over
    /// int.MaxValue ms) means "no timeout". Returns the reply, or null for any
    /// cancellation/failure; never throws.
    /// </summary>
    private static async Task<string?> WaitForPendingAsync(
        TaskCompletionSource<string> tcs, TimeSpan timeout, CancellationToken ct)
    {
        if (timeout.TotalMilliseconds > int.MaxValue)
        {
            timeout = Timeout.InfiniteTimeSpan;
        }
        try
        {
            using var timeoutCts = new CancellationTokenSource(timeout);
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, timeoutCts.Token);
            using CancellationTokenRegistration registration =
                linked.Token.Register(() => tcs.TrySetCanceled());
            return await tcs.Task;
        }
        catch (OperationCanceledException)
        {
            return null;
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] Waiting for action failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Background loop: accept one client at a time, serve it until it goes
    /// away, then accept the next. Runs until Stop().
    /// </summary>
    private async Task ServeClientsAsync(CancellationToken ct)
    {
        while (_running && !ct.IsCancellationRequested)
        {
            try
            {
                TcpListener? listener = _tcpListener;
                if (listener == null)
                {
                    break;
                }

                Logger.Log("[BridgeServer] Listening; no client connected yet");
                TcpClient client = await listener.AcceptTcpClientAsync(ct);
                Logger.Log($"[BridgeServer] Client connected from {client.Client.RemoteEndPoint}");

                lock (_clientLock)
                {
                    client.SendTimeout = WriteTimeoutMs;
                    // Reads must block indefinitely: NetworkStream.ReadTimeout
                    // rejects 0, so "no timeout" has to be Timeout.Infinite.
                    // CLEANROOM2-Q12: set through the Socket, which maps
                    // Timeout.Infinite portably (TcpClient.ReceiveTimeout passes
                    // -1 straight to setsockopt, which Linux rejects).
                    client.Client.ReceiveTimeout = Timeout.Infinite;
                    NetworkStream stream = client.GetStream();
                    stream.WriteTimeout = WriteTimeoutMs;
                    stream.ReadTimeout = Timeout.Infinite;
                    _client = client;
                    _netStream = stream;
                    _lineRemainder = "";
                    _sawFirstLine = false;
                    _decoder = Encoding.UTF8.GetDecoder();
                }

                await ReadLoopAsync(ct);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                if (!_running || ct.IsCancellationRequested)
                {
                    break;
                }
                Logger.Log($"[BridgeServer] Accept loop error: {ex.Message}");
                try
                {
                    await Task.Delay(1000, ct);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
        }
    }

    /// <summary>
    /// Reads the current client's stream until EOF, error or cancellation,
    /// splitting the byte stream into '\n'-terminated lines. Always leaves the
    /// pending slot cancelled and the client disconnected.
    /// </summary>
    private async Task ReadLoopAsync(CancellationToken ct)
    {
        try
        {
            while (_running && !ct.IsCancellationRequested)
            {
                NetworkStream? stream;
                lock (_clientLock)
                {
                    stream = _netStream;
                }
                if (stream == null)
                {
                    break;
                }

                int read = await stream.ReadAsync(_rxBytes.AsMemory(0, _rxBytes.Length), ct);
                if (read == 0)
                {
                    Logger.Log("[BridgeServer] Client closed the connection");
                    break;
                }

                // CLEANROOM2-Q11: a stateful decoder keeps a multi-byte UTF-8
                // sequence that is split across two reads intact.
                int charCount = _decoder.GetCharCount(_rxBytes, 0, read);
                char[] chars = new char[charCount];
                _decoder.GetChars(_rxBytes, 0, read, chars, 0);
                _lineRemainder += new string(chars);

                if (!DispatchCompleteLines())
                {
                    break;
                }
                if (_lineRemainder.Length > MaxInboundLineChars)
                {
                    Logger.Log($"[BridgeServer] Dropping client: unterminated line exceeds {MaxInboundLineChars} chars");
                    break;
                }
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] Read loop error: {ex.Message}");
        }
        finally
        {
            AbandonAwaitedReply("client disconnected");
            DropClient();
        }
    }

    /// <summary>
    /// Routes every complete line in the remainder; keeps the trailing partial
    /// line. Returns false when the connection must be dropped.
    /// </summary>
    private bool DispatchCompleteLines()
    {
        int newline;
        while ((newline = _lineRemainder.IndexOf('\n')) >= 0)
        {
            string line = _lineRemainder.Substring(0, newline).Trim();
            _lineRemainder = _lineRemainder.Substring(newline + 1);
            if (line.Length == 0)
            {
                continue;
            }
            if (!_sawFirstLine)
            {
                _sawFirstLine = true;
                if (HttpRequestLine.IsMatch(line))
                {
                    Logger.Log("[BridgeServer] Dropping client: it spoke HTTP (a web page, not an agent)");
                    return false;
                }
            }

            try
            {
                ProcessIncomingMessage(line);
            }
            catch (Exception ex)
            {
                // CLEANROOM2-Q2: one bad line must not tear down the connection.
                Logger.Log($"[BridgeServer] Error processing inbound line: {ex.Message}");
            }
        }
        return true;
    }

    /// <summary>
    /// Inbound routing (socket thread; must not touch the scene tree). Order:
    /// transport/side-lane verbs first (ping, advice, rewind), then delivery
    /// to the pending autopilot request, else drop.
    /// </summary>
    private void ProcessIncomingMessage(string line)
    {
        JsonDocument? doc = null;
        try
        {
            doc = JsonDocument.Parse(line);
        }
        catch (JsonException)
        {
            // Unparsable: may still satisfy an uncorrelated WaitForActionAsync.
        }

        using (doc)
        {
            JsonElement? root = doc != null && doc.RootElement.ValueKind == JsonValueKind.Object
                ? doc.RootElement
                : null;

            if (root is JsonElement obj && TryRouteSideLane(obj))
            {
                return;
            }

            DeliverToPending(line, root);
        }
    }

    /// <summary>
    /// Handles ping/advice/rewind messages that never touch the pending slot.
    /// Returns true when the message was consumed.
    /// </summary>
    private bool TryRouteSideLane(JsonElement root)
    {
        if (TryGetString(root, "action", out string action))
        {
            if (Is(action, "ping"))
            {
                SendState(PongJson);
                return true;
            }
            if (Is(action, "advice"))
            {
                // The document is disposed after routing; hand over a copy.
                DeliverAdvice(root.Clone());
                return true;
            }
            if (Is(action, "coach"))
            {
                // PROTOCOL.md v5 "coach": after-the-fact feedback, answers no
                // request. Routed here -- never to DeliverToPending, where an
                // uncorrelated autopilot wait would swallow it as its reply.
                DeliverCoach(root);
                return true;
            }
            if (Is(action, "list_snapshots"))
            {
                SendSnapshotList();
                return true;
            }
            if (Is(action, "undo_turn"))
            {
                // A wrong-typed field drops the request: rewinding is never guessed.
                int turn = 0;
                if (root.TryGetProperty("turn", out JsonElement turnEl) &&
                    !(turnEl.ValueKind == JsonValueKind.Number && turnEl.TryGetInt32(out turn)))
                {
                    Logger.Log("[BridgeServer] undo_turn ignored: 'turn' is not an integer");
                    return true;
                }
                string? tapeId = null;
                if (root.TryGetProperty("tape_id", out JsonElement tapeEl) &&
                    tapeEl.ValueKind != JsonValueKind.Null)
                {
                    if (tapeEl.ValueKind != JsonValueKind.String)
                    {
                        Logger.Log("[BridgeServer] undo_turn ignored: 'tape_id' is not a string");
                        return true;
                    }
                    tapeId = tapeEl.GetString();
                }
                SimCombatUndo.RequestUndoTurn(turn, tapeId);
                return true;
            }
            if (Is(action, "restore_snapshot"))
            {
                string? snapshotId = TryGetString(root, "snapshot_id", out string sid) ? sid : null;
                if (string.IsNullOrEmpty(snapshotId))
                {
                    bool wrongType = root.TryGetProperty("snapshot_id", out JsonElement sidEl) &&
                                     sidEl.ValueKind != JsonValueKind.String &&
                                     sidEl.ValueKind != JsonValueKind.Null;
                    string detail = wrongType ? "snapshot_id must be a string" : "missing snapshot_id";
                    SendState($"{{\"type\":\"error\",\"error\":\"snapshot_not_found\",\"detail\":\"{detail}\"}}");
                }
                else
                {
                    SimRestore.RequestRestore(snapshotId);
                }
                return true;
            }
        }

        if (TryGetString(root, "type", out string type) && Is(type, "PING"))
        {
            SendState(PongJson);
            return true;
        }

        return false;
    }

    /// <summary>
    /// Completes the pending autopilot request with <paramref name="line"/> if
    /// it is the awaited reply; otherwise logs and drops it.
    /// </summary>
    private void DeliverToPending(string line, JsonElement? root)
    {
        lock (_awaitLock)
        {
            if (_awaitedReply != null)
            {
                if (_awaitedRequestId != null)
                {
                    if (root is not JsonElement obj)
                    {
                        Logger.Log($"[BridgeServer] Dropping unparsable reply while awaiting request_id={_awaitedRequestId}: {line}");
                        return;
                    }

                    // CLEANROOM2-Q3: request_id must be echoed as a JSON string.
                    string? gotId = TryGetString(obj, "request_id", out string id) ? id : null;
                    if (gotId != _awaitedRequestId)
                    {
                        Logger.Log($"[BridgeServer] Dropping reply with request_id={gotId ?? "null"}; " +
                                   $"pending request_id={_awaitedRequestId}");
                        return;
                    }
                }

                TaskCompletionSource<string> tcs = _awaitedReply;
                _awaitedReply = null;
                _awaitedRequestId = null;
                tcs.TrySetResult(line);
                return;
            }
        }

        Logger.Log($"[BridgeServer] Message arrived with no pending request, dropping: {line}");
    }

    private static bool Is(string value, string expected) =>
        string.Equals(value, expected, StringComparison.OrdinalIgnoreCase);

    private static bool TryGetString(JsonElement obj, string name, out string value)
    {
        if (obj.TryGetProperty(name, out JsonElement el) && el.ValueKind == JsonValueKind.String)
        {
            value = el.GetString() ?? "";
            return true;
        }
        value = "";
        return false;
    }

    /// <summary>Answer `list_snapshots` with the current run's snapshot
    /// metadata (rewind lane, PROTOCOL.md section 7).</summary>
    private void SendSnapshotList()
    {
        try
        {
            var snapshots = SimSnapshots.List();
            var payload = new Dictionary<string, object?>
            {
                ["type"] = "snapshots",
                ["run_start_time"] = SimSnapshots.CurrentStartTime,
                ["snapshots"] = snapshots.ConvertAll(m => (object)new Dictionary<string, object?>
                {
                    ["id"] = m.Id,
                    ["seq"] = m.Seq,
                    ["ts"] = m.Ts,
                    ["floor"] = m.Floor,
                    ["act"] = m.Act,
                    ["post_combat"] = m.PostCombat,
                    ["has_tape"] = m.HasTape,
                    ["num_reloads"] = m.NumReloads,
                }),
            };
            SendState(JsonSerializer.Serialize(payload));
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] SendSnapshotList failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Raised for every inbound {"action":"coach"} message whose `coach`
    /// field is an object (PROTOCOL.md section 5, v5). The argument is a
    /// detached copy of that `coach` record. Runs on the SOCKET thread:
    /// marshal before touching the scene tree. A throwing handler is logged
    /// and never tears down the connection.
    /// </summary>
    public event Action<JsonElement>? CoachReceived;

    private void DeliverCoach(JsonElement root)
    {
        if (!root.TryGetProperty("coach", out JsonElement rec) || rec.ValueKind != JsonValueKind.Object)
        {
            Logger.Log("[BridgeServer] coach message ignored: 'coach' is not an object");
            return;
        }
        Action<JsonElement>? handler = CoachReceived;
        if (handler == null)
        {
            return;
        }
        try
        {
            // The document is disposed after routing; hand over a copy.
            handler(rec.Clone());
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] coach handler failed: {ex.Message}");
        }
    }

    private void DeliverAdvice(JsonElement root)
    {
        string? requestId = root.TryGetProperty("request_id", out JsonElement idEl)
            ? idEl.GetString()
            : null;

        Action<JsonElement>? callback = null;
        if (requestId != null)
        {
            lock (_adviceLock)
            {
                if (_pendingAdvice.TryGetValue(requestId, out callback))
                {
                    _pendingAdvice.Remove(requestId);
                }
            }
        }

        if (callback == null)
        {
            Logger.Log($"[BridgeServer] Dropping advice for unknown/stale request_id={requestId ?? "null"}");
            return;
        }

        try
        {
            callback(root);
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] Advice callback threw: {ex.Message}");
        }
    }

    /// <summary>
    /// Cancels the pending autopilot request (its waiter sees null) and drops
    /// every pending advice callback.
    /// </summary>
    private void AbandonAwaitedReply(string reason)
    {
        lock (_awaitLock)
        {
            if (_awaitedReply != null)
            {
                Logger.Log($"[BridgeServer] Cancelling pending action ({reason})");
                _awaitedReply.TrySetCanceled();
            }
            _awaitedReply = null;
            _awaitedRequestId = null;
        }
        lock (_adviceLock)
        {
            _pendingAdvice.Clear();
        }
    }

    private void DropClient()
    {
        lock (_clientLock)
        {
            try
            {
                _netStream?.Close();
            }
            catch (Exception)
            {
            }
            try
            {
                _client?.Close();
            }
            catch (Exception)
            {
            }
            _netStream = null;
            _client = null;
        }
    }

    /// <summary>
    /// Returns <paramref name="stateJson"/> with a top-level string field
    /// "request_id" (appended, or replaced in place if present). Any failure
    /// returns the input unchanged.
    /// </summary>
    private static string WithRequestId(string stateJson, string requestId)
    {
        try
        {
            var fields = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(stateJson);
            if (fields == null)
            {
                return stateJson;
            }
            fields["request_id"] = JsonSerializer.SerializeToElement(requestId);
            return JsonSerializer.Serialize(fields);
        }
        catch (Exception)
        {
            return stateJson;
        }
    }
}
