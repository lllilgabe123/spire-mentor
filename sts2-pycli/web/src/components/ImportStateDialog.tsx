import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { encodePositionHash } from "../simlab/positionHash";
import type { ImportFidelity, ImportLane, ImportPreview } from "../api";

/**
 * Modal for the live-state import: paste what the mod copied, see what it is,
 * then rebuild it.
 *
 * The parse and the rebuild are two separate calls on purpose. Parsing is
 * free and touches nothing, so the dialog can say WHICH run this is and what
 * the import will and will not reproduce before the user gives up the run
 * they currently have. It debounces the preview rather than parsing on every
 * keystroke — a pasted advisor log can be megabytes.
 *
 * Closes on backdrop click or Escape, except mid-import: a rebuild replays
 * the position through the engine, and dismissing would remove the only sign
 * that anything is happening.
 */
export default function ImportStateDialog({
  onImported,
  onClose,
  liveGame,
  autoLive = false,
}: {
  onImported: (fidelity: ImportFidelity) => void;
  onClose: () => void;
  /** Fetches the connected live game's current position (the same message
   *  Copy state yields); offered as a button when set. */
  liveGame?: () => Promise<string>;
  /** Fetch from `liveGame` as soon as the dialog opens. */
  autoLive?: boolean;
}) {
  const [text, setText] = useState("");
  const [fetchingLive, setFetchingLive] = useState(false);
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [parsing, setParsing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [lane, setLane] = useState<ImportLane>("auto");
  // Guards against an out-of-order preview landing after a newer one: the
  // parse is debounced but the responses are not ordered.
  const seq = useRef(0);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape" && !busy) { e.preventDefault(); onClose(); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, busy]);

  useEffect(() => {
    const payload = text.trim();
    if (!payload) { setPreview(null); setError(null); return; }
    const mine = ++seq.current;
    setParsing(true);
    const t = setTimeout(() => {
      api.live
        .importPreview(payload)
        .then((p) => {
          if (mine !== seq.current) return;
          setPreview(p);
          setError(null);
          setLane(p.has_combat ? "combat" : "run");
        })
        .catch((e: Error) => {
          if (mine !== seq.current) return;
          setPreview(null);
          setError(e.message || "could not read that paste");
        })
        .finally(() => { if (mine === seq.current) setParsing(false); });
    }, 350);
    return () => clearTimeout(t);
  }, [text]);

  const pasteFromClipboard = useCallback(async () => {
    try {
      setText(await navigator.clipboard.readText());
    } catch {
      // Clipboard read needs permission and a secure context; over plain HTTP
      // on the LAN it simply is not available. Say so instead of failing mute.
      setError("this browser will not read the clipboard here — paste with Ctrl+V");
    }
  }, []);

  const fetchFromLiveGame = useCallback(async () => {
    if (!liveGame) return;
    setFetchingLive(true);
    try {
      setText(await liveGame());
    } catch (e) {
      // 404 = the advisor has seen no decision yet (or the run just ended).
      const msg = (e as Error).message || "";
      setError(/\b404\b|no_state/.test(msg)
        ? "the live game has no position yet — it sends one at each combat turn or choice screen"
        : `could not read the live game: ${msg}`);
    } finally {
      setFetchingLive(false);
    }
  }, [liveGame]);

  // Once per open: a later re-fetch would overwrite what the user is reading.
  const autoFetched = useRef(false);
  useEffect(() => {
    if (!autoLive || autoFetched.current || !liveGame) return;
    autoFetched.current = true;
    void fetchFromLiveGame();
  }, [autoLive, liveGame, fetchFromLiveGame]);

  async function doImport() {
    if (!preview || busy) return;
    setBusy(true);
    try {
      // The rebuild reports things the PARSE could not know: a map that
      // disagrees with the paste, a refused RNG restore, a card that could not
      // be upgraded. Hand them up — the preview only covered the paste.
      const step = await api.live.importState(text.trim(), lane);
      onImported(step.import);
    } catch (e) {
      setError((e as Error).message || "could not rebuild that state");
    } finally {
      setBusy(false);
    }
  }

  /** The fight only, into the sandbox: nothing here is replaced. */
  async function openInSimlab() {
    if (!preview?.has_combat || busy) return;
    setBusy(true);
    try {
      const r = await api.simlab.importPosition(text.trim());
      window.open(`/simlab.html#p=${encodePositionHash(r.position)}`, "_blank", "noopener");
    } catch (e) {
      setError((e as Error).message || "SimLab could not read that state");
    } finally {
      setBusy(false);
    }
  }

  const close = busy ? () => {} : onClose;
  const run = preview?.run;

  return (
    <div role="dialog" aria-label="Import state" onClick={close}
      style={{ position: "fixed", inset: 0, zIndex: 50, background: "oklch(20% 0.02 255 / .55)",
        display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
      <div onClick={(e) => e.stopPropagation()}
        style={{ background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 10,
          width: "min(760px, 94vw)", maxHeight: "86vh", overflow: "auto",
          boxShadow: "0 12px 40px oklch(30% 0.04 255 / .4)" }}>
        <div className="mono" style={{ display: "flex", alignItems: "center",
          justifyContent: "space-between", gap: 12, padding: "12px 16px",
          borderBottom: "1px solid var(--border)", position: "sticky", top: 0, background: "var(--bg)" }}>
          {busy ? <b aria-live="polite">rebuilding the position…</b> : <b>Import state</b>}
          <button onClick={close} aria-label="Close" disabled={busy}
            style={{ padding: "4px 10px", border: "1px solid var(--border)", borderRadius: 4,
              cursor: busy ? "default" : "pointer", opacity: busy ? 0.5 : 1 }}>
            close (Esc)
          </button>
        </div>

        <div style={{ padding: 16, display: "grid", gap: 12 }}>
          <div className="mono" style={{ color: "var(--muted)", fontSize: 12, lineHeight: 1.5 }}>
            In the game: <b>Copy state</b> on the SpireMentor strip, or <b>Ctrl+Shift+C</b>.
            Then paste it here. An advisor log line, or a whole log, works too —
            the newest position in it wins.
          </div>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder='{"type":"advice_request","advice_for":"combat_action", …}'
            spellCheck={false}
            disabled={busy}
            className="mono"
            style={{ width: "100%", minHeight: 120, resize: "vertical", fontSize: 11,
              padding: 8, borderRadius: 6, border: "1px solid var(--border)",
              background: "var(--bg)", color: "inherit" }}
          />

          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <button onClick={pasteFromClipboard} disabled={busy} className="mono"
              style={{ padding: "4px 10px", border: "1px solid var(--border)", borderRadius: 4 }}>
              paste from clipboard
            </button>
            {liveGame && (
              <button onClick={() => void fetchFromLiveGame()} disabled={busy || fetchingLive} className="mono"
                title="the position the connected game is at right now (no copy-paste)"
                style={{ padding: "4px 10px", border: "1px solid var(--border)", borderRadius: 4 }}>
                {fetchingLive ? "reading the game…" : "🎮 from live game"}
              </button>
            )}
            {parsing &&<span className="mono" style={{ color: "var(--muted)", fontSize: 12 }}>reading…</span>}
          </div>

          {error && (
            <div className="mono" role="alert"
              style={{ fontSize: 12, lineHeight: 1.5, padding: 10, borderRadius: 6,
                border: "1px solid var(--border)", color: "var(--bad, #d66)" }}>
              {error}
            </div>
          )}

          {run && preview && (
            <div className="mono" style={{ fontSize: 12, lineHeight: 1.6, padding: 12,
              borderRadius: 6, border: "1px solid var(--border)" }}>
              <div>
                <b>{run.character}</b> · seed <b>{run.seed}</b> · A{run.ascension} ·
                {" "}act {run.act}, floor {run.floor}
              </div>
              <div style={{ color: "var(--muted)" }}>
                {run.hp}/{run.max_hp} HP · {run.gold} gold · {run.deck_size} cards ·
                {" "}{run.relics.length} relics · on the <b>{preview.surface}</b> screen
              </div>

              <div style={{ marginTop: 10, display: "flex", gap: 12, flexWrap: "wrap" }}>
                {preview.lanes.includes("combat") && (
                  <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <input type="radio" checked={lane === "combat"} disabled={busy}
                      onChange={() => setLane("combat")} />
                    resume the fight (exact hand, piles and intents)
                  </label>
                )}
                <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <input type="radio" checked={lane === "run"} disabled={busy}
                    onChange={() => setLane("run")} />
                  rebuild the run (map, deck, shops)
                </label>
              </div>

              <div style={{ marginTop: 10, color: "var(--muted)" }}>
                exact: {preview.fidelity.exact.join(", ")}
                {preview.fidelity.rng_restored
                  ? " · RNG cursors restored, so the next reward/shop/event roll matches your run"
                  : " · no RNG cursors: downstream rolls will NOT match your run"}
              </div>
              {preview.fidelity.warnings.length > 0 && (
                <ul style={{ margin: "8px 0 0", paddingLeft: 18, color: "var(--warn, #c90)" }}>
                  {preview.fidelity.warnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              )}
            </div>
          )}

          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
            {preview?.has_combat && (
              <button onClick={openInSimlab} disabled={busy} className="mono"
                title="load this fight into the SimLab sandbox in a new tab — the run you have open stays as it is"
                style={{ padding: "6px 14px", borderRadius: 4, border: "1px solid var(--border)",
                  cursor: busy ? "default" : "pointer", opacity: busy ? 0.5 : 1 }}>
                open in SimLab instead ↗
              </button>
            )}
            <button onClick={doImport} disabled={!preview || busy} className="mono"
              style={{ padding: "6px 14px", borderRadius: 4, border: "1px solid var(--border)",
                cursor: preview && !busy ? "pointer" : "default",
                opacity: preview && !busy ? 1 : 0.5 }}>
              {busy ? "rebuilding…" : "import (replaces the current run)"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
