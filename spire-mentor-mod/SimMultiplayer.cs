// SimMultiplayer.cs -- client-side-only behaviour in co-op runs.
//
// The manifest marks the mod `affects_gameplay: false`, so the game leaves it
// out of the multiplayer handshake's mod comparison (HandshakeManager.cs,
// ModManager.GetGameplayRelevantModNameList) and the other players need not
// install it. That label is only honest if the mod never acts on the shared
// game, so in a multiplayer run everything that would is refused here:
//
//   - AI takeover (F6 / F7 / advisor autoplay) -- it would send actions over
//     the network as the local player;
//   - rewind / undo / restart room -- reloads the run as SINGLEPLAYER
//     (SimRestore.ReloadRunFromSave), which would drop out of the session;
//   - Autopilot -- starts and drives its own runs;
//   - game speed other than 1.0x.
//
// What stays on is read-only: the advisor watches the local player's
// decisions and draws its advice on this machine's HUD.

using System;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal static class SimMultiplayer
{
    /// <summary>True while a host or client co-op run is in progress
    /// (NetGameType Host/Client; a singleplayer or replay run is false).
    /// Anything unreadable answers true: refusing a takeover is always
    /// cheaper than acting on someone else's game.</summary>
    public static bool InRun
    {
        get
        {
            try
            {
                RunManager rm = RunManager.Instance;
                return rm != null && rm.IsInProgress && rm.NetService != null &&
                       rm.NetService.Type.IsMultiplayer();
            }
            catch (Exception)
            {
                return true;
            }
        }
    }

    /// <summary>
    /// Returns true (and tells the player why) when <paramref name="what"/>
    /// must not run because this is a multiplayer run.
    /// </summary>
    public static bool Refuse(string what)
    {
        if (!InRun)
        {
            return false;
        }
        Logger.Log($"[SpireMentor] {what} is off in multiplayer (the mod only gives advice there).");
        SimActionLog.Note($"{what} is off in multiplayer -- the mod only gives advice there.");
        return true;
    }

    /// <summary>On every multiplayer run start, drop whatever the AI was
    /// doing and put the speed back to normal.</summary>
    public static void Install()
    {
        RunManager.Instance.RunStarted += _ =>
        {
            if (!InRun)
            {
                return;
            }
            Logger.Log("[SpireMentor] Multiplayer run: advice only (AI takeover, rewind, autopilot and speed are off).");
            SimActionLog.Note("Multiplayer run: advice only. AI moves, rewind and speed controls are off.");
            MainFile.StopAutopilot();
            SimAdvisor.Instance.StopAutoPlayForMultiplayer();
            if (Math.Abs(SimSpeed.Desired - 1.0) > 1e-9)
            {
                SimSpeed.Apply(1.0, showVfx: false);
            }
        };
    }
}
