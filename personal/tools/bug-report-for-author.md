# Fight advice stops in co-op when a multiplayer-only v0.111.0 card is in a pile (`UnsupportedStateError: MIDNIGHT`)

**Game:** v0.111.0 (public-beta) · **Spire Mentor:** 0.7.2 (Workshop build, packaged SimAdvisor) · **Mode:** 2-player co-op

## What happens
After picking up **Midnight** (MP-only Ironclad rare) in a co-op run, every combat advice request fails, so the HUD shows no fight advice for as long as Midnight sits in hand, draw or discard. Map, reward and event advice keep working.

`advisor_stdout.log`:
```
ERROR __main__: advice_request failed: $.deck.discard_pile[4]: MIDNIGHT
sts2_rs.UnsupportedStateError: $.deck.discard_pile[4]: MIDNIGHT
INFO __main__: advice #a418: best=None p_win=None (0.0s) [advisor error: $.deck.discard_pile[4]: MIDNIGHT]
```
95 failures in one session, across hand, draw and discard piles.

## Cause
`sts2-wire/src/state_dict.rs` `card_id_from_python_name` searches only `CardId::all()` (the v0.107.1 universe). Unlike `power_kind_from_python_name` and `potion_kind_from_python_name` just above it, it doesn't chain `CardId::V0111_ONLY`. So every v0.111.0-only card (GH #793) fails to parse in a live state, even though `resolve_card_name` and the card specs already know them. This is still the case on `main` as of 2026-10-22 23:36 UTC.

## Fix (tested)
```rust
pub fn card_id_from_python_name(s: &str) -> Option<CardId> {
    let all = || CardId::all().iter().chain(CardId::V0111_ONLY.iter()).copied();
    if let Some(c) = all().find(|k| k.to_python_name() == s) {
        return Some(c);
    }
    all().find(|k| {
        let canonical = k.to_python_name();
        canonical.strip_suffix("_CARD") == Some(s)
    })
}
```
Plus a unit test that every `CardId::V0111_ONLY` id round-trips through `card_id_from_python_name`.

- `cargo nextest run --workspace`: 8085/8086 pass. The one failure, `sts2-workbench::game_version_smoke capabilities_list_the_supported_builds`, is unrelated: `data/game/v0.111.0` isn't present in a public checkout.
- A rebuilt `sts2_rs.pyd`, dropped into the packaged SimAdvisor, passes `--smoke`.
- In a 2-instance co-op test with Midnight in the host's deck: 0 advisor errors, 27 pieces of advice including MCTS combat lines.

## Related co-op notes (optional)
These come from running the mod client-side in co-op with friends who don't have it:
- With `affects_gameplay: false`, the mod passes the handshake against unmodded peers. That only holds if nothing acts on the shared game. I gated AI takeover (F6/F7), rewind/undo/restart room, autopilot and game speed on `NetService.Type.IsMultiplayer()`, leaving advice-only in co-op.
- `SimAdvisor.EndTurnAsAi` calls `PlayerCmd.EndTurn`, which in a networked game only marks the turn ended locally. `NEndTurnButton` enqueues `EndPlayerTurnAction` through `ActionQueueSynchronizer` instead. This doesn't matter while takeover is single-player only.
- The combat engine models a single player (`state.allies` unused). On one A9 co-op run, the advisor's average fight-start `p_win` was 0.47, and all 8 fights were won.
- `SimCoachPanel.Place()` calls `GetViewportRect()` before the deferred `AddChild` of its layer on the first `Show` ("Condition !is_inside_tree()"). Guarding on `IsInsideTree()` fixes it.

A full patch of these changes is attached (`spire-mentor-changes.patch`, against the 2026-10-22 morning `main`).
