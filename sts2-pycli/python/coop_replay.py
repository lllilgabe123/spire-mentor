"""Replay recorded co-op fights in the engine, decision by decision, and diff.

Step 6 of ``docs/plans/2026-10-26-coop-multiseat-engine.md``. The advisor's
workbench run log (``realgame_workbench_log``) stores, per decision, the board
the mod sent (headless ``combat_play`` shape, every teammate's whole seat
included) and the game's own combat-log rows since the previous decision
(``combat_log_raw``). Those rows say who played which card at what, in the
order the game resolved them (``card_play_started`` rows carry ``owner``,
``target_index`` and ``play_index``).

For each pair of consecutive combat boards ``(a, b)`` this builds a whole co-op
fight from ``a`` (``$.coop_full``), plays every seat's recorded cards in the
logged order -- ours through ``apply``, a teammate's through
``apply_as_ally`` -- ends the turn when the log shows the enemies acting, and
diffs the result against ``b``. Each pair starts from the recorded board, so a
divergence does not carry into the next one.

Every pick a player made mid-combat is logged too (``player_choice`` rows,
newer mods): each seat's choices -- in a play, at a turn start -- are answered
from them in order, ours from the open choice, a teammate's through the
engine's ``teammate_chooser``. A teammate's pick with no row is left to the
engine (counted as ``unanswered``).

Potions are drunk from the logged seat's belt (a teammate's comes with newer
mods; for older boards the potions the log shows them drinking are put on it).

Usage::

    python -m coop_replay <workbench-run-log.jsonl> [...]

Prints one JSON line per replayed pair and a summary. Pairs the replay cannot
drive yet (our choice screen with no logged pick, a potion not on the drinker's
belt, an unknown owner)
are counted as skipped with the reason, never guessed at.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from sts2_hint.adapter import translate


def _norm(name: Any) -> str:
    """A monster / card id, compared loosely: ``Nibbit`` == ``NIBBIT``, and the
    engine wire's ``STAMPEDE_CARD`` (a card named like a power) == the game's
    ``STAMPEDE``, and the adapter's ``STATUS:WOUND`` == ``WOUND``."""
    s = str(name or "").upper()
    s = s.removeprefix("STATUS:").removesuffix("_CARD")
    return s.replace("_", "")


def load_states(path: Path) -> list[dict[str, Any]]:
    """The run log's combat boards, in order (``type == "state"`` records
    whose data is a ``combat_play`` decision).

    The rows of any other decision logged in between (a card-select screen, a
    reward) are carried onto the next board, and that board is marked
    ``interrupted``: the replay cannot answer the screen, so it skips the
    pair rather than miss what the screen did.
    """
    out: list[dict[str, Any]] = []
    carried: list[dict[str, Any]] = []
    interrupted = False
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A log still being written ends in a half line.
                continue
            if rec.get("type") != "state":
                continue
            data = rec.get("data") or {}
            rows = data.get("combat_log_raw") or []
            if data.get("decision") == "combat_play":
                data["combat_log_raw"] = carried + list(rows)
                data["interrupted"] = interrupted
                carried, interrupted = [], False
                out.append(data)
            else:
                carried.extend(rows)
                interrupted = True
    return out


def full_fight_wire(board: dict[str, Any]) -> dict[str, Any]:
    """``board`` as the engine wire of a whole co-op fight in which the
    teammates' remaining plays come from the log, not the engine's policy."""
    if board.get("coop_rng"):
        # The run's dice at this board (newer mods): shuffles and intents
        # replay exactly instead of from a default seed.
        board = {**board, "rng": board["coop_rng"]}
    wire = translate(board)
    wire["coop_full"] = True
    # Our place in the run's player order: the teammates ahead of us set up,
    # draw and end their turn first, from the shuffle stream the seats share.
    if board.get("seat_index"):
        wire["coop_seat_index"] = board["seat_index"]
    for ally in wire.get("allies") or []:
        ally["ended_turn"] = True
    return wire


def _hand_index(hand: list[dict[str, Any]], card: str, upgraded: bool) -> int | None:
    for i, c in enumerate(hand):
        if _norm(c.get("id")) == _norm(card) and bool(c.get("upgraded")) == upgraded:
            return i
    for i, c in enumerate(hand):
        if _norm(c.get("id")) == _norm(card):
            return i
    return None


def _target(wire: dict[str, Any], row: dict[str, Any], board: dict[str, Any]) -> int | None:
    """The engine's target index for a logged play.

    By ``combat_id`` when the mod sent it: the target's position in ``board``'s
    enemy list, which is the engine's encounter order (C# drops a killed enemy
    from its list before the row is read, so the row's own ``target_index``
    can be stale). Else the row's index when the monster there matches, else
    the first living enemy of that kind.
    """
    combat_id = (row.get("target") or {}).get("combat_id")
    if combat_id is not None:
        for i, e in enumerate(board.get("enemies") or []):
            if e.get("combat_id") == combat_id:
                return i
    index = row.get("target_index")
    if index is None:
        return None
    want = _norm((row.get("target") or {}).get("id"))
    encounter = wire.get("encounter") or []
    if 0 <= index < len(encounter) and (not want or _norm(encounter[index].get("kind")) == want):
        return index
    for i, e in enumerate(encounter):
        if not e.get("is_dead") and _norm(e.get("kind")) == want:
            return i
    return index


def observables(wire: dict[str, Any]) -> dict[str, Any]:
    """What a replayed pair is judged on: every seat's HP, block and hand, our
    energy, and every LIVING enemy's HP and block, in order (C# drops a killed
    enemy from its list where the engine keeps the corpse in place, so the
    positions of the dead are not comparable)."""
    out: dict[str, Any] = {
        "player.hp": wire["player"]["hp"],
        "player.block": wire["player"]["block"],
        "player.energy": wire["player"]["energy"],
        "player.hand": sorted(_norm(c.get("id")) for c in wire["deck"]["hand"]),
    }
    for i, a in enumerate(wire.get("allies") or []):
        out[f"allies[{i}].hp"] = a["hp"]
        out[f"allies[{i}].block"] = a["block"]
        if "hand" in a:
            out[f"allies[{i}].hand"] = sorted(_norm(c.get("id")) for c in a["hand"])
    living = [e for e in wire.get("encounter") or [] if e["hp"] > 0 and not e.get("is_dead")]
    out["encounter.living"] = len(living)
    for i, e in enumerate(living):
        out[f"encounter[{i}].hp"] = e["hp"]
        out[f"encounter[{i}].block"] = e["block"]
    return out


def _choice_selection(recorded: dict[str, Any], options: list[dict[str, Any]]) -> list[int] | None:
    """The option indexes a logged ``player_choice`` row picked, or None when
    they cannot be told apart. Cards match by id (upgraded first), each option
    at most once; an ``Index`` choice names its options by position (-1, a
    skip, names none)."""
    if recorded.get("choice_type") == "Index":
        picks = [i for i in recorded.get("indexes") or [] if i != -1]
        return picks if all(0 <= i < len(options) for i in picks) else None
    if recorded.get("choice_type") == "Player":
        return None
    used: set[int] = set()
    for card in recorded.get("cards") or []:
        want, upgraded = _norm(card.get("id")), bool(card.get("upgraded"))
        same = [i for i, o in enumerate(options) if i not in used and _norm(o.get("id")) == want]
        exact = [i for i in same if bool(options[i].get("upgraded")) == upgraded]
        if not same:
            return None
        used.add((exact or same)[0])
    return sorted(used)


def _choice_action(
    recorded: dict[str, Any], choice: dict[str, Any], actions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The ``choose`` action making a logged pick on the open ``choice``.

    One of ``actions`` when it is there; else, within the choice's
    ``min_select`` / ``max_select``, a ``choose`` with the whole selection:
    the engine lists a multi-pick's single picks only (Gambler's Brew may
    discard any number of cards at once)."""
    selection = _choice_selection(recorded, choice.get("cards") or [])
    if selection is None:
        return None
    for act in actions:
        if sorted(act.get("selection") or []) == selection:
            return act
    lo, hi = choice.get("min_select"), choice.get("max_select")
    if isinstance(lo, int) and isinstance(hi, int) and lo <= len(selection) <= hi:
        return {
            "action": "choose",
            "index": selection[0] if selection else 0,
            "selection": selection,
        }
    return None


def replay_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Replay what happened between boards ``a`` and ``b``; see the module doc."""
    import sts2_rs

    me = a.get("net_id")
    seats = {str(al.get("net_id")): i for i, al in enumerate(a.get("allies") or [])}
    if me is None or not seats:
        return {"skipped": "not a co-op board with net ids"}
    if any("draw_pile" not in al for al in a.get("allies") or []):
        return {"skipped": "teammate piles missing (older mod)"}
    rows = b.get("combat_log_raw") or []
    # Every player's picks, in the order each made them (a player's choice
    # ids run in sequence). Our own screens show up as decisions of their
    # own; with our picks logged the pair can still be replayed.
    picks: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        if row.get("kind") == "player_choice":
            picks[str(row.get("owner"))].append(row)
    if b.get("interrupted") and not picks[str(me)]:
        return {"skipped": "a decision screen in between"}
    owner_of = {i: net for net, i in seats.items()}
    answered = {"choices": 0, "unanswered": 0}
    # The cards just picked on a multi-pick choice, by owner. The engine takes
    # a multi-pick one card at a time and reopens it (minus the cards taken)
    # until `[]` closes it; the game logs the whole pick at once.
    closing: dict[str, list[int]] = {}

    def answer(owner: str, choice: dict[str, Any], actions: list[Any]) -> tuple[Any, bool]:
        """The action for ``owner``'s open choice, and whether it only closes
        the multi-pick answered before (no logged pick of its own)."""
        taken = closing.pop(owner, None)
        if taken is not None:
            offered = [act.get("selection") or [] for act in actions]
            if [] in offered and not any([i] in offered for i in taken):
                return actions[offered.index([])], True
        queue = picks[owner]
        pick = _choice_action(queue.popleft(), choice, actions) if queue else None
        if pick is not None and pick.get("selection") and (choice.get("max_select") or 1) > 1:
            closing[owner] = pick["selection"]
        return pick, False

    def teammate_chooser(
        ally: int, choice: dict[str, Any], actions: list[Any]
    ) -> dict[str, Any] | None:
        pick, closes = answer(owner_of.get(ally, ""), choice, actions)
        if not closes:
            answered["choices" if pick is not None else "unanswered"] += 1
        return pick

    def settle_ours() -> str | None:
        """Answer our open choices from the log; the skip reason if we can't."""
        while (acts := eng.legal_actions()) and acts[0].get("action") == "choose":
            pick, closes = answer(str(me), eng.pending_choice() or {}, acts)
            if pick is None:
                return "a choice screen"
            eng.apply(pick, teammate_chooser=teammate_chooser)
            answered["choices"] += 0 if closes else 1
        return None

    wire = full_fight_wire(a)
    _carry_teammate_potions(wire, rows, seats)
    eng = sts2_rs.state_from_dict(wire)
    ended = False
    played = potions = 0
    for row in rows:
        kind = row.get("kind")
        if row.get("side") == "enemy" and not ended:
            eng.apply({"action": "end_turn"}, teammate_chooser=teammate_chooser)
            ended = True
            if (why := settle_ours()) is not None:
                return {"skipped": why}
        if ended or kind not in ("card_play_started", "potion"):
            continue
        if kind == "card_play_started" and (row.get("auto_played") or row.get("play_index")):
            continue
        owner = str(row.get("owner"))
        now = sts2_rs.state_to_dict(eng)
        if owner == str(me):
            hand, belt = now["deck"]["hand"], now["player"].get("potions") or []

            def apply(action: dict[str, Any]) -> None:
                eng.apply(action, teammate_chooser=teammate_chooser)
        elif owner in seats:
            seat = seats[owner]
            hand = now["allies"][seat].get("hand") or []
            belt = now["allies"][seat].get("potions") or []

            def apply(action: dict[str, Any], seat: int = seat) -> None:
                eng.apply_as_ally(seat, action, teammate_chooser=teammate_chooser)
        else:
            return {"skipped": f"unknown owner {owner}"}
        if kind == "potion":
            slot = next(
                (i for i, p in enumerate(belt) if _norm(p) == _norm(row.get("potion"))), None
            )
            if slot is None:
                return {"skipped": f"{row.get('potion')} not in {owner}'s belt"}
            apply({"action": "potion", "slot": slot, "target_index": _target(now, row, a)})
            potions += 1
        else:
            idx = _hand_index(hand, row.get("card"), bool(row.get("upgraded")))
            if idx is None:
                return {"skipped": f"{row.get('card')} not in {owner}'s hand"}
            apply({"action": "play", "card_index": idx, "target_index": _target(now, row, a)})
            played += 1
        if (why := settle_ours()) is not None:
            return {"skipped": why}

    got = observables(sts2_rs.state_to_dict(eng))
    want = observables(translate(b))
    diffs = {k: [got.get(k), want[k]] for k in want if got.get(k) != want[k]}
    return {
        "played": played,
        "potions": potions,
        "ended_turn": ended,
        **answered,
        "diffs": diffs,
    }


def _carry_teammate_potions(
    wire: dict[str, Any], rows: list[dict[str, Any]], seats: dict[str, int]
) -> None:
    """A board from before the mod sent a teammate's belt: give them the
    potions the log shows them drinking, so the drinks can be replayed."""
    drunk: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        seat = seats.get(str(row.get("owner")))
        if row.get("kind") == "potion" and seat is not None:
            drunk[seat].append(row.get("potion"))
    for seat, ids in drunk.items():
        wire["allies"][seat].setdefault("potions", ids)


def replay_log(path: Path, seat_index: int | None = None) -> list[dict[str, Any]]:
    """Replay every pair of ``path``'s boards. ``seat_index`` stands in for the
    boards' own when they carry none (recorded before the mod sent it)."""
    states = load_states(path)
    if seat_index is not None:
        for s in states:
            s.setdefault("seat_index", seat_index)
    results = []
    for i, (a, b) in enumerate(itertools.pairwise(states)):
        if a.get("floor") != b.get("floor"):
            continue
        res = replay_pair(a, b)
        res.update(log=path.name, pair=i, floor=a.get("floor"), round=a.get("round"))
        results.append(res)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument(
        "--seat-index",
        type=int,
        help="our place in the run's player order, for logs recorded before the mod sent it",
    )
    args = ap.parse_args(argv)
    results = [r for p in args.logs for r in replay_log(p, args.seat_index)]
    for r in results:
        print(json.dumps(r))
    skipped = Counter(r["skipped"] for r in results if "skipped" in r)
    replayed = [r for r in results if "skipped" not in r]
    fields = Counter(k for r in replayed for k in r["diffs"])
    print(
        json.dumps(
            {
                "summary": {
                    "pairs": len(results),
                    "replayed": len(replayed),
                    "in_sync": sum(1 for r in replayed if not r["diffs"]),
                    "choices": sum(r["choices"] for r in replayed),
                    "unanswered": sum(r["unanswered"] for r in replayed),
                    "skipped": dict(skipped),
                    "diverging_fields": dict(fields.most_common()),
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
