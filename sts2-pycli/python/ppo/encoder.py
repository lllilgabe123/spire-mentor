"""Decision → token arrays for the run-level policy net (M-P2).

``featurize(decision)`` consumes a ``ppo.env.Decision`` (or its dict form
from a recorded JSONL episode — ``featurize_row``) and emits the
``model.Featurized`` token arrays:

- obs tokens: one STATE scalar token (carrying the fine-grained decision
  surface and the current act's boss identity), per-card DECK tokens, RELIC /
  POTION tokens, and R5b MAPNODE tokens (ego-relative + reachability + scaled
  forward-cone aggregates, plus the boss identity on the terminal boss node);
- action tokens: one per candidate, section + categorical ids + payload
  continuous features resolved per action kind (see ``_ACTION_BUILDERS``).

Event options carry semantic payload features (gold/HP deltas — plan
§3.2's anti-memorization payload) plus the ``event_id × option_idx``
categorical pair. Those deltas come from the engine's structured effect
data when the obs surfaces it (``gold_delta`` / ``hp_delta`` /
``max_hp_delta``, GH #237), falling back to regex-scraping the option's
description text only when it does not. Shop items carry
price; path options carry the destination forward cone (the fork's
repr-lab winner, causally +9.8pp in STS1).

Every unknown action kind degrades to a generic FIXED token — never a
crash — so new engine actions can't silently kill collection; the trainer's
behavior dashboard surfaces their frequency instead.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from . import mapfeat, schema, vocab
from .schema import (
    ACT_FLOOR_SCALE,
    ASC_SCALE,
    AUX_IGNORE,
    C_ACT,
    C_ACTFLOOR,
    C_ASC,
    C_CARD_GAIN,
    C_CARD_LOSS,
    C_DECKSZ,
    C_DREST,
    C_DX,
    C_DY,
    C_ENABLED,
    C_ENCH_AMT,
    C_FLOOR,
    C_GOLD,
    C_GOLD_D,
    C_HP,
    C_HP_D,
    C_MAXE,
    C_MINE,
    C_NREL,
    C_NUPG,
    C_POTION_GAIN,
    C_POTION_LOSS,
    C_PRICE,
    C_REACH,
    C_RELIC_GAIN,
    C_RELIC_LOSS,
    DECK_SCALE,
    ENCH_AMT_SCALE,
    FLOOR_SCALE,
    GOLD_SCALE,
    N_CONT,
    SEC_CARD,
    SEC_DECK,
    SEC_EVENT_OPT,
    SEC_FIXED,
    SEC_MAPNODE,
    SEC_PATH,
    SEC_POTION,
    SEC_POTION_OPT,
    SEC_RELIC,
    SEC_RELIC_OPT,
    SEC_STATE,
    Featurized,
    empty_cat,
)

# ── event payload parsing ────────────────────────────────────────────────

# Enum-form label (Neow options carry the offered relic's id as the label,
# e.g. "LOST_COFFER" — the option's real payload is the relic identity).
# Only labels already in UPPER_SNAKE form are registry-matched; prose labels
# ("Power", "Skill") would false-positive through `canonical()` otherwise.
_RE_ENUM_LABEL = re.compile(r"^[A-Z0-9_]{3,}$")

_RE_GOLD = re.compile(r"(gain|receive|lose|pay|give|offer)\s+(?:up to\s+)?(\d+)\s*gold", re.I)
_RE_DMG = re.compile(r"(?:take|lose|suffer)\s+(\d+)\s*(?:damage|hp)", re.I)
_RE_HEAL = re.compile(r"heal\s+(?:for\s+)?(\d+)", re.I)
_RE_PCT_HP = re.compile(r"(?:lose|take)\s+(\d+)%\s*(?:of\s+your\s+)?(?:max(?:imum)?\s+)?hp", re.I)
_RE_MAXHP = re.compile(r"(gain|lose)\s+(\d+)\s*max(?:imum)?\s+hp", re.I)


def event_payload(text: str, max_hp: int) -> tuple[float, float]:
    """(gold_delta, hp_delta_frac) parsed from an option label/description.

    Conservative: unmatched → zeros. Signs: gain +, lose/pay −. HP deltas
    are fractions of ``max_hp``; percent-HP costs resolve directly.
    """
    gold = 0.0
    hp = 0.0
    for verb, amt in _RE_GOLD.findall(text):
        sign = 1.0 if verb.lower() in ("gain", "receive") else -1.0
        gold += sign * float(amt)
    for amt in _RE_DMG.findall(text):
        hp -= float(amt)
    for amt in _RE_HEAL.findall(text):
        hp += float(amt)
    for verb, amt in _RE_MAXHP.findall(text):
        hp += (1.0 if verb.lower() == "gain" else -1.0) * float(amt)
    hp_frac = hp / max(1, max_hp)
    for pct in _RE_PCT_HP.findall(text):
        hp_frac -= float(pct) / 100.0
    return gold, max(-1.0, min(1.0, hp_frac))


def structured_event_payload(opt: dict[str, Any], max_hp: int) -> tuple[float, float] | None:
    """(gold_delta, hp_delta_frac) from an option's STRUCTURED engine deltas.

    GH #237: the engine already computed these numbers; prefer them over
    regex-scraping the English ``description`` (a lossy round-trip that goes
    silently dark on phrasing / game-version text churn). Returns ``None`` when
    the option carries no structured payload (pre-#237 obs — e.g. the C#
    backend or an older wheel), so the caller falls back to ``event_payload``.

    ``hp_delta_frac`` folds current-HP and max-HP changes into one signed
    fraction of ``max_hp``: a max-HP gain already bumps current HP (engine
    contract) so it is counted once via ``hp_delta``; a max-HP loss takes the
    more-negative of the current-HP drop and the ceiling drop, since
    ``lose_max_hp`` only lowers current HP once it exceeds the new ceiling.
    """
    g = opt.get("gold_delta")
    h = opt.get("hp_delta")
    mh = opt.get("max_hp_delta")
    if g is None or h is None or mh is None:
        return None
    hp_units = h if mh >= 0 else min(h, mh)
    hp_frac = max(-1.0, min(1.0, hp_units / max(1, max_hp)))
    return float(g), hp_frac


#: ``(cont slot, option key)`` for the non-gold, non-HP half of the payload.
#: Gained and lost stay separate — netting them collapses a trade to zero, and
#: two of these events exist to trade. Absent keys (an older wheel, the C#
#: backend) leave the slots at zero, which is what they meant before.
_RESOURCE_SLOTS = (
    (C_RELIC_GAIN, "relics_gained"),
    (C_RELIC_LOSS, "relics_lost"),
    (C_POTION_GAIN, "potions_gained"),
    (C_POTION_LOSS, "potions_lost"),
    (C_CARD_GAIN, "cards_gained"),
    (C_CARD_LOSS, "cards_lost"),
)


# ── surface refinement (state-token categorical) ─────────────────────────

_KIND_SURFACE = {
    "move": "path",
    "pick_card_reward": "card-reward",
    "sacrifice_card_reward": "card-reward",
    "pick_relic_reward": "relic-reward",
    "skip_relic": "relic-reward",
    "pick_boss_relic": "boss-relic",
    "smith_card": "card-smith",
    "remove_deck_card": "card-remove",
    "upgrade_deck_card": "card-upgrade",
    "transform_deck_card": "card-transform",
    "enchant_deck_card": "card-enchant",
    "grant_deck_card": "card-grant",
    "shop_buy_card": "shop",
    "shop_buy_relic": "shop",
    "shop_buy_potion": "shop",
    "shop_remove_card": "shop",
    "leave_shop": "shop",
    "rest_option": "rest",
    "event_choice": "event",
    "collect_treasure": "treasure",
    "potion_swap": "potion-swap",
}


def fine_surface(actions: list[dict[str, Any]]) -> str:
    """Finer-grained surface for the state token, from the first
    surface-defining action kind (skip/leave variants defer to a sibling)."""
    deferring = {"skip_relic", "leave_shop"}
    for a in actions:
        k = a.get("action", "")
        if k in _KIND_SURFACE and k not in deferring:
            return _KIND_SURFACE[k]
    for a in actions:
        s = _KIND_SURFACE.get(a.get("action", ""))
        if s:
            return s
    return "unknown"


# ── helpers ──────────────────────────────────────────────────────────────


def _pair(x: Any) -> tuple[str, bool]:
    """Deck-entry (card_id, upgraded) from tuple (in-process) or list (JSONL)."""
    return str(x[0]), bool(x[1])


class _Ctx:
    """Featurization context: obs lookups shared by the action builders."""

    def __init__(
        self,
        obs: dict[str, Any],
        actions: list[dict[str, Any]],
        features: frozenset[str] = schema.ACTIVE_FEATURES,
    ) -> None:
        self.obs = obs
        # Opt-in encoder experiments (`STS2_PPO_FEATURES`). Passed rather
        # than read here so a caller can featurize at the schema a fixture
        # row was RECORDED at instead of inheriting the runner's environment
        # — mirrored by `encoder.rs::featurize_with`.
        self.features = features
        self.actions = actions
        self.deck: list[tuple[str, bool]] = [_pair(x) for x in obs.get("deck") or []]
        self.potions: list[str] = list(obs.get("potions") or [])
        self.max_hp: int = int(obs.get("max_hp") or 1)
        self.ascension: int = int(obs.get("ascension") or 0)
        shop = obs.get("shop") or {}
        self.shop_cards: list[dict] = shop.get("cards") or []
        self.shop_relics: list[dict] = shop.get("relics") or []
        self.shop_potions: list[dict] = shop.get("potions") or []
        self.shop_removal_cost: int = int(shop.get("removal_cost") or 0)
        # The potion being OFFERED on a full belt (GH #252/#379). Without
        # it a `potion_swap` decision is unanswerable: the options name what
        # you would throw away, never what you would gain.
        self.pending_potion_swap: str = str(obs.get("pending_potion_swap") or "")
        # WHICH enchantment the queued pick stamps (GH #545). Without it every
        # enchant option is the same decision to the net, and Goopy is a
        # downside that inverts which card you want.
        self.pending_enchantment: str = str(obs.get("pending_enchantment") or "")
        self.event_name: str = str(obs.get("event_name") or "")
        self.event_options: list[dict] = obs.get("event_options") or []
        self.card_reward_ids: list[str] = obs.get("card_reward_option_ids") or []
        # Per-option upgrade flags aligned with `card_reward_ids`; short or
        # missing means "not upgraded" (bundle / choose-screen offerings carry
        # no upgrade state). Only read under the `reward_upgrade` feature.
        self.card_reward_upgraded: list[bool] = obs.get("card_reward_option_upgraded") or []
        self.relic_reward_ids: list[str] = obs.get("relic_reward_ids") or []
        self.boss_relics: list[str] = obs.get("boss_relic_options") or []
        self.smith: list[str] = obs.get("smith_options") or []
        self.removal: list[Any] = obs.get("card_removal_options") or []
        self.grants: list[Any] = obs.get("card_grant_options") or []
        # Per-card enchantments aligned with `deck` / `card_reward_ids` /
        # `smith` / `removal` / `grants` (GH #947): `None` or `(UPPER_SNAKE id, amount)`.
        # Short or missing means "not enchanted". Only read under `card_mods`.
        self.deck_ench: list[Any] = obs.get("deck_enchantments") or []
        self.card_reward_ench: list[Any] = obs.get("card_reward_option_enchantments") or []
        self.smith_ench: list[Any] = obs.get("smith_option_enchantments") or []
        self.removal_ench: list[Any] = obs.get("card_removal_option_enchantments") or []
        self.grants_ench: list[Any] = obs.get("card_grant_option_enchantments") or []
        self.transform_idxs: list[int] = obs.get("transform_select_deck_indices") or []
        self.enchant_idxs: list[int] = obs.get("enchant_select_deck_indices") or []
        # Master-deck positions the engine offers for `upgrade_deck_card`, in
        # offer order. NOT derivable from `deck` here: the engine's index space
        # is `Card::is_upgradable()` (curses / statuses / Quest cards excluded),
        # not merely `!upgraded` — see the branch in `_build_action_token`.
        self.upgrade_idxs: list[int] = obs.get("upgrade_select_deck_indices") or []
        # Path options: mapfeat cones aligned with move actions, in order.
        move_cols = [int(a["col"]) for a in actions if a.get("action") == "move"]
        mc = obs.get("map_coord")
        self.mapf = mapfeat.compute(
            obs.get("map"), tuple(mc) if mc is not None else None, move_cols
        )
        self._move_seen = 0

    def next_move_cone(self) -> mapfeat.OptionConeFeat | None:
        i = self._move_seen
        self._move_seen += 1
        opts = self.mapf.options
        return opts[i] if i < len(opts) else None


def _card_tok(
    ctx: _Ctx, card_id: str | None, upgraded: bool = False, price: int = 0, op: str = "none"
):
    cat = empty_cat()
    if card_id:
        cat[0] = vocab.cards().index(card_id)
        cat[1] = min(vocab.MAX_UPGRADE, 1 if upgraded else 0)
    cat[schema.CAT_DECK_OP] = vocab.deck_op_index(op)
    cont = [0.0] * N_CONT
    if price:
        cont[C_PRICE] = price * GOLD_SCALE
    return SEC_CARD, cat, cont, AUX_IGNORE


def _stamp_enchant(ctx: _Ctx, cat: list[int], cont: list[float], lst: list, i: Any) -> None:
    """Stamp a card's own enchantment onto its token (GH #947).

    The id goes into the enchantment column, the amount into ``C_ENCH_AMT``.
    A no-op unless ``card_mods`` is on, and for an unenchanted card (the pad
    row stays). A short or missing list reads as "not enchanted", like the
    upgrade flags. Mirrors ``encoder.rs::stamp_enchant``.
    """
    if schema.CARD_MODS not in ctx.features or i is None:
        return
    ent = _at(lst, int(i), "enchantment")
    if ent:
        name, amount = ent
        cat[schema.CAT_ENCHANT] = vocab.enchantments().index(str(name))
        cont[C_ENCH_AMT] = int(amount) * ENCH_AMT_SCALE


def _fixed_tok(kind: str, price: int = 0):
    cat = empty_cat()
    cat[6] = vocab.fixed_index(kind)
    cont = [0.0] * N_CONT
    if price:
        cont[C_PRICE] = price * GOLD_SCALE
    return SEC_FIXED, cat, cont, AUX_IGNORE


def _at(lst: list, i: int, what: str):
    if 0 <= i < len(lst):
        return lst[i]
    return None


def _build_action_token(ctx: _Ctx, a: dict[str, Any]):
    """(section, cat, cont, aux_room_label) for one action dict."""
    kind = a.get("action", "unknown")
    if kind == "move":
        of = ctx.next_move_cone()
        cat = empty_cat()
        cont = [0.0] * N_CONT
        aux = AUX_IGNORE
        if of is not None:
            cat[4] = of.dest_room_idx
            cont[C_DX], cont[C_DY], cont[C_REACH] = of.dx, of.dy, 1.0
            cont[C_MINE], cont[C_MAXE], cont[C_DREST] = (
                of.min_elites,
                of.max_elites,
                of.dist_rest,
            )
            aux = of.dest_room_idx  # free dest-room label
        return SEC_PATH, cat, cont, aux
    if kind == "pick_card_reward":
        idx = a.get("idx")
        if idx is None:
            return _fixed_tok("skip_card")
        cid = _at(ctx.card_reward_ids, int(idx), "card reward")
        # `roll_for_upgrade` pre-upgrades non-rare reward options in acts 2+,
        # but every champion to date was trained with the flag dropped — see
        # `ppo.schema`'s `reward_upgrade` note for why telling one the truth
        # mid-life is worse than keeping it blind.
        up = schema.REWARD_UPGRADE in ctx.features and bool(
            _at(ctx.card_reward_upgraded, int(idx), "card reward upgrade")
        )
        sec, cat, cont, aux = _card_tok(ctx, cid, up, op="reward")
        _stamp_enchant(ctx, cat, cont, ctx.card_reward_ench, idx)
        return sec, cat, cont, aux
    if kind == "pick_relic_reward":
        cat = empty_cat()
        if ctx.relic_reward_ids:
            cat[2] = vocab.relics().index(ctx.relic_reward_ids[0])
        return SEC_RELIC_OPT, cat, [0.0] * N_CONT, AUX_IGNORE
    if kind == "skip_relic":
        return _fixed_tok("skip_relic")
    if kind == "pick_boss_relic":
        cat = empty_cat()
        rid = _at(ctx.boss_relics, int(a.get("idx", 0)), "boss relic")
        if rid:
            cat[2] = vocab.relics().index(rid)
        return SEC_RELIC_OPT, cat, [0.0] * N_CONT, AUX_IGNORE
    if kind == "smith_card":
        idx = int(a.get("idx", 0))
        cid = _at(ctx.smith, idx, "smith")
        sec, cat, cont, aux = _card_tok(ctx, cid, op="smith")
        _stamp_enchant(ctx, cat, cont, ctx.smith_ench, idx)
        return sec, cat, cont, aux
    if kind == "remove_deck_card":
        idx = int(a.get("idx", 0))
        ent = _at(ctx.removal, idx, "removal")
        cid, up = _pair(ent) if ent is not None else (None, False)
        sec, cat, cont, aux = _card_tok(ctx, cid, up, op="remove")
        _stamp_enchant(ctx, cat, cont, ctx.removal_ench, idx)
        return sec, cat, cont, aux
    if kind == "upgrade_deck_card":
        # `idx` is a position in the engine's `is_upgradable()` subset
        # (`run_manager.rs:954-968` offers it, `do_upgrade_deck_card_at`
        # resolves it) — Curse / Status / Quest cards are excluded, so a raw
        # `!upgraded` scan over `deck` over-offers and every option after the
        # first curse names the wrong card (GH #399 Task 5b; Ascender's Bane
        # puts a curse in EVERY A5+ deck from floor 1). The obs therefore
        # carries the engine's own offer map, exactly as transform/enchant do.
        deck_i = _at(ctx.upgrade_idxs, int(a.get("idx", 0)), "upgrade")
        ent = _at(ctx.deck, int(deck_i), "deck") if deck_i is not None else None
        sec, cat, cont, aux = _card_tok(
            ctx, ent[0] if ent else None, bool(ent[1]) if ent else False, op="upgrade"
        )
        _stamp_enchant(ctx, cat, cont, ctx.deck_ench, deck_i)
        return sec, cat, cont, aux
    if kind == "transform_deck_card":
        deck_i = _at(ctx.transform_idxs, int(a.get("idx", 0)), "transform")
        ent = _at(ctx.deck, int(deck_i), "deck") if deck_i is not None else None
        sec, cat, cont, aux = _card_tok(
            ctx, ent[0] if ent else None, bool(ent[1]) if ent else False, op="transform"
        )
        _stamp_enchant(ctx, cat, cont, ctx.deck_ench, deck_i)
        return sec, cat, cont, aux
    if kind == "enchant_deck_card":
        deck_i = _at(ctx.enchant_idxs, int(a.get("idx", 0)), "enchant")
        ent = _at(ctx.deck, int(deck_i), "deck") if deck_i is not None else None
        sec, cat, cont, aux = _card_tok(
            ctx, ent[0] if ent else None, bool(ent[1]) if ent else False, op="enchant"
        )
        # The target's own enchantment first, then the pending id over it:
        # `CanEnchant` admits only a bare target or one already holding this
        # same stackable enchantment, so the ids agree whenever both are set
        # and the amount carries the target's existing stacks.
        _stamp_enchant(ctx, cat, cont, ctx.deck_ench, deck_i)
        if ctx.pending_enchantment:
            cat[schema.CAT_ENCHANT] = vocab.enchantments().index(ctx.pending_enchantment)
        return sec, cat, cont, aux
    if kind == "grant_deck_card":
        idx = int(a.get("idx", 0))
        ent = _at(ctx.grants, idx, "grant")
        cid, up = _pair(ent) if ent is not None else (None, False)
        sec, cat, cont, aux = _card_tok(ctx, cid, up, op="grant")
        _stamp_enchant(ctx, cat, cont, ctx.grants_ench, idx)
        return sec, cat, cont, aux
    if kind == "shop_buy_card":
        e = _at(ctx.shop_cards, int(a.get("idx", 0)), "shop card") or {}
        sec, cat, cont, aux = _card_tok(
            ctx, e.get("id"), price=int(e.get("price") or 0), op="shop_buy"
        )
        cont[C_ENABLED] = 1.0 if e.get("available") else 0.0
        return sec, cat, cont, aux
    if kind == "shop_buy_relic":
        e = _at(ctx.shop_relics, int(a.get("idx", 0)), "shop relic") or {}
        cat = empty_cat()
        if e.get("id"):
            cat[2] = vocab.relics().index(e["id"])
        cont = [0.0] * N_CONT
        cont[C_PRICE] = int(e.get("price") or 0) * GOLD_SCALE
        cont[C_ENABLED] = 1.0 if e.get("available") else 0.0
        return SEC_RELIC_OPT, cat, cont, AUX_IGNORE
    if kind == "shop_buy_potion":
        e = _at(ctx.shop_potions, int(a.get("idx", 0)), "shop potion") or {}
        cat = empty_cat()
        if e.get("id"):
            cat[3] = vocab.potions().index(e["id"])
        cont = [0.0] * N_CONT
        cont[C_PRICE] = int(e.get("price") or 0) * GOLD_SCALE
        cont[C_ENABLED] = 1.0 if e.get("available") else 0.0
        return SEC_POTION_OPT, cat, cont, AUX_IGNORE
    if kind == "shop_remove_card":
        deck_i = int(a.get("deck_idx", 0))
        ent = _at(ctx.deck, deck_i, "deck")
        sec, cat, cont, aux = _card_tok(
            ctx,
            ent[0] if ent else None,
            bool(ent[1]) if ent else False,
            price=ctx.shop_removal_cost,
            op="shop_remove",
        )
        cat[6] = vocab.fixed_index("shop_remove")
        _stamp_enchant(ctx, cat, cont, ctx.deck_ench, deck_i)
        return sec, cat, cont, aux
    if kind == "leave_shop":
        return _fixed_tok("leave_shop")
    if kind == "rest_option":
        idx = int(a.get("idx", 0))
        name = ("rest_heal", "rest_smith")[idx] if idx < 2 else "rest_other"
        tok = _fixed_tok(name)
        tok[1][7] = min(vocab.MAX_OPT_IDX - 1, idx)
        return tok
    if kind == "event_choice":
        idx = int(a.get("idx", 0))
        cat = empty_cat()
        cat[5] = vocab.events().index(ctx.event_name) if ctx.event_name else empty_cat()[5]
        cat[7] = min(vocab.MAX_OPT_IDX - 1, idx)
        cont = [0.0] * N_CONT
        opt = _at(ctx.event_options, idx, "event option") or {}
        label = str(opt.get("label") or "")
        # Identity payload: enum-form labels are relic/card ids (Neow).
        if _RE_ENUM_LABEL.match(label):
            ridx = vocab.relics().index(label)
            if ridx != vocab.relics().oov:
                cat[2] = ridx
            else:
                cidx = vocab.cards().index(label)
                if cidx != vocab.cards().oov:
                    cat[0] = cidx
        # Prefer the engine's structured deltas (GH #237); fall back to
        # scraping the label+description when the obs predates them.
        payload = structured_event_payload(opt, ctx.max_hp)
        if payload is None:
            payload = event_payload(f"{label}. {opt.get('description') or ''}", ctx.max_hp)
        gold_d, hp_d = payload
        cont[C_GOLD_D] = gold_d * GOLD_SCALE
        cont[C_HP_D] = hp_d
        cont[C_ENABLED] = 1.0 if opt.get("enabled", True) else 0.0
        # What the option costs in things that are not gold or HP. Without
        # these an option that trades a relic away or drinks a potion is
        # indistinguishable from one that does nothing.
        for slot, key in _RESOURCE_SLOTS:
            cont[slot] = float(opt.get(key) or 0.0)
        return SEC_EVENT_OPT, cat, cont, AUX_IGNORE
    if kind == "potion_swap":
        # `discard_slot: None` declines the reward outright (the potion is
        # lost, and `potions_overflow_dropped` counts it); `Some(slot)`
        # names the held potion you would throw away to make room. So the
        # option token is the DISCARDED potion — what you gain is on the
        # state token (`state_cat[3]`), identical for every option.
        slot = a.get("discard_slot")
        if slot is None:
            return _fixed_tok("decline_potion")
        cat = empty_cat()
        pid = _at(ctx.potions, int(slot), "potion slot")
        if pid:
            cat[3] = vocab.potions().index(pid)
        return SEC_POTION_OPT, cat, [0.0] * N_CONT, AUX_IGNORE
    if kind == "collect_treasure":
        return _fixed_tok("open_chest")
    if kind == "sacrifice_card_reward":
        return _fixed_tok("sacrifice")
    # Unknown action kind: generic token, never a crash (dashboarded instead).
    return _fixed_tok("none")


def act_boss_kind(obs: dict[str, Any]) -> str | None:
    """Lead monster of THIS act's pre-rolled boss, or None.

    Already present in every observation and every recorded corpus:
    `RunManager.full_map()` appends a boss meta-node carrying `kind` (the
    lead monster's python name) and `id` (the encounter id), resolved from
    `RunState::pinned_act_boss` / the C#-faithful generated room. The encoder
    has simply never read it — `mapfeat.compute` takes a boss node's room
    type and nothing else.

    Player-visible by construction (the map widget shows this label), so
    reading it does not breach `env.observe`'s clairvoyance discipline. It is
    the CURRENT act's boss only; future acts' bosses are not in the obs and
    must not be added.
    """
    m = obs.get("map")
    if not isinstance(m, dict):
        return None
    boss = m.get("boss")
    if not isinstance(boss, dict):
        return None
    kind = boss.get("kind")
    return kind if isinstance(kind, str) and kind else None


# ── main entry points ────────────────────────────────────────────────────


def featurize_row(
    obs: dict[str, Any],
    actions: list[dict[str, Any]],
    metrics: dict[str, Any],
    features: frozenset[str] = schema.ACTIVE_FEATURES,
) -> Featurized:
    """Featurize from raw (obs, actions, metrics) — the JSONL episode shape."""
    ctx = _Ctx(obs, actions, features)

    obs_cat: list[list[int]] = []
    obs_cont: list[list[float]] = []
    obs_sec: list[int] = []

    def add_obs(sec: int, cat: list[int], cont: list[float]) -> None:
        obs_cat.append(cat)
        obs_cont.append(cont)
        obs_sec.append(sec)

    # State scalar token — carries the fine decision surface and the identity
    # of the boss this act's map terminates in. The boss is fixed at map
    # generation, so it is known from the act's first floor: every card
    # reward, removal, upgrade and transform can be judged against the fight
    # the deck is actually being built for, instead of against the average
    # boss. Pad row when the engine could not resolve one.
    state_cat = empty_cat()
    state_cat[8] = vocab.surface_index(fine_surface(actions))
    state_cat[9] = ctx.mapf.boss_idx
    # Pad on every other surface, so this is inert for a net that never sees
    # a `potion_swap`: the potion embedding's pad row contributes nothing.
    if ctx.pending_potion_swap:
        state_cat[3] = vocab.potions().index(ctx.pending_potion_swap)
    # Pad row on every other surface, exactly like the potion above.
    if ctx.pending_enchantment:
        state_cat[schema.CAT_ENCHANT] = vocab.enchantments().index(ctx.pending_enchantment)
    state_cont = [0.0] * N_CONT
    state_cont[C_HP] = int(metrics.get("hp") or 0) / max(1, ctx.max_hp)
    state_cont[C_GOLD] = int(metrics.get("gold") or 0) * GOLD_SCALE
    state_cont[C_ACT] = float(metrics.get("act_index") or 0)
    state_cont[C_DECKSZ] = len(ctx.deck) * DECK_SCALE
    state_cont[C_NUPG] = int(metrics.get("num_upgraded") or 0) * DECK_SCALE
    state_cont[C_NREL] = int(metrics.get("num_relics") or 0) * 0.1
    state_cont[C_FLOOR] = int(metrics.get("total_floor") or 0) * FLOOR_SCALE
    state_cont[C_ACTFLOOR] = int(metrics.get("act_floor") or 0) * ACT_FLOOR_SCALE
    state_cont[C_ASC] = ctx.ascension * ASC_SCALE
    add_obs(SEC_STATE, state_cat, state_cont)

    # Deck / relic / potion tokens.
    for i, (cid, up) in enumerate(ctx.deck):
        cat = empty_cat()
        cat[0] = vocab.cards().index(cid)
        cat[1] = min(vocab.MAX_UPGRADE, 1 if up else 0)
        cont = [0.0] * N_CONT
        _stamp_enchant(ctx, cat, cont, ctx.deck_ench, i)
        add_obs(SEC_DECK, cat, cont)
    for rid in obs.get("relics") or []:
        cat = empty_cat()
        cat[2] = vocab.relics().index(rid)
        add_obs(SEC_RELIC, cat, [0.0] * N_CONT)
    for pid in obs.get("potions") or []:
        cat = empty_cat()
        cat[3] = vocab.potions().index(pid)
        add_obs(SEC_POTION, cat, [0.0] * N_CONT)

    # Map node tokens (R5b). The terminal boss node also carries the boss's
    # identity; every other node carries the pad row.
    for nf in ctx.mapf.nodes:
        cat = empty_cat()
        cat[4] = nf.room_idx
        cat[9] = nf.boss_idx
        cont = [0.0] * N_CONT
        cont[C_DX], cont[C_DY], cont[C_REACH] = nf.dx, nf.dy, nf.reachable
        cont[C_MINE], cont[C_MAXE], cont[C_DREST] = (
            nf.min_elites,
            nf.max_elites,
            nf.dist_rest,
        )
        add_obs(SEC_MAPNODE, cat, cont)

    # Action tokens.
    act_cat: list[list[int]] = []
    act_cont: list[list[float]] = []
    act_sec: list[int] = []
    aux_room: list[int] = []
    for a in actions:
        sec, cat, cont, aux = _build_action_token(ctx, a)
        act_sec.append(sec)
        act_cat.append(cat)
        act_cont.append(cont)
        aux_room.append(aux)

    return Featurized(
        obs_cat=np.array(obs_cat, dtype=np.int64),
        obs_cont=np.array(obs_cont, dtype=np.float32),
        obs_sec=np.array(obs_sec, dtype=np.int64),
        act_cat=np.array(act_cat, dtype=np.int64),
        act_cont=np.array(act_cont, dtype=np.float32),
        act_sec=np.array(act_sec, dtype=np.int64),
        aux_room=np.array(aux_room, dtype=np.int64),
        n_actions=len(act_sec),
    )


def featurize(decision) -> Featurized:  # type: ignore[no-untyped-def]
    """Featurize a live ``ppo.env.Decision``."""
    return featurize_row(decision.obs, decision.actions, decision.metrics)
