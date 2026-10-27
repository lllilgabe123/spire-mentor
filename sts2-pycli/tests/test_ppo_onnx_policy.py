"""Gates for the torch-free ONNX run policy (GH #380).

Parity rides the same recorded fixture the Rust tract consumer is gated on:
``sts2-rs/crates/sts2-runai/tests/fixtures/ppo_encoder_parity.jsonl`` carries
150 decisions (obs/actions/metrics) with the torch ``NetPolicy`` π/V recorded
from the **deployed Ironclad champion's** checkpoint -- so this file needs
neither torch nor a game process, and is deliberately NOT slow-marked.

The net is resolved from ``data/ppo/champions.toml`` rather than named here.
The fixture is regenerated from the champion's ``.pt`` at every promotion
(``python -m ppo.dump_parity_fixtures --ckpt ...``), so a hard-coded name goes
stale the moment a champion is promoted and the parity claim silently becomes
"this ONNX matches a *different* net's recorded probs". That is the same
hand-kept-list failure ``champions.toml`` exists to prevent, and it bit during
the GH #399 promotion: the fixture had moved to the new champion while these
constants still pointed at ``asc_it180``, giving 0.82 prob drift.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

import champions
import realgame_advisor as ra
import realgame_ppo as rp
from ppo import schema
from ppo.onnx_policy import OnnxPolicy
from ppo.rust_policy import RustPolicy
from ppo.schema import N_CAT, N_CONT, BudgetExceededError, Featurized, pad_to_fixed

_PYCLI = Path(__file__).resolve().parent.parent


def _champion(suffix: str) -> Path:
    """Deployed Ironclad champion artifact, per ``data/ppo/champions.toml``.

    The fixture is recorded from this net's ``.pt``, so the ONNX twin of
    *exactly* this champion is the one the parity claim is about. Resolving it
    from the manifest keeps the two in step across promotions.
    """
    import champions

    try:
        name = (
            champions.onnx_path("ironclad")
            if suffix == ".onnx"
            else champions.ckpt_path("ironclad")
        )
    except FileNotFoundError as e:
        # The .pt checkpoints are private (only the .onnx champions are
        # published); module-level, so the whole file skips rather than errors.
        pytest.skip(f"champion artifact not on disk: {e}", allow_module_level=True)
    if name is None:
        pytest.skip(
            "no Ironclad champion declared in data/ppo/champions.toml", allow_module_level=True
        )
    return Path(name)


_ONNX = _champion(".onnx")
_CKPT = _champion(".pt")
_FIXTURE = (
    _PYCLI.parent
    / "sts2-rs"
    / "crates"
    / "sts2-runai"
    / "tests"
    / "fixtures"
    / "ppo_encoder_parity.jsonl"
)


#: One decision that actually reaches `policy.distribution` (a reward_screen is
#: answered by `reward_screen_choice` before the net is ever consulted).
_MAP_SELECT_STATE: dict[str, Any] = {
    "type": "map_select",
    "floor": 5,
    "act": 1,
    "run_state": {
        "character": "Ironclad",
        "hp": 60,
        "max_hp": 75,
        "gold": 120,
        "ascension": 0,
        "master_deck": [{"id": "STRIKE_IRONCLAD"}],
        "relics": [{"id": "BURNING_BLOOD"}],
        "potions": [],
    },
    "nodes": [
        {"index": 0, "type": "Monster", "row": 6, "col": 1},
        {"index": 1, "type": "Shop", "row": 6, "col": 3},
    ],
    "map": {
        "row_count": 10,
        "col_count": 6,
        "current": {"col": 2, "row": 5},
        "boss": {"col": 2, "row": 15},
        "visited": [],
        "nodes": [],
    },
}


@pytest.fixture(scope="module")
def policy() -> OnnxPolicy:
    if not _ONNX.exists():
        pytest.skip(f"champion ONNX missing: {_ONNX}")
    return OnnxPolicy(str(_ONNX))


class TestParityWithRecordedTorch:
    def test_distribution_matches_recorded_torch_probs_and_value(self, policy: OnnxPolicy) -> None:
        if not _FIXTURE.exists():
            pytest.skip(f"sibling sts2-rs fixture missing: {_FIXTURE}")
        # Both fixture files: the default-schema one and the per-feature one.
        # Which of them the champion's schema selects is decided below, not by
        # a hardcoded filename -- that is what went stale when Ironclad's
        # champion started declaring `reward_upgrade`.
        rows = [
            json.loads(line)
            for path in sorted(_FIXTURE.parent.glob(f"{_FIXTURE.stem}*.jsonl"))
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) >= 100, "fixture unexpectedly small -- regenerate it"
        # Only rows recorded at the SERVED champion's schema can be scored.
        # Feeding it any other schema is the mis-serve GH #545 exists to stop:
        # a gated feature rewrites an existing `cat` slot, so the shapes match,
        # nothing raises, and the probabilities are merely wrong. Rows at
        # another schema are still pinned by the encoder gate, which needs no
        # net at all.
        champ = champions.get("ironclad")
        assert champ is not None
        served = sorted(f for f in champ.features.replace(",", " ").split() if f)
        rows = [r for r in rows if sorted(r.get("features") or []) == served]
        assert len(rows) >= 100, (
            f"only {len(rows)} fixture rows are at the champion's schema "
            f"({served}) -- re-dump with STS2_PPO_FEATURES set to match "
            f"champions.toml, or this gate scores almost nothing"
        )
        worst_p, worst_v = 0.0, 0.0
        for row in rows:
            probs, value = policy.distribution(row["obs"], row["actions"], row["metrics"])
            assert len(probs) == len(row["actions"])
            assert abs(sum(probs) - 1.0) < 1e-6
            worst_p = max(worst_p, float(np.abs(np.array(probs) - np.array(row["probs"])).max()))
            worst_v = max(worst_v, abs(value - row["value"]))
        # Same bound the Rust tract consumer passes on this fixture (gate B).
        assert worst_p < 1e-3, f"prob drift vs torch: {worst_p}"
        assert worst_v < 1e-3, f"value drift vs torch: {worst_v}"

    def test_budget_read_off_the_graph(self, policy: OnnxPolicy) -> None:
        # GH #315: budgets come from the input facts, not duplicated consts.
        assert policy.t_max == 320
        assert policy.a_max == 160
        # Against the LIVE schema, not a literal. `n_cont` was pinned at 19 and
        # `n_cat` at 10, which described a champion retired on 2026-08-28 -- so
        # the assertion silently became "the champion is stale" instead of "the
        # graph matches the encoder that feeds it". A champion exported at a
        # narrower schema than the encoder produces is exactly the mismatch
        # worth failing on. The one exception is an APPENDED slot a champion
        # predates: `C_ENCH_AMT` (GH #947, `card_mods`) is the trailing cont
        # column and both consumers slice it off a 25-cont graph, so the
        # pre-#947 width is the only narrower one allowed here.
        assert policy.n_cont in (schema.N_CONT, schema.C_ENCH_AMT)
        assert policy.n_cat == schema.N_CAT


class TestBudgetOverflow:
    @staticmethod
    def _feat(n_obs: int, n_actions: int) -> Featurized:
        # Widths come from the schema, never literals: this fixture is about
        # the T_MAX/A_MAX budget, and hardcoding them meant appending a
        # categorical slot failed here instead of anywhere that cares.
        return Featurized(
            obs_cat=np.zeros((n_obs, N_CAT), dtype=np.int64),
            obs_cont=np.zeros((n_obs, N_CONT), dtype=np.float32),
            obs_sec=np.zeros(n_obs, dtype=np.int64),
            act_cat=np.zeros((n_actions, N_CAT), dtype=np.int64),
            act_cont=np.zeros((n_actions, N_CONT), dtype=np.float32),
            act_sec=np.zeros(n_actions, dtype=np.int64),
            aux_room=np.zeros(n_actions, dtype=np.int64),
            n_actions=n_actions,
        )

    def test_too_many_actions_raises_the_named_error(self) -> None:
        with pytest.raises(BudgetExceededError):
            pad_to_fixed(self._feat(10, 161), 320, 160)

    def test_too_many_tokens_raises_the_named_error(self) -> None:
        with pytest.raises(BudgetExceededError):
            pad_to_fixed(self._feat(300, 30), 320, 160)

    def test_within_budget_pads_clean(self) -> None:
        inp = pad_to_fixed(self._feat(10, 5), 320, 160)
        assert inp["cont"].shape == (1, 320, N_CONT)
        assert inp["cat"].shape == (1, 320, N_CAT)
        assert inp["act_mask"].sum() == 5
        assert int(inp["obs_len"][0]) == 10


class TestAdvisorDegradation:
    """A raising policy must cost the ranking, never the advice (GH #315's
    103-option shop removal killed a run; the advisor must instead degrade
    to the unranked shape)."""

    def test_screen_advice_survives_a_raising_policy(self) -> None:
        class _Broken:
            def distribution(self, obs: Any, actions: Any, metrics: Any) -> Any:
                raise BudgetExceededError("decision exceeds export budget")

        policies = rp.PpoPolicyCache(None)
        policies._cache["ironclad"] = (_Broken(), None)
        state = {
            "type": "map_select",
            "floor": 5,
            "act": 1,
            "run_state": {
                "character": "Ironclad",
                "hp": 60,
                "max_hp": 75,
                "gold": 120,
                "ascension": 0,
                "master_deck": [{"id": "STRIKE_IRONCLAD"}],
                "relics": [{"id": "BURNING_BLOOD"}],
                "potions": [],
            },
            "nodes": [
                {"index": 0, "type": "Monster", "row": 6, "col": 1},
                {"index": 1, "type": "Shop", "row": 6, "col": 3},
            ],
            "map": {
                "row_count": 10,
                "col_count": 6,
                "current": {"col": 2, "row": 5},
                "boss": {"col": 2, "row": 15},
                "visited": [],
                "nodes": [],
            },
        }
        advice = ra.compute_screen_advice(state, "map_select", policies)
        assert "run-policy inference failed" in advice["note"]


class TestBackendSelection:
    def test_default_resolves_the_onnx_champion_through_rust(self) -> None:
        """The committed ONNX champion, scored by ``sts2_rs``.

        The backend moved from onnxruntime to tract so the packaged advisor
        can ship without numpy/onnxruntime (~54 MB); the *model* is
        unchanged, and ``diag_ppo_policy_parity.py`` is what shows the two
        pick the same action on recorded decisions. Asserting the type here
        is what keeps the bundle honest -- the frozen exe has no
        onnxruntime to fall back to, so a silently flipped default would
        leave it with no run policy at all.
        """
        if not _ONNX.exists():
            pytest.skip(f"champion ONNX missing: {_ONNX}")
        policy, error = rp.load_ppo_policy(None, "Ironclad")
        assert error is None
        assert isinstance(policy, RustPolicy)

    def test_env_var_forces_the_onnxruntime_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``STS2_PPO_RUST_BACKEND=0`` is the one-variable A/B.

        Without it, comparing the two backends on one machine would mean a
        rebuild, and a suspected inference difference could not be isolated
        from everything else that changed.
        """
        if not _ONNX.exists():
            pytest.skip(f"champion ONNX missing: {_ONNX}")
        monkeypatch.setenv("STS2_PPO_RUST_BACKEND", "0")
        policy, error = rp.load_ppo_policy(None, "Ironclad")
        assert error is None
        assert isinstance(policy, OnnxPolicy)

    def test_ckpt_env_outranks_the_onnx_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`STS2_WB_PPO_CKPT` is the documented way to score a candidate before
        promotion; resolving ONNX first made it silently inert, so an eval
        credited to the candidate measured the old champion instead."""
        pytest.importorskip("torch")
        from ppo.net_policy import NetPolicy

        pt = _CKPT
        if not pt.exists():
            pytest.skip(f"torch champion missing: {pt}")
        monkeypatch.setenv("STS2_WB_PPO_CKPT", str(pt))
        policy, error = rp.load_ppo_policy(None, "Ironclad")
        assert error is None
        assert isinstance(policy, NetPolicy)

    def test_missing_ckpt_env_names_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STS2_WB_PPO_CKPT", str(_PYCLI / "data" / "ppo" / "nope.pt"))
        policy, error = rp.load_ppo_policy(None, "Ironclad")
        assert policy is None
        assert error is not None and "STS2_WB_PPO_CKPT" in error

    @pytest.mark.parametrize("name", ["snapshot.pth", "run_it400.ckpt", "model.bin"])
    def test_other_torch_extensions_are_not_routed_to_onnx(self, name: str) -> None:
        """A `.pt`-only test sent every other torch spelling to onnxruntime,
        which raised InvalidProtobuf -- and the driver answered that with the
        random-legal policy while still reporting a win rate."""
        policy, error = rp.load_ppo_policy(name, "Ironclad")
        assert policy is None
        assert error is not None and "onnx" not in error.lower()

    def test_unknown_extension_refuses_to_guess(self) -> None:
        policy, error = rp.load_ppo_policy("wandb_artifact_no_suffix", "Ironclad")
        assert policy is None
        assert error is not None and "cannot tell which backend" in error

    def test_explicit_pt_model_selects_torch(self) -> None:
        pytest.importorskip("torch")
        from ppo.net_policy import NetPolicy

        pt = _CKPT
        if not pt.exists():
            pytest.skip(f"torch champion missing: {pt}")
        policy, error = rp.load_ppo_policy(str(pt), "Ironclad")
        assert error is None
        assert isinstance(policy, NetPolicy)

    def test_unknown_character_degrades_with_an_error(self) -> None:
        # A plainly bogus name, not a plausible-looking class: STS2's roster is
        # exactly Ironclad/Silent/Defect/Regent/Necrobinder, and naming a
        # non-existent character here is how one gets mistaken for a real one.
        policy, error = rp.load_ppo_policy(None, "not_a_character")
        assert policy is None
        assert error is not None and "not_a_character" in error

    def test_manifest_naming_an_absent_onnx_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The manifest resolvers raise FileNotFoundError when champions.toml
        names a file absent from the tree; load_ppo_policy promises "never
        raising" and must degrade to (None, error) instead."""
        import workbench.ppo_policy as wpp

        def _raise(character: str = "Ironclad") -> str | None:
            raise FileNotFoundError(f"manifest names an absent net for {character!r}")

        monkeypatch.setattr(wpp, "onnx_path", _raise)
        policy, error = rp.load_ppo_policy(None, "Ironclad")
        assert policy is None
        assert error is not None and "absent net" in error

    def test_onnx_load_failure_with_absent_ckpt_reports_the_onnx_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The frozen advisor exe bundles the .onnx but never the .pt the
        manifest declares: an onnxruntime failure must surface AS the onnx
        error, not as a FileNotFoundError from ckpt_path (GH #380)."""
        import workbench.ppo_policy as wpp

        bad_onnx = tmp_path / "corrupt.onnx"
        bad_onnx.write_bytes(b"not a protobuf")

        def _raise(character: str = "Ironclad") -> str | None:
            raise FileNotFoundError(f"manifest names an absent ckpt for {character!r}")

        monkeypatch.setattr(wpp, "onnx_path", lambda character="Ironclad": str(bad_onnx))
        monkeypatch.setattr(wpp, "ckpt_path", _raise)
        policy, error = rp.load_ppo_policy(None, "Ironclad")
        assert policy is None
        assert error is not None and "failed to load ONNX champion" in error

    def test_onnx_and_torch_agree_on_a_live_decision(self, policy: OnnxPolicy) -> None:
        """One end-to-end cross-backend check on a real recorded decision
        (the fixture rows already pin this at scale; this pins the two
        LOADED-by-selection backends against each other)."""
        pytest.importorskip("torch")
        if not _FIXTURE.exists():
            pytest.skip(f"sibling sts2-rs fixture missing: {_FIXTURE}")
        row = json.loads(_FIXTURE.read_text().splitlines()[0])
        pt = _CKPT
        if not pt.exists():
            pytest.skip(f"torch champion missing: {pt}")
        torch_policy, error = rp.load_ppo_policy(str(pt), "Ironclad")
        assert error is None
        p_onnx, v_onnx = policy.distribution(row["obs"], row["actions"], row["metrics"])
        p_t, v_t = torch_policy.distribution(row["obs"], row["actions"], row["metrics"])
        assert np.abs(np.array(p_onnx) - np.array(p_t)).max() < 1e-3
        assert abs(v_onnx - v_t) < 1e-3


class TestExplicitModelIsNotOptional:
    """A named `--model` is a measurement instruction. Degrading to the
    random-legal policy still finishes the sweep and still reports a win rate
    -- one the named checkpoint never produced."""

    def test_driver_cache_raises_on_an_unloadable_explicit_model(self) -> None:
        policies = rp.PpoPolicyCache("no_such_model.onnx")
        with pytest.raises(RuntimeError, match=re.escape("no_such_model.onnx")):
            policies.get("Ironclad")

    def test_driver_cache_raises_on_every_get_not_just_the_first(self) -> None:
        # The sweep drivers catch per-run exceptions and keep going, so a
        # raise that fired only on the cache miss would abort run 1 and
        # silently score runs 2..N with the random-legal policy.
        policies = rp.PpoPolicyCache("no_such_model.onnx")
        with pytest.raises(RuntimeError):
            policies.get("Ironclad")
        with pytest.raises(RuntimeError, match=re.escape("no_such_model.onnx")):
            policies.get("Ironclad")

    def test_advisor_cache_still_degrades(self) -> None:
        # strict=False: advice must render with no net at all.
        policies = rp.PpoPolicyCache("no_such_model.onnx", strict=False)
        policy, error = policies.get("Ironclad")
        assert policy is None
        assert error is not None

    def test_missing_default_champion_never_raises(self) -> None:
        # No --model given, so strict has nothing to enforce.
        policy, error = rp.PpoPolicyCache(None).get("not_a_character")
        assert policy is None
        assert error is not None


class TestDriverBudgetOverflow:
    """The driver lanes' own `distribution()` call site. The advisor got a
    degrade path for the fixed-shape budget; this one kept only the blanket
    handler that logs "PPO decision failed" and picks at RANDOM."""

    def test_budget_overflow_is_tagged_distinctly(self) -> None:
        class _OverBudget:
            def distribution(self, obs: Any, actions: Any, metrics: Any) -> Any:
                raise BudgetExceededError("161 actions > A_MAX=160")

        policies = rp.PpoPolicyCache(None)
        policies._cache["ironclad"] = (_OverBudget(), None)
        action, ann = rp.pick_noncombat_action_ppo(_MAP_SELECT_STATE, random.Random(0), policies)
        assert action is not None
        assert ann is not None and ann.get("ppo_budget_exceeded") is True


class TestSelectIndexIntegration:
    def test_greedy_pick_matches_argmax_of_onnx_probs(self, policy: OnnxPolicy) -> None:
        if not _FIXTURE.exists():
            pytest.skip(f"sibling sts2-rs fixture missing: {_FIXTURE}")
        row = json.loads(_FIXTURE.read_text().splitlines()[0])
        probs, _ = policy.distribution(row["obs"], row["actions"], row["metrics"])
        assert rp._select_index(probs, 0.0, random.Random(0)) == int(np.argmax(probs))
