"""``RustPolicy`` serves a champion at ITS OWN encoder schema (GH #545).

The Rust side parses ``STS2_PPO_FEATURES`` **once per process**
(``Features::active``'s ``OnceLock``), so a process holding one policy per
character -- which ``realgame_ppo.PpoPolicyCache`` and the workbench backend
both do -- cannot express "on for Ironclad, off for the other four". The
#545 campaign creates exactly that state: it trains Ironclad with
``reward_upgrade`` while the four incumbents were trained without it.

The mismatch is silent. A gated feature rewrites an **existing** ``cat`` slot
instead of appending a column, so the tensor shapes are identical either way
and neither side can detect it from the loaded graph -- unlike the
``emb_boss``/``N_CONT`` appends, which are self-describing. Nothing crashes;
the net is simply evaluated on an observation it never trained on, against an
unconditional anti-upgraded-card prior that then runs backwards.

So the schema is passed EXPLICITLY across the FFI, resolved from
``champions.toml``. The Rust-side twin of these assertions is
``sts2-rs/crates/sts2-runai/tests/ppo_champion_features_are_per_net.rs``.
"""

from __future__ import annotations

import json
import pathlib

import champions
import pytest

rust_policy = pytest.importorskip("ppo.rust_policy")
sts2_rs = pytest.importorskip("sts2_rs")


def _champion_path() -> str:
    path = champions.onnx_path("ironclad")
    if not path:
        pytest.skip("no committed Ironclad champion in this checkout")
    return path


_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "sts2-rs/crates/sts2-runai/tests/fixtures/ppo_encoder_parity.jsonl"
)


def _upgraded_card_reward_decision() -> tuple[dict, list[dict], dict]:
    """A real recorded card-reward decision, with one option pre-upgraded and
    the first deck card and first option enchanted.

    ``reward_upgrade`` only moves the upgrade column of a card-reward option
    token and ``card_mods`` only the enchantment columns of card tokens, so a
    decision without both cannot tell the schemas apart — a fabricated obs
    would risk asserting on a difference that is not there.
    """
    if not _FIXTURE.exists():
        pytest.skip(f"sibling sts2-rs fixture missing: {_FIXTURE}")
    for line in _FIXTURE.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        ids = row["obs"].get("card_reward_option_ids")
        if ids:
            obs = {
                **row["obs"],
                # First option upgraded, rest not: the flag the feature reads.
                "card_reward_option_upgraded": [i == 0 for i in range(len(ids))],
                # First deck card and first option enchanted: what `card_mods` reads.
                "deck_enchantments": [
                    ["SWIFT", 2] if i == 0 else None for i in range(len(row["obs"]["deck"]))
                ],
                "card_reward_option_enchantments": [
                    ["SWIFT", 2] if i == 0 else None for i in range(len(ids))
                ],
            }
            return obs, row["actions"], row["metrics"]
    pytest.skip("no card-reward decision in the parity fixture")


def test_the_declared_schema_actually_reaches_the_rust_encoder() -> None:
    """The spec is not merely stored — it changes what the net is shown.

    Without this, every other assertion here could pass while the value was
    recorded on the Python side and dropped at the FFI boundary. The two
    policies below load the SAME graph and differ only in declared schema, so
    any difference in output is the feature and nothing else.
    """
    obs, actions, metrics = _upgraded_card_reward_decision()
    path = _champion_path()

    off, _ = sts2_rs.PpoNetPolicy(path, "").distribution(obs, actions, metrics)
    on, _ = sts2_rs.PpoNetPolicy(path, "reward_upgrade").distribution(obs, actions, metrics)

    assert off != on, (
        "declaring reward_upgrade changed nothing about the distribution on a "
        "decision with a pre-upgraded reward option, so the spec is not "
        "reaching the encoder"
    )


def _declared_features() -> str:
    champ = champions.get("ironclad")
    assert champ is not None
    return champ.features


@pytest.mark.parametrize("hostile", ["", "reward_upgrade"])
def test_rust_policy_takes_its_schema_from_the_manifest_not_the_environment(
    monkeypatch: pytest.MonkeyPatch, hostile: str
) -> None:
    """The environment cannot change what a declared champion is served at.

    Asserted on OUTPUT, not on the stored string: the net must be *evaluated*
    at the manifest's schema, which only a comparison against an explicitly
    constructed policy at that schema can show.

    Parametrised over both env values on purpose. With a single value the test
    goes vacuous the moment the champion's own schema happens to match it --
    which is exactly what happened on 2026-08-28, when Ironclad's champion
    started declaring `reward_upgrade` and the old hostile value became the
    *correct* answer. One of these two cases always disagrees with the
    manifest, whatever the manifest says.
    """
    monkeypatch.setenv("STS2_PPO_FEATURES", hostile)
    obs, actions, metrics = _upgraded_card_reward_decision()
    path = _champion_path()
    declared = _declared_features()

    served, _ = rust_policy.RustPolicy(path).distribution(obs, actions, metrics)
    at_manifest, _ = sts2_rs.PpoNetPolicy(path, declared).distribution(obs, actions, metrics)

    assert rust_policy.RustPolicy(path).features == declared
    assert served == at_manifest, (
        f"the committed Ironclad champion declares features={declared!r} in "
        f"champions.toml, so it must be served at that schema no matter what "
        f"STS2_PPO_FEATURES says (here: {hostile!r})"
    )
    if hostile != declared:
        at_hostile, _ = sts2_rs.PpoNetPolicy(path, hostile).distribution(obs, actions, metrics)
        assert served != at_hostile, (
            "serving at the manifest schema and at the hostile env schema "
            "produced identical output, so this test could not have detected "
            "the env winning"
        )


def test_an_undeclared_model_gets_the_default_schema_not_a_champions(
    tmp_path,
) -> None:
    """A file merely NAMED like a champion inherits nothing from it.

    ``champions.by_onnx_filename`` requires the resolved path to live in the
    champions directory. Without that, a scratch export sharing a champion's
    basename would silently pick up its declared ``features`` -- the same
    "served at a schema it wasn't trained at" failure, moved from the env-var
    layer down to the path layer.
    """
    champ = champions.get("ironclad")
    assert champ is not None
    decoy = tmp_path / champ.onnx
    decoy.write_bytes(b"not a real onnx")

    assert champions.by_onnx_filename(str(decoy)) is None


def test_the_binding_rejects_a_misspelled_feature_rather_than_ignoring_it() -> None:
    """A typo must fail loudly, not screen as the default arm.

    Silently running the default arm is how an idea gets written down as "does
    not help" when it was never actually switched on.
    """
    with pytest.raises(ValueError, match="reward_upgrde"):
        sts2_rs.PpoNetPolicy(_champion_path(), "reward_upgrde")


def test_the_binding_accepts_a_declared_feature() -> None:
    """The escape hatch works: an explicit spec is honoured.

    This is the lane a gate sweep of a not-yet-promoted champion runs in --
    the net has no manifest entry, so the caller declares its schema.
    """
    assert sts2_rs.PpoNetPolicy(_champion_path(), "reward_upgrade") is not None
