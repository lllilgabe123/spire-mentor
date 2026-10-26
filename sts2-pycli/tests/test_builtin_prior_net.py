"""The compiled-in PUCT move prior (GH #830): ``sts2_rs.builtin_prior_net_json``.

Only Ironclad has one, the behaviour-cloned prior of the bundle that passed
its full-run gate (Ironclad A10, PR #938). The wheel must carry exactly the
gated net with its ``meta`` stripped. This pins the literal sha256, never
"whatever is committed": a new prior net needs a new full-run gate.
Game-free, so it runs in the public CI too (the net ships, PROVENANCE.md §2).
"""

from __future__ import annotations

import hashlib

import pytest
import sts2_rs

# sha256 of sts2-rs/data/prior_nets/ironclad.json: the gated
# bc_ironclad_v2.json (sha256 95f3e909…3651) with only `meta` set to {}.
SHIPPED_IRONCLAD_PRIOR_SHA256 = "ca4692595c22d1c26741c4557e1f7878151e253129ff2e8779d59269624ad4f2"


def test_the_built_in_ironclad_prior_is_the_shipped_gated_net() -> None:
    text = sts2_rs.builtin_prior_net_json("Ironclad")
    assert text is not None, "this wheel was built without data/prior_nets/ironclad.json"
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert sha == SHIPPED_IRONCLAD_PRIOR_SHA256, (
        "the gated net is 95f3e909… with meta stripped; "
        "a new prior net needs a new full-run gate (GH #830)"
    )


@pytest.mark.parametrize("character", ["Silent", "Defect", "Regent", "Necrobinder"])
def test_no_other_character_has_a_built_in_prior(character: str) -> None:
    assert sts2_rs.builtin_prior_net_json(character) is None


def test_an_unknown_character_raises() -> None:
    with pytest.raises(ValueError, match="unknown character"):
        sts2_rs.builtin_prior_net_json("Watcher")
