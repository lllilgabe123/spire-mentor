"""Rebuild the `a10_it700` torch checkpoint that `ppo.probe --ckpt` loads, from the shipped ONNX.

The public snapshot ships the champion as `sts2-pycli/data/ppo/ppo_fullrun_ironclad_a10_it700.onnx`
(sha256 pinned in `sts2-pycli/data/ppo/champions.toml`) but not the torch checkpoint (`*.pt` files
are not exported). `ppo.probe` loads a torch state dict, so this script copies the weights out of
the ONNX graph into one.

The ONNX holds 52 of the checkpoint's 54 tensors bit for bit: 38 under their own names with a
`net.` prefix, and 14 Linear weights stored transposed under generated names (the table below).
The other two are `aux_head.weight` / `aux_head.bias`, an auxiliary training head the ONNX export
drops. `ppo.net_policy.NetPolicy` ignores that head's output (`logits, value, _ = self.net(b)`), so
it cannot change a decision; it is left out, and `ppo.model.load_compat` keeps the net's own init
for it. The output has no `cfg`, so `NetPolicy` reads the default feature schema, which is the
schema this champion was trained with (`features = ""` in champions.toml).

The 52 tensors were checked against the private checkpoint
(`ppo_fullrun_ironclad_a10_it700.pt`, sha256 91c83c1aafe1...a156a44fc): EXPECTED_DIGEST below is
the digest of that checkpoint's tensors, and this script refuses to write a file whose tensors do
not hash to it.

Usage (from sts2-pycli/, after `uv sync --dev`; needs torch, onnx and numpy):

    E=../docs/experiments/2026-10-26-ic-a10-it700-standard-eval
    .venv/bin/python $E/scripts/onnx_to_ckpt.py --out a10_it700_from_onnx.pt
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
DEFAULT_ONNX = REPO / "sts2-pycli" / "data" / "ppo" / "ppo_fullrun_ironclad_a10_it700.onnx"
ONNX_SHA256 = "0b091ad6a0c08ddb1668052041cb6b815d961ae3aed2813cf0fcb66e95ca8de4"
# sha256 over the 52 tensors, as `tensor_digest` computes it, of the private checkpoint.
EXPECTED_DIGEST = "fa6148d63ada9418ffbd40059dabee076d398f0f58051d6feb46112ac08bcf24"
N_LAYERS = 3

# checkpoint key -> ONNX initializer holding its transpose (this .onnx only; ONNX_SHA256 pins it)
TRANSPOSED = {
    "cont_proj.proj.weight": "val_18",
    "blocks.0.attn.in_proj_weight": "val_30",
    "blocks.0.w1.weight": "val_137",
    "blocks.0.v.weight": "val_140",
    "blocks.0.w2.weight": "val_141",
    "blocks.1.attn.in_proj_weight": "val_149",
    "blocks.1.w1.weight": "val_254",
    "blocks.1.v.weight": "val_257",
    "blocks.1.w2.weight": "val_258",
    "blocks.2.attn.in_proj_weight": "val_266",
    "blocks.2.w1.weight": "val_371",
    "blocks.2.v.weight": "val_374",
    "blocks.2.w2.weight": "val_375",
    "policy_head.weight": "val_394",
}
EMBEDDINGS = (
    "card",
    "upg",
    "relic",
    "potion",
    "room",
    "event",
    "fixed",
    "optidx",
    "surface",
    "boss",
    "deckop",
    "ench",
    "sec",
)
DIRECT = (
    [f"emb_{e}.weight" for e in EMBEDDINGS]
    + ["cont_proj.freqs", "cont_proj.proj.bias"]
    + [
        f"blocks.{i}.{p}"
        for i in range(N_LAYERS)
        for p in (
            "n1.weight",
            "attn.in_proj_bias",
            "attn.out_proj.weight",
            "attn.out_proj.bias",
            "n2.weight",
            "w1.bias",
        )
    ]
    + [
        "norm.weight",
        "policy_head.bias",
        "value_norm.weight",
        "value_head.weight",
        "value_head.bias",
    ]
)


def tensor_digest(state: dict[str, np.ndarray]) -> str:
    """sha256 over (key, dtype, shape, C-order bytes) of every tensor, keys sorted."""
    h = hashlib.sha256()
    for k in sorted(state):
        a = np.ascontiguousarray(state[k])
        h.update(f"{k}\0{a.dtype.str}\0{a.shape}\0".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def from_onnx(path: Path) -> dict[str, np.ndarray]:
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(path)).graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    state = {k: inits[f"net.{k}"] for k in DIRECT}
    state |= {k: inits[v].T for k, v in TRANSPOSED.items()}
    return {k: np.ascontiguousarray(v) for k, v in state.items()}


def from_ckpt(path: Path) -> dict[str, np.ndarray]:
    """The same 52 tensors from a torch checkpoint (to check the digest against the original)."""
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)
    return {k: sd[k].numpy() for k in (*DIRECT, *TRANSPOSED)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="the shipped champion ONNX")
    ap.add_argument("--out", type=Path, help="where to write the torch checkpoint")
    ap.add_argument("--digest-of", type=Path, help="print the tensor digest of this .pt and exit")
    args = ap.parse_args()

    if args.digest_of:
        print(tensor_digest(from_ckpt(args.digest_of)))
        return 0
    if args.out is None:
        ap.error("--out is required")
    sha = hashlib.sha256(args.onnx.read_bytes()).hexdigest()
    if sha != ONNX_SHA256:
        raise SystemExit(
            f"{args.onnx}: sha256 {sha}, expected {ONNX_SHA256} (the mapping is for that file only)"
        )
    state = from_onnx(args.onnx)
    digest = tensor_digest(state)
    if digest != EXPECTED_DIGEST:
        raise SystemExit(f"tensor digest {digest} != expected {EXPECTED_DIGEST}; nothing written")

    import torch

    torch.save(
        {"state_dict": {k: torch.from_numpy(v.copy()) for k, v in state.items()}}, str(args.out)
    )
    print(
        f"wrote {args.out}: {len(state)} tensors, digest {digest} (matches the private checkpoint)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
