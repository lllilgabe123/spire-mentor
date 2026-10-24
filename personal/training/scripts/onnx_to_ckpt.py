"""Rebuild a trainable RunPolicyNet checkpoint from a shipped champion ONNX.

The author publishes champions as ONNX only. torch.onnx (dynamo) keeps most
parameter names under `net.`, but stores every bias-free Linear weight
transposed under an anonymous `val_N`. This maps those back by shape and graph
order, then proves the result: onnxruntime on the ONNX and the rebuilt torch
net must give the same logits and value on random valid inputs. The aux head
is not exported (training-only), so it keeps a fresh init.

usage: python onnx_to_ckpt.py <champion.onnx> <out.pt>   (run from sts2-pycli)
"""
import itertools
import sys

sys.path.insert(0, "python")
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from ppo.model import RunPolicyNet
from ppo.export_onnx import ExportWrapper

src, out = sys.argv[1], sys.argv[2]
model = onnx.load(src)
inits = {}
anon = []  # transposed Linear weights, in graph order
for i in model.graph.initializer:
    a = numpy_helper.to_array(i)
    if i.name.startswith("net."):
        inits[i.name[4:]] = a
    elif a.ndim == 2 and min(a.shape) > 1 or a.shape == (192, 1):
        anon.append((i.name, a))

net = RunPolicyNet()
sd = net.state_dict()
n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
by_shape = {}
for name, a in anon:
    by_shape.setdefault(a.shape, []).append(a)
proj = by_shape[(sd["cont_proj.proj.weight"].shape[1], sd["cont_proj.proj.weight"].shape[0])]
attn = by_shape[(192, 576)]
up = by_shape[(192, 768)]
down = by_shape[(768, 192)]
head = by_shape[(192, 1)]
assert len(proj) == 1 and len(attn) == n_blocks and len(up) == 2 * n_blocks and len(down) == n_blocks and len(head) == 1


def build(swap: bool) -> RunPolicyNet:
    new = {k: torch.from_numpy(v.copy()) for k, v in inits.items() if k in sd}
    new["cont_proj.proj.weight"] = torch.from_numpy(proj[0].T.copy())
    new["policy_head.weight"] = torch.from_numpy(head[0].T.copy())
    for b in range(n_blocks):
        new[f"blocks.{b}.attn.in_proj_weight"] = torch.from_numpy(attn[b].T.copy())
        first, second = up[2 * b], up[2 * b + 1]
        if swap:
            first, second = second, first
        new[f"blocks.{b}.w1.weight"] = torch.from_numpy(first.T.copy())
        new[f"blocks.{b}.v.weight"] = torch.from_numpy(second.T.copy())
        new[f"blocks.{b}.w2.weight"] = torch.from_numpy(down[b].T.copy())
    missing = [k for k in sd if k not in new]
    assert missing == ["aux_head.weight", "aux_head.bias"], missing
    n = RunPolicyNet()
    n.load_state_dict(new, strict=False)
    return n.eval()


sess = ort.InferenceSession(src)
cat_rows = [sd[f"{t}.weight"].shape[0] for t in (
    "emb_card emb_upg emb_relic emb_potion emb_room emb_event emb_fixed "
    "emb_optidx emb_surface emb_boss emb_deckop emb_ench").split()]
rng = np.random.default_rng(0)


def sample():
    T, A, L = 320, 160, int(rng.integers(40, 300))
    cat = np.stack([rng.integers(0, r, size=(1, T)) for r in cat_rows], axis=-1).astype(np.int64)
    cont = rng.normal(size=(1, T, 25)).astype(np.float32)
    sec = rng.integers(0, sd["emb_sec.weight"].shape[0], size=(1, T)).astype(np.int64)
    pad = np.zeros((1, T), dtype=bool); pad[0, L:] = True
    act = np.zeros((1, A), dtype=bool); act[0, : int(rng.integers(2, 40))] = True
    return dict(cat=cat, cont=cont, sec=sec, pad_mask=pad, obs_len=np.array([L], dtype=np.int64), act_mask=act)


samples = [sample() for _ in range(8)]
best = None
for swap in (False, True):
    w = ExportWrapper(build(swap)).eval()
    worst = 0.0
    for s in samples:
        ref_logits, ref_value = sess.run(["logits", "value"], s)
        with torch.no_grad():
            lg, v = w(*(torch.from_numpy(s[k]) for k in ("cat", "cont", "sec", "pad_mask", "obs_len", "act_mask")))
        m = np.isfinite(ref_logits)
        worst = max(worst, float(np.abs(lg.numpy()[m] - ref_logits[m]).max()), float(np.abs(v.numpy() - ref_value).max()))
    print(f"swap={swap}: max abs diff vs shipped ONNX = {worst:.2e}")
    if worst < 1e-3 and best is None:
        best = swap
if best is None:
    raise SystemExit("no mapping reproduces the shipped net")
torch.save({"state_dict": build(best).state_dict()}, out)
print(f"OK: wrote {out}")
