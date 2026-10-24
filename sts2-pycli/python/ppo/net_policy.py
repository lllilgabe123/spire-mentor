"""In-worker policy: load a RunPolicyNet checkpoint, sample per decision.

Each collection worker process builds one ``NetPolicy`` (checkpoint read
once per iteration per worker — the net is ~2M params) and calls
``sample()`` at every yielded decision. Inference is no_grad, CPU,
single-threaded (``torch.set_num_threads(1)`` — the pool parallelizes
across runs, combat MCTS dominates anyway).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import pro_prior, schema
from .encoder import featurize_row
from .model import Featurized, RunPolicyNet, collate, load_compat


class NetPolicy:
    def __init__(self, ckpt_path: str | None, dim: int = 192, n_layers: int = 3, n_heads: int = 4):
        torch.set_num_threads(1)
        self.net = RunPolicyNet(dim=dim, n_layers=n_layers, n_heads=n_heads)
        # GH #545: the feature set is a property of THIS checkpoint, not of the
        # shell that launched us. `STS2_PPO_FEATURES` is process-global and read
        # at import, so a gate sweep that exports it for the candidate arm would
        # silently corrupt the incumbent arm -- `reward_upgrade` appends no slot,
        # so the tensor looks identical either way.
        #
        # A checkpoint with no `cfg` (the trainer's rolling `<save>.pt` writes
        # `state_dict` only; the cfg rides on `<save>.iter_N.pt`) reads as the
        # DEFAULT schema, which is what every committed champion was trained at.
        self.features: frozenset[str] = frozenset()
        if ckpt_path:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state = ck.get("state_dict", ck)
            load_compat(self.net, state)
            try:
                cfg = ck.get("cfg") if isinstance(ck, dict) else None
                if isinstance(cfg, dict):
                    self.features = schema.parse_features(str(cfg.get("features") or ""))
            except Exception:
                # Defensive: a future cfg field that doesn't survive
                # `weights_only=True` must not fail the load -- fall back to
                # the default schema rather than crash a collection worker.
                self.features = frozenset()
        self.net.eval()

    @torch.no_grad()
    def distribution(
        self, obs: dict, actions: list[dict], metrics: dict
    ) -> tuple[list[float], float]:
        """Full masked softmax over the candidates + V(s).

        The probability-panel / greedy-recommend twin of :meth:`sample`:
        identical features and forward pass, returning the whole distribution
        instead of one draw.
        """
        feat = featurize_row(obs, actions, metrics, features=self.features)
        b = collate([feat])
        logits, value, _ = self.net(b)
        probs = F.softmax(logits[0, : feat.n_actions], dim=-1)
        return [float(p) for p in probs], float(value.item())

    @torch.no_grad()
    def logits_value(
        self, obs: dict, actions: list[dict], metrics: dict
    ) -> tuple[np.ndarray, float, Featurized]:
        """Full masked log-softmax over candidates + V(s).

        The search wrapper's root/leaf query (``ppo.search_policy``).
        Deliberately additive — :meth:`sample` below is on the live PPO
        collection path and stays byte-identical.
        """
        feat = featurize_row(obs, actions, metrics, features=self.features)
        b = collate([feat])
        logits, value, _ = self.net(b)
        logp = F.log_softmax(logits[0, : feat.n_actions], dim=-1)
        return logp.cpu().numpy().astype(np.float64), float(value.item()), feat

    @torch.no_grad()
    def sample(
        self,
        obs: dict,
        actions: list[dict],
        metrics: dict,
        rng: np.random.Generator,
        greedy: bool = False,
    ) -> tuple[int, float, float, Featurized]:
        """→ (chosen idx, log-prob under the sampling policy, V(s), features).

        ``greedy`` takes the argmax instead of sampling — diagnostic mode
        (isolates BC quality from temperature-1.0 entropy); PPO collection
        always samples.
        """
        feat = featurize_row(obs, actions, metrics, features=self.features)
        b = collate([feat])
        logits, value, _ = self.net(b)
        row = logits[0, : feat.n_actions]
        if pro_prior.active():
            # Personal-build A/B nudges (ppo.pro_prior); off unless env-set.
            row = torch.as_tensor(pro_prior.adjust(obs, actions, row.cpu().numpy()), dtype=row.dtype)
        logp = F.log_softmax(row, dim=-1)
        if greedy:
            idx = int(logp.argmax().item())
        else:
            probs = logp.exp().cpu().numpy().astype(np.float64)
            probs = probs / probs.sum()
            idx = int(rng.choice(feat.n_actions, p=probs))
        return idx, float(logp[idx].item()), float(value.item()), feat
