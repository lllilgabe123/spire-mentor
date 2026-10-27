# Built-in MCTS move priors (`data/prior_nets/`)

`ironclad.json` is the Ironclad run agent's PUCT move prior (GH #830). The
search reads it as a prior over a decision node's moves, blended half and half
with uniform (mix 0.5, softmax temperature 1). It is not a playout net: the
playouts still use `data/rollout_nets/ironclad.json`.

- **Where it applies:** elite and boss fights of the Ironclad run agent
  (`sts2-runai`). No other character has a prior. The advisor, the browser
  hint and the other interactive search surfaces do not use it.
- **How it is compiled in:** `sts2-mcts` embeds the file with `include_str!`
  (`rollout_net::builtin_prior_json`) when it exists (`cfg(prior_net_ironclad)`,
  set by `crates/sts2-mcts/build.rs`). The file is part of the engine's
  `src_hash` (`sts2-srchash` `MCTS_DATA_INPUTS`), so editing it rebuilds the
  engine and moves the wheel's identity. On a checkout without the file the
  test `the_ironclad_prior_net_is_compiled_in` fails.
- **Switches (`STS2_RUNAI_PRIOR_NET`):** unset = this built-in prior, Ironclad
  only, in elite and boss fights unless `STS2_RUNAI_PRIOR_ROOMS` names other
  tiers. `off` = no prior. Any other value is the path of a net file, used for
  every character and in every fight unless `STS2_RUNAI_PRIOR_ROOMS` narrows
  it. An empty value is an error. `STS2_RUNAI_PRIOR_MIX` / `_TEMP` set the
  blend and temperature for either route.
- **Format:** the playout-net JSON (`RolloutNet::from_json`): feature version
  1, hidden 64, embedding 32, 594 action keys. One line, no trailing newline.

## Training

Behaviour cloning on 137,357 Ironclad combat decisions by human players. The
fights were replayed exactly in this engine from community-uploaded
per-action replays (game build v0.111.0, which logs every RNG stream at
combat start). The target is the human's move at each decision. Held-out
top-1 agreement with the human move: 0.656 (the heuristic playout policy:
0.351).

## Provenance and gate

| | sha256 | bytes |
|---|---|---|
| gated file (`bc_ironclad_v2.json`) | `95f3e90920f5db8bd0a57661279df92e6a12ce8bcdbe871dcbb8284899513651` | 269,136 |
| shipped file (`ironclad.json`) | `ca4692595c22d1c26741c4557e1f7878151e253129ff2e8779d59269624ad4f2` | 268,757 |

The shipped file is the gated file with only its `meta` value replaced by `{}`
(the training metadata named local paths; nothing reads `meta`). Every weight
and key is byte-identical. The tests pin the shipped sha256
(`crates/sts2-mcts/tests/prior_net.rs`,
`sts2-pycli/tests/test_builtin_prior_net.py`). A different prior net needs a
new full-run gate.

The full-run gate (PR #938) measured this net and three search knobs as one
bundle. Its statement, verbatim:

```text
Ironclad A10, champion a10_it700 net_greedy; control: default search settings (no STS2_* variables) at 1000/4000; honest BeliefDraw, rl2 playout, --act1 roll; bundle: knobs 'removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0' plus the BC prior net 95f3e909 at mix 0.5, T 1 in elite and boss rooms; seeds 96,500,000-96,501,499, 1,500 paired; look 1 of 2.
The bundle raised the A10 win rate by at least +2.62pp (repeated 97.06% CI [+2.62, +7.78]; +5.20 +/- 1.18pp, 19.5% -> 24.7%, discordant 199/121, exact McNemar p 1.53e-05, look 1 of 2, alpha 0.0294 per look). This licenses a proposal to ship; it is not a ship decision. The estimate at a stopped look overstates the effect: under this design, at a true +1pp the expected estimate given PASS is 2.55pp (3.07 at a look-1 PASS), at +2pp 2.79 (3.28), at +3pp 3.32 (3.64); the median-unbiased estimate is +5.17pp.
Translation reading: 4x-like translation not rejected (U +7.78pp >= +3.0pp).
```

Plan with the lower bound, +2.62pp, not with the point estimate.

A second, separately pre-registered A10 gate of the same bundle (4,000 paired
seeds, 132,000,000-132,003,999) measured +3.45pp (19.5% -> 22.9%, McNemar
p 8.8e-07). Neither gate had a prior-only arm, so the prior's own share of
the gain is not measured.

## Why it ships publicly (D6 exception)

The scraped community run data and the nets trained on it stay private (D6,
`PROVENANCE.md` §2). This net is an exception by owner decision (2026-10-25,
GH #830). Unlike the playout nets in `data/rollout_nets/`, which learned in
the simulator, it learned from human actions in community-uploaded replays.
It ships as network weights and its action-key vocabulary only, with no run
data. The export ships exactly the two reviewed files in this folder: any
other file here fails `just check-public` until it is reviewed.
