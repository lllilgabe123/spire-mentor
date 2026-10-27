#!/usr/bin/env bash
# Ironclad release win-rate panel on the GH #830 default (see PREREG.md).
# Same protocol as the 2026-10-11 release sweep (8000/8000 sims, single tree, act1 roll,
# seeds 81,000,000+, 200 at A0-A8 / 500 at A9-A10), Ironclad only, champion a10_it700,
# no STS2_* variables. Resumable: a cell whose log already says "win rate" is skipped.
#
# As run on 2026-10-26 this driver hard-coded two private paths (this folder and the worktree
# with its PGO venv), read the champion from data/ppo/champions.toml, and refused to start on any
# engine .so but sha256 823fa393eab74db57d0cffd6f71c4a5c873a74d345a354adfca8458bf1333ba2, any
# HEAD but 0181fa552a78053f260f0796e1cb086d770c7cef, or any checkpoint but sha256
# 91c83c1aafe137b22ab0617bea2a8fcb00e6b8777a67fe718b4d477a156a44fc. Those are settings now; the
# probe command, the cell order and the log lines are unchanged.
#
#   OUT        output folder (cells/, progress.log)    default ./ic-release-winrates
#   ROOT       the repo checkout                       default the checkout holding this script
#   PY         python with torch and sts2_rs           default $ROOT/sts2-pycli/.venv/bin/python
#   CKPT       run-policy checkpoint, resolved from $ROOT/sts2-pycli
#                                                      default data/ppo/ppo_fullrun_ironclad_a10_it700.pt
#              (the public snapshot ships no .pt: build one with the standard eval's
#              scripts/onnx_to_ckpt.py and pass its path)
#   PROCS      worker processes                        default 30, as run (a run does not depend on it)
#   ASC_ORDER  levels, in order                        default "10 9 0 6 3 8 1 5 2 4 7", as run
#   EXPECT_SO_SHA256, EXPECT_HEAD, EXPECT_CKPT_SHA256  optional pins; set them to the values
#              above to refuse anything but the 2026-10-26 build, commit and checkpoint
#
# Linux only (it reads /proc/loadavg and calls sha256sum). Log lines also go to the terminal;
# `wc -l "$OUT"/cells/*.jsonl` shows progress within a cell.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$(cd "$HERE/../../.." && pwd -P)}
mkdir -p "${OUT:-ic-release-winrates}"/cells
A=$(cd "${OUT:-ic-release-winrates}" && pwd -P)
PY=${PY:-$ROOT/sts2-pycli/.venv/bin/python}
CKPT=${CKPT:-data/ppo/ppo_fullrun_ironclad_a10_it700.pt}
PROCS=${PROCS:-30}
SEED_START=81000000
ASC_ORDER=${ASC_ORDER:-"10 9 0 6 3 8 1 5 2 4 7"}
P="$A/progress.log"
log() { echo "$(date -Is) $*" | tee -a "$P"; }

cd "$ROOT/sts2-pycli"
so_file=$("$PY" -c 'import sts2_rs; print(getattr(sts2_rs, "sts2_rs", sts2_rs).__file__)')
so=$(sha256sum "$so_file" | cut -d' ' -f1)
head=$(git -C "$ROOT" rev-parse --verify -q HEAD 2>/dev/null || echo none)
ck_sha=$(sha256sum "$CKPT" | cut -d' ' -f1)
[ -z "${EXPECT_SO_SHA256:-}" ] || [ "$so" = "$EXPECT_SO_SHA256" ] || { log "ABORT so $so"; exit 2; }
[ -z "${EXPECT_HEAD:-}" ] || [ "$head" = "$EXPECT_HEAD" ] || { log "ABORT head $head"; exit 2; }
[ -z "${EXPECT_CKPT_SHA256:-}" ] || [ "$ck_sha" = "$EXPECT_CKPT_SHA256" ] || { log "ABORT ckpt $CKPT $ck_sha"; exit 2; }

for v in $(env | grep '^STS2_' | cut -d= -f1); do unset "$v"; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python
log "driver start pid $$ head=$head so=$so ckpt=$CKPT"

for asc in $ASC_ORDER; do
  n=200; [ "$asc" -ge 9 ] && n=500
  cell="$A/cells/ironclad_A${asc}"
  if grep -q "^win rate" "$cell.log" 2>/dev/null; then continue; fi
  log "START Ironclad A$asc n=$n load $(cut -d' ' -f1-3 /proc/loadavg)"
  "$PY" -m ppo.probe \
    --policy net_greedy --ckpt "$CKPT" --character Ironclad \
    --ascension "$asc" --act1 roll --seeds "$n" --seed-start "$SEED_START" \
    --procs "$PROCS" --sims 8000 --boss-sims 8000 --run-timeout 14400 \
    --out "$cell.jsonl" > "$cell.log" 2>&1
  log "DONE  Ironclad A$asc rc=$? $(grep -m1 '^win rate' "$cell.log") $(grep -m1 '^wall/run' "$cell.log")"
done
log "ALL DONE"
