#!/bin/bash
# Playbook §2.6 standard eval, Ironclad a10_it700, arms D then P (see ../PREREG.md).
#
# As run on 2026-10-26 this driver hard-coded three private paths (the output archive, the
# worktree, its venv) and refused to start on any engine .so but sha256
# 823fa393eab74db57d0cffd6f71c4a5c873a74d345a354adfca8458bf1333ba2 or any HEAD but
# a1bd07c7193ce2f0ee303250d57573b52541b986. Those are settings now; the run itself is unchanged.
#
#   OUT    output folder (runs/, logs/, analysis/)  default ./ic-a10-it700-eval
#   ROOT   the repo checkout                     default the checkout holding this script
#   PY     python with torch and sts2_rs         default $ROOT/sts2-pycli/.venv/bin/python
#   CKPT   run-policy checkpoint, resolved from $ROOT/sts2-pycli
#                                                default data/ppo/ppo_fullrun_ironclad_a10_it700.pt
#          (the public snapshot ships no .pt: build one with onnx_to_ckpt.py and pass its path)
#   PROCS  worker processes                      default 30, as run (a run does not depend on it)
#   EXPECT_SO_SHA256, EXPECT_HEAD                optional pins; set them to the values above to
#                                                refuse anything but the 2026-10-26 build
#
# Linux only (it reads /proc/loadavg and calls sha256sum). Log lines also go to the terminal;
# `wc -l "$OUT"/runs/*.jsonl` shows progress within an arm.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd -P)}
mkdir -p "${OUT:-ic-a10-it700-eval}"/runs "${OUT:-ic-a10-it700-eval}"/logs "${OUT:-ic-a10-it700-eval}"/analysis
A=$(cd "${OUT:-ic-a10-it700-eval}" && pwd -P)
PY=${PY:-$ROOT/sts2-pycli/.venv/bin/python}
CKPT=${CKPT:-data/ppo/ppo_fullrun_ironclad_a10_it700.pt}
PROCS=${PROCS:-30}
LOG=$A/logs/driver.log
log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

# record the wheel/commit; refuse to run on anything but the pinned ones when pins are set
so_file=$(cd "$ROOT/sts2-pycli" && "$PY" -c 'import sts2_rs; print(getattr(sts2_rs, "sts2_rs", sts2_rs).__file__)')
so=$(sha256sum "$so_file" | cut -d' ' -f1)
head=$(git -C "$ROOT" rev-parse --verify -q HEAD 2>/dev/null || echo none)
[ -z "${EXPECT_SO_SHA256:-}" ] || [ "$so" = "$EXPECT_SO_SHA256" ] || { log "ABORT so sha $so"; exit 2; }
[ -z "${EXPECT_HEAD:-}" ] || [ "$head" = "$EXPECT_HEAD" ] || { log "ABORT head $head"; exit 2; }
log "driver start pid $$ so=$so head=$head"

for v in $(env | grep '^STS2_' | cut -d= -f1); do unset "$v"; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python
export STS2_RUNAI_MCTS_OVERRIDES="removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0"
cd "$ROOT/sts2-pycli"

run_arm() {
  local arm=$1; shift
  log "arm $arm start; load $(cut -d' ' -f1-3 /proc/loadavg); env: $(env | grep '^STS2_' | sort | tr '\n' ' ')"
  local t0=$(date +%s)
  env "$@" "$PY" -m ppo.probe --ckpt "$CKPT" \
    --policy net_greedy --character Ironclad --ascension 10 --act1 roll \
    --seeds 400 --seed-start 91000000 --procs "$PROCS" \
    --sims 4000 --boss-sims 16000 --run-timeout 14400 \
    --out "$A/runs/$arm.jsonl" > "$A/logs/$arm.stdout" 2> "$A/logs/$arm.stderr"
  local rc=$?
  local t1=$(date +%s)
  echo "$rc" > "$A/runs/$arm.exit"
  log "arm $arm end rc=$rc wall_s=$((t1-t0)) rows=$(wc -l < "$A/runs/$arm.jsonl" 2>/dev/null || echo 0)"
}

run_arm D
run_arm P STS2_RUNAI_PRIOR_NET=off
log "driver done"
echo DONE > "$A/logs/DONE"
