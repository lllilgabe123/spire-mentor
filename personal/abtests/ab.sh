#!/usr/bin/env bash
# Run-AI A/B on the beta: the shipped Ironclad run AI, honest fights
# (deployed BeliefDraw; the live advisor knows draws only), eval-cheap combat
# (1000/4000 sims), the same seeds in every arm. Arms:
#   base     nothing changed
#   caution  STS2_LOW_HP_CAUTION=1 (HP below half counts double, engine)
#   pros     STS2_PRO_PRIOR=1 (card picks nudged by top beta players' Elo)
#   smith50  STS2_SMITH_AT=0.5 (upgrade instead of heal at >= 50% HP)
# Starts once Slay the Spire 2 has stayed closed for 15 minutes, backs off
# (and redoes the arm) if the game starts, and restarts training when done.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/run2"
PYCLI="$LOCALAPPDATA/sm073/sts2-pycli"
CKPT="$LOCALAPPDATA/sts2train/ironclad_champion_rebuilt.pt"
SEEDS=${SEEDS:-250}
ASCS=${ASCS:-"10 7 5"}
ARMS=${ARMS:-"base caution pros smith50"}
mkdir -p "$OUT"
# One copy at a time: a second start (e.g. a re-armed waiter) exits here.
mkdir "$OUT/.lock" 2>/dev/null || { echo "another A/B is already running on $OUT"; exit 0; }
trap 'rmdir "$OUT/.lock"' EXIT

game_running() { tasklist //FI "IMAGENAME eq SlayTheSpire2.exe" | grep -qi SlayTheSpire2; }
wait_idle() {
  local quiet=0
  while [ "$quiet" -lt 900 ]; do
    if game_running; then quiet=0; else quiet=$((quiet + 30)); fi
    sleep 30
  done
}

arm_env() {
  unset STS2_LOW_HP_CAUTION STS2_PRO_PRIOR STS2_PRO_PRIOR_FILE STS2_SMITH_AT
  case "$1" in
    caution) export STS2_LOW_HP_CAUTION=1 ;;
    pros) export STS2_PRO_PRIOR=1 STS2_PRO_PRIOR_FILE='E:\SpireMentorTraining\data\codex_ironclad_a10_wr50_v0111.json' ;;
    smith50) export STS2_SMITH_AT=0.5 ;;
  esac
}

wait_idle
echo "$(date +%T) game closed for 15 min; starting A/B ($ARMS x A$ASCS x $SEEDS seeds)" | tee -a "$OUT/progress.txt"
cd "$PYCLI"
export PYTHONPATH=python PYTHONUTF8=1
unset STS2_RUNAI_HONEST  # honest fights: closest sim match to the live "known draws" mode
for asc in $ASCS; do
  for arm in $ARMS; do
    f="$OUT/${arm}_A${asc}.jsonl"
    [ -s "$f" ] && [ "$(wc -l < "$f")" -ge "$SEEDS" ] && continue
    arm_env "$arm"
    while true; do
      rm -f "$f"
      nice -n 10 .venv/Scripts/python.exe -m ppo.probe --policy net_greedy --ckpt "$CKPT" \
        --seeds "$SEEDS" --ascension "$asc" --act1 roll --game-version v0.111.0 \
        --procs 16 --run-timeout 1200 --out "$f" > "$OUT/${arm}_A${asc}.log" 2>&1 &
      pid=$!
      while kill -0 "$pid" 2>/dev/null; do
        if game_running; then
          taskkill //PID "$(cat /proc/$pid/winpid)" //T //F > /dev/null 2>&1
          echo "$(date +%T) game started; pausing the A/B" | tee -a "$OUT/progress.txt"
          wait "$pid" 2>/dev/null
          wait_idle
          continue 2
        fi
        sleep 20
      done
      wait "$pid"
      break
    done
    echo "$(date +%T) done $arm A$asc: $(grep -c '"won": true' "$f") wins / $(wc -l < "$f")" | tee -a "$OUT/progress.txt"
  done
done
arm_env base
.venv/Scripts/python.exe "$HERE/compare.py" "$OUT" $ARMS | tee "$OUT/result.txt"
powershell.exe -NoProfile -Command "Start-Process 'E:\SpireMentorTraining\Start AI training.cmd' -WorkingDirectory 'E:\SpireMentorTraining' -WindowStyle Minimized"
echo "$(date +%T) training restarted" | tee -a "$OUT/progress.txt"
