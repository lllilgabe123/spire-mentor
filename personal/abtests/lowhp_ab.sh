#!/usr/bin/env bash
# Low-HP caution A/B on the beta: the shipped Ironclad run AI, clairvoyant
# fights (STS2_RUNAI_HONEST=0, like the live advisor), eval-cheap combat
# (1000/4000 sims), the same seeds in both arms. Arm "base": STS2_LOW_HP_CAUTION
# unset. Arm "caution": STS2_LOW_HP_CAUTION=1. Waits for Slay the Spire 2 to
# close first, and starts the training campaign again when done.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/lowhp"
PYCLI="$LOCALAPPDATA/sm073/sts2-pycli"
CKPT="$LOCALAPPDATA/sts2train/ironclad_champion_rebuilt.pt"
SEEDS=${SEEDS:-250}
mkdir -p "$OUT"

game_running() { tasklist //FI "IMAGENAME eq SlayTheSpire2.exe" | grep -qi SlayTheSpire2; }
while game_running; do sleep 30; done
echo "$(date +%T) game closed; starting A/B" | tee -a "$OUT/progress.txt"

cd "$PYCLI"
export PYTHONPATH=python PYTHONUTF8=1 STS2_RUNAI_HONEST=0
for asc in 10 9 8 7; do
  for arm in base caution; do
    f="$OUT/${arm}_A${asc}.jsonl"
    [ -s "$f" ] && [ "$(wc -l < "$f")" -ge "$SEEDS" ] && continue
    if [ "$arm" = caution ]; then export STS2_LOW_HP_CAUTION=1; else unset STS2_LOW_HP_CAUTION; fi
    nice -n 10 .venv/Scripts/python.exe -m ppo.probe --policy net_greedy --ckpt "$CKPT" \
      --seeds "$SEEDS" --ascension "$asc" --act1 roll --game-version v0.111.0 \
      --procs 16 --run-timeout 1200 --out "$f" > "$OUT/${arm}_A${asc}.log" 2>&1
    echo "$(date +%T) done $arm A$asc: $(grep -c '"won": true' "$f") wins / $(wc -l < "$f")" | tee -a "$OUT/progress.txt"
  done
done
unset STS2_LOW_HP_CAUTION
.venv/Scripts/python.exe "$HERE/compare.py" "$OUT" | tee "$OUT/result.txt"

# Back to training (the Start shortcut resumes from the last saved round).
powershell.exe -NoProfile -Command "Start-Process 'E:\SpireMentorTraining\Start AI training.cmd' -WorkingDirectory 'E:\SpireMentorTraining' -WindowStyle Minimized"
echo "$(date +%T) training restarted" | tee -a "$OUT/progress.txt"
