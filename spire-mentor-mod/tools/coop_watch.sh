#!/usr/bin/env bash
# coop_watch.sh -- watch a coop_test.ps1 run unattended; one line per event.
#
# Usage: coop_watch.sh <log-dir> [stall-seconds]
#
# Emits, from both instances' godot logs and the advisor's stdout log:
#   - run-level picks (map / event / rest / treasure / shop) as a heartbeat
#   - errors, refusals, desync / checksum mismatches, failed saves
#   - advisor failures (a reply with no move: the run waits on it), from
#     advisor_stdout.log and the dummy's advisor_stdout.<port>.log
#   - STALL when neither instance executed an AI move for stall-seconds
#     (default 240); repeated while it lasts
#   - GAME EXITED when fewer than two game processes remain
# Silence is never "fine": a hung run produces STALL lines.
set -uo pipefail
dir="$1"
stall="${2:-240}"
advisor_dir="${LOCALAPPDATA:-$HOME/AppData/Local}/SpireMentor"

tail -q -F -n0 "$dir/host.godot.log" "$dir/dummy.godot.log" "$advisor_dir/advisor_stdout.log" "$advisor_dir"/advisor_stdout.9*.log 2>/dev/null \
  | grep -v --line-buffered "Generating checksum" \
  | grep -E --line-buffered \
      "Takeover executed on N(Map|EventRoom|RestSite|Treasure|MerchantRoom)|refuses|refused|without a move|Exception|\[ERROR\]|Failed to save|Disconnect|[Dd]esync|[Dd]ivergen|[Mm]ismatch|has lost|Victory|runaway|disarm|not available in co-op|advice_request failed|UnsupportedStateError" &
grep_pid=$!
trap 'kill $grep_pid 2>/dev/null' EXIT

last_count=-1
idle=0
while sleep 30; do
  n=$(cat "$dir/host.godot.log" "$dir/dummy.godot.log" 2>/dev/null | grep -c "Takeover executed")
  if [[ "$n" != "$last_count" ]]; then
    last_count=$n
    idle=0
  else
    idle=$((idle + 30))
    if (( idle >= stall && idle % stall < 30 )); then
      echo "STALL: no AI move in either instance for ${idle}s (moves so far: $n)"
    fi
  fi
  procs=$(tasklist 2>/dev/null | grep -ci "SlayTheSpire2")
  if (( procs < 2 )); then
    echo "GAME EXITED: $procs game process(es) left"
    exit 0
  fi
done
