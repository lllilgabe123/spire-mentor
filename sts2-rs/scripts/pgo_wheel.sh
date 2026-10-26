#!/usr/bin/env bash
# PGO build of the sts2_rs PyO3 wheel — the engine the PPO trainer,
# campaigns, `ppo.probe` and gate runs all burn their CPU in.
#
# scripts/pgo_build.sh covers the CLI binaries; the PPO trainer never runs
# those — each worker burns its CPU inside the maturin-built sts2_rs cdylib,
# which wheel-venv.sh builds plain `--release`. This script closes that gap:
# it produces a PGO wheel in the SAME venv wheel-venv.sh would use, with
# the same freshness gate, plus an identity check that the optimized wheel
# makes byte-identical decisions to the plain one.
#
#   phase 0  plain wheel via wheel-venv.sh (provisions the venv, baseline)
#            + identity traces A on that plain wheel
#   phase 1  instrumented build   (-Cprofile-generate, dedicated target dir)
#   phase 2  profile workload     (pgo_wheel_workload.py --agent ppo: full
#            runs through RunManager + PpoRunAgent = the trainer's MCTS
#            combat with the five deployed champions making the meta
#            decisions, mixed ascensions, Act 1 rolled — so the profile
#            reaches the relic-heavy act-2/3 bosses that dominate combat
#            wall; profraw flushes on interpreter exit)
#   phase 3  llvm-profdata merge + optimized build (-Cprofile-use)
#            + freshness gate + identity traces B, diffed against A
#   install  the verified PGO .so replaces the venv's plain one atomically
#            (copy to a temp name in the same dir, then rename), and uv's
#            metadata gets the key taken before phase 0 (see below)
#
# The venv holds only the plain wheel until the install step. Phases 1-3
# `maturin build` wheels (not `maturin develop`) and unpack them under
# target-pgo-wheel-stage/, and every workload imports a STAGED copy through
# PYTHONPATH (identity A too, from a copy of the plain install). So nothing
# else ever imports the instrumented build (it would write stray .profraw), a
# concurrent `just sim-engine` in the same venv cannot swap a build out from
# under a workload, and a kill, OOM or failure at any point before the install
# leaves the venv's plain wheel in place. A SIGKILL skips the EXIT trap, so the
# staged builds (target-pgo-wheel-stage/, the PGO one unverified but stamped
# pgo=True) stay until the next run deletes them: never put them on PYTHONPATH
# (the trainer guard refuses any sts2_rs imported from a target-pgo-wheel-*
# dir).
#
# Both workloads run one process per (character, seed), --jobs at a time
# (`default_%m.profraw` merges concurrent writers). The identity check is
# full runs on --identity-seeds for every character (default 10 x 5 = 50
# runs); ppo traces also carry every combat root's visit/value sums, so a
# PGO build that perturbs a single search fails here even if the decision
# survives. Any difference (or any other failure) refuses the install and
# exits 1; a failure after the install puts the plain .so back.
# The check drives PpoRunAgent (tract meta + the wheel's MCTS combat), not
# the trainer's play_batch (torch meta + MctsCombatAgent): the workload is
# stdlib-only, so that binding path is not re-verified here.
#
# --skip-identity skips both identity runs and therefore NEVER installs: the
# venv keeps the plain wheel, and the unverified PGO module is left in
# target-pgo-wheel-unverified/ (for PYTHONPATH measurements; delete it after).
# So any PGO .so in a venv has passed the identity check, and the trainer
# guard's `build_info()["pgo"]` needs no second marker. The unverified module
# reports pgo=True too; the guard refuses it by its path (target-pgo-wheel-*),
# so it cannot train even from a PYTHONPATH.
#
# uv: `uv run` / `uv sync` (`just test`, `lint`, `typecheck`, ...) reinstall
# the editable sts2-rs unless its dist-info `uv_cache.json` equals the key uv
# computes from the source: the newest ctime among sts2-py's `[tool.uv]
# cache-keys` files (crates/sts2-py/pyproject.toml itself, every .rs and
# Cargo.toml under crates/, every file in a crate's src/, runai's
# data/runai/heuristic_v.toml, the move prior data/prior_nets/*.json, the
# workspace Cargo.toml, Cargo.lock and rust-toolchain.toml). maturin's
# install never matches. So before phase 0 the script takes that key
# (`sts2-pycli/scripts/uv_stamp_wheel.py key`), and after the install it
# writes exactly that key (`stamp --check`,
# which refuses if the key has moved since or src_hash disagrees, and asks uv
# whether it agrees). The PGO wheel then survives `uv run` until one of those
# files changes after the build STARTED (an edit during the build included,
# and a ctime-only change such as a checkout rewriting a file), when uv
# rebuilds a plain one as before. A failed stamp is only a warning: the
# trainer guard refuses the plain wheel uv would install. The key needs Python
# >= 3.11 (tomllib): the venv's interpreter when the venv exists, else
# `python3`, else whatever `uv python find '>=3.11'` finds. With none of them
# the build goes on and only the stamp is skipped, with the same warning.
#
# Right before the install the script re-checks that the source still hashes
# to what the PGO build compiled (identity B takes minutes, and a concurrent
# edit plus `just sim-engine` into the same venv must not be overwritten by a
# PGO build of the older source); a moved source refuses the install. One run
# per checkout: the fixed build dirs below are shared, so a second
# pgo_wheel.sh on the same SRC refuses to start while the first one lives
# (lock dir target-pgo-wheel.lock, reaped when its PID is gone).
#
# Every workload process runs under `timeout --shard-timeout` (default 1800 s;
# a shard is one full run: at most 42 s instrumented and 30 s plain on the
# 2026-10-13 build), so an engine hang fails the build with a TIMED OUT shard
# instead of stalling it forever.
#
# Provenance: the instrumented build is compiled with
# STS2_WHEEL_PGO_INSTRUMENTED=1 and the optimized one with
# STS2_WHEEL_PGO_PROFDATA_SHA256=<sha256 of the merged .profdata> (plus
# STS2_WHEEL_TARGET_CPU with --v3). sts2-py bakes them in with `option_env!`,
# `sts2_rs.build_info()` reports them, and the script checks that every staged
# build, and the installed .so, carries the stamp it was built with. The PPO
# trainer refuses a wheel whose build_info()["pgo"] is false
# (sts2-pycli/python/ppo/wheel_guard.py). No other build path sets these vars:
# wheel-venv.sh strips them from its plain builds.
#
# The fat-LTO target dirs (target-pgo-wheel-gen/-use, 1-2 GB each) are
# cargo-cleaned on exit, so every run builds cold. The staged copies go too. A
# successful run also deletes the profraw and the .profdata, so nothing
# invites reusing them; a failed one keeps them, the logs and the identity
# traces for diagnosis.
#
# Usage (from anywhere; SRC defaults to the checkout containing this script):
#   scripts/pgo_wheel.sh [--src PATH] [--v3]
#                        [--agent ppo|heuristic] [--characters "Ironclad ..."]
#                        [--seeds "101 ... 108"] [--ascensions "0 5 10"]
#                        [--sims 250] [--boss-sims 1000] [--max-steps 5000]
#                        [--identity-seeds "1001 ... 1010"] [--jobs 10]
#                        [--shard-timeout 1800] [--skip-identity]
# --agent heuristic uses the same defaults (all five characters, full runs,
# at A0). The pre-2026-10 profile needs explicit flags: --agent heuristic
# --characters Ironclad --seeds "1 4 11 14" --sims 400 --boss-sims 800
# --max-steps 150. --shard-timeout 0 disables the per-shard timeout.
#
# Entry point: `just sim-engine-pgo` (sts2-pycli), i.e. STS2_WHEEL_PGO=1
# wheel-venv.sh build, which delegates here and installs into the current
# checkout's own venv. Not the `just sim-engine` default: the two-phase build
# costs ~2 extra cold compiles of the workspace (fat LTO) plus the workloads.
# Training requires it; other uses (probes, gates) may opt in.
#
# No -Ctarget-cpu by default, so the .so stays portable (the Docker and
# advisor wheels can use the same recipe); --v3 adds -Ctarget-cpu=x86-64-v3,
# which measured only +0-2% on top of PGO and narrows the wheel to this host
# class. --no-v3 is accepted as a no-op alias for old callers.
#
# Re-profile for every build: never reuse a .profdata across binaries (the
# August +2.4% on mcts_realistic_profile used this wheel's profdata, so it
# measured profile mismatch). A same-source, freshly profiled wheel measured
# 1.42x on the trainer workload before the 2026-10 engine-speed work and
# 1.33-1.35x after it. The August wheel's own same-source 1.037x is
# unexplained: the old recipe (v3, the old profile, build-script .profraw
# merged in) re-run on the 2026-10 source gains 1.34x too. See item M13 in
# docs/perf/2026-10-13-engine-speed-exact-design.md.
#
# PGO and v3 are behavior-identical on the search workload
# (docs/perf/2026-07-05-mcts-single-thread-speedup.md; the identity check
# below re-verifies that through the wheel on every build).
set -euo pipefail
# mapfile and `wait -n` (macOS ships bash 3.2; use a newer bash there).
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    echo "pgo-wheel: needs bash >= 4.3 (this is $BASH_VERSION)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd -P)"
EXTRA_CPU=""
AGENT=ppo
CHARACTERS="Ironclad Silent Defect Regent Necrobinder"
SEEDS="101 102 103 104 105 106 107 108"
ASCENSIONS="0 5 10"
SIMS=250
BOSS_SIMS=1000
MAX_STEPS=5000
IDENTITY_SEEDS="1001 1002 1003 1004 1005 1006 1007 1008 1009 1010"
JOBS=10
SHARD_TIMEOUT=1800
SKIP_IDENTITY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src) SRC="$(cd "$2" && pwd -P)"; shift 2 ;;
        --v3) EXTRA_CPU="-Ctarget-cpu=x86-64-v3"; shift ;;
        --no-v3) EXTRA_CPU=""; shift ;;
        --agent) AGENT="$2"; shift 2 ;;
        --characters) CHARACTERS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --ascensions) ASCENSIONS="$2"; shift 2 ;;
        --sims) SIMS="$2"; shift 2 ;;
        --boss-sims) BOSS_SIMS="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --identity-seeds) IDENTITY_SEEDS="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --shard-timeout) SHARD_TIMEOUT="$2"; shift 2 ;;
        --skip-identity) SKIP_IDENTITY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
case "$AGENT" in ppo|heuristic) ;; *) echo "pgo-wheel: --agent must be ppo or heuristic" >&2; exit 2 ;; esac
[[ $SHARD_TIMEOUT =~ ^[0-9]+$ ]] || { echo "pgo-wheel: --shard-timeout takes whole seconds" >&2; exit 2; }
# An empty list would make a workload of zero runs: an empty profile, or an
# identity check that compares nothing and passes.
if [[ -z ${CHARACTERS// /} || -z ${SEEDS// /} ]]; then
    echo "pgo-wheel: --characters and --seeds must each name at least one" >&2
    exit 2
fi
if (( SKIP_IDENTITY == 0 )) && [[ -z ${IDENTITY_SEEDS// /} ]]; then
    echo "pgo-wheel: --identity-seeds is empty (an identity check of zero runs proves" \
        "nothing); pass seeds, or --skip-identity" >&2
    exit 2
fi

[ -f "$SRC/crates/sts2-py/Cargo.toml" ] || { echo "pgo-wheel: no sts2-py crate under $SRC" >&2; exit 1; }
PYCLI_DIR="$(cd "$SRC/../sts2-pycli" && pwd -P)"
WHEEL_VENV="$PYCLI_DIR/scripts/wheel-venv.sh"
MANIFEST="$SRC/crates/sts2-py/Cargo.toml"
PGO_DIR="$SRC/target-pgo-wheel-data"
PROFDATA="$SRC/target-pgo-wheel.profdata"
IDENTITY_DIR="$SRC/target-pgo-wheel-identity"
LOG_DIR="$SRC/target-pgo-wheel-logs"
STAGE="$SRC/target-pgo-wheel-stage"
UNVERIFIED="$SRC/target-pgo-wheel-unverified"
LOCK="$SRC/target-pgo-wheel.lock"
# Preflight, before the first build: a missing tool must not surface only
# after ~10 minutes of compiling. A host that cannot provide them trains on the
# plain wheel with the trainer's --allow-plain-wheel.
# Every rustc/cargo/maturin below runs on $SRC's rust-toolchain.toml pin.
# rustup resolves the pin from the cwd, and `just sim-engine-pgo` runs from
# sts2-pycli, which has none: the builds silently used the default toolchain,
# and once the pin moved past it (#956) the instrumented build's profile no
# longer matched the pinned llvm-profdata ("raw profile version mismatch").
if command -v rustup >/dev/null 2>&1; then
    RUSTUP_TOOLCHAIN="$(cd "$SRC" && rustup show active-toolchain | cut -d' ' -f1)"
    export RUSTUP_TOOLCHAIN
fi
LLVM_PROFDATA="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/host: //p')/bin/llvm-profdata"
if [[ ! -x "$LLVM_PROFDATA" ]]; then
    echo "pgo-wheel: llvm-profdata not found — run: rustup component add llvm-tools --toolchain ${RUSTUP_TOOLCHAIN:-<pinned>}" >&2
    exit 1
fi
declare -a SHARD_LIMIT=()
if (( SHARD_TIMEOUT > 0 )); then
    if ! timeout --version >/dev/null 2>&1; then
        echo "pgo-wheel: GNU timeout not found (coreutils) — install it, or pass --shard-timeout 0 to run without per-shard timeouts" >&2
        exit 1
    fi
    SHARD_LIMIT=(timeout --kill-after=60 "$SHARD_TIMEOUT")
fi

# One pgo_wheel.sh per checkout: every dir above is fixed per SRC, and a second
# run would delete the first one's staged modules, logs, profiles and target
# dirs under it. Atomic mkdir, as build-mutex.sh; a lock whose PID is gone is
# stale (a SIGKILL skips the EXIT trap that releases it).
take_lock() {
    local owner
    while ! mkdir "$LOCK" 2>/dev/null; do
        owner="$(cat "$LOCK/pid" 2>/dev/null || true)"
        if [[ -z $owner ]]; then
            sleep 1  # the holder may be between its mkdir and its pid write
            owner="$(cat "$LOCK/pid" 2>/dev/null || true)"
        fi
        if [[ -z $owner ]] || ! kill -0 "$owner" 2>/dev/null; then
            echo "pgo-wheel: reaping a stale lock ${owner:+from dead PID $owner }($LOCK)" >&2
            rm -rf "$LOCK"
            continue
        fi
        echo "pgo-wheel: another pgo_wheel.sh (PID $owner) is building $SRC;" \
            "not starting a second one" >&2
        exit 1
    done
    echo "$$" >"$LOCK/pid"
}
release_lock() {
    if [[ "$(cat "$LOCK/pid" 2>/dev/null || true)" == "$$" ]]; then
        rm -rf "$LOCK"
    fi
}
take_lock
trap release_lock EXIT

MUTEX="$SRC/scripts/build-mutex.sh"
declare -a PREFIX=()
[ -x "$MUTEX" ] && PREFIX=("$MUTEX")

# `maturin develop`'s native tag: the plain `linux` one on Linux (no manylinux
# claim); other platforms have no such tag to ask for.
declare -a COMPAT=()
[[ $(uname -s) == Linux ]] && COMPAT=(--compatibility linux)

# build_stage NAME RUSTFLAGS TARGET_DIR [VAR=value...]: build the wheel with
# those RUSTFLAGS in its own CARGO_TARGET_DIR and the given build_info() stamps
# (any stamp inherited from the caller's env is dropped first), and unpack its
# sts2_rs package into $STAGE/NAME, for PYTHONPATH=$STAGE/NAME. The same
# invocation shape as wheel-venv.sh's `maturin develop` minus the install: the
# venv's interpreter, and the native tag (`linux` there) with no auditwheel
# pass, as develop builds. The dedicated target dir keeps the instrumented/optimized
# artifacts out of the shared target/ and the sccache cache (sccache is
# disabled outright — it does not replay profile-generate/-use compilations
# correctly across profile paths).
build_stage() {
    local name="$1" rustflags="$2" target_dir="$3"; shift 3
    local wheels="$STAGE/$name-wheel"
    rm -rf "$wheels" "${STAGE:?}/$name"
    env -u STS2_WHEEL_PGO_PROFDATA_SHA256 -u STS2_WHEEL_PGO_INSTRUMENTED -u STS2_WHEEL_TARGET_CPU \
        "$@" VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" \
        RUSTC_WRAPPER= RUSTFLAGS="$rustflags" CARGO_TARGET_DIR="$target_dir" \
        "${PREFIX[@]}" maturin build --release -m "$MANIFEST" -i "$VENV/bin/python" \
        "${COMPAT[@]}" --auditwheel skip -o "$wheels" >&2
    "$VENV/bin/python" -I - "$wheels" "$STAGE/$name" "$SO_NAME" >&2 <<'PY'
import sys
import zipfile
from pathlib import Path

wheels, dest, so_name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
found = sorted(wheels.glob("*.whl"))
if len(found) != 1:
    sys.exit(f"pgo-wheel: expected one wheel in {wheels}, found {found}")
with zipfile.ZipFile(found[0]) as zf:
    for info in zf.infolist():
        if info.filename.startswith("sts2_rs/") and not info.is_dir():
            zf.extract(info, dest)
            (dest / info.filename).chmod(info.external_attr >> 16 & 0o777 or 0o644)
if not (dest / "sts2_rs" / so_name).is_file():
    sys.exit(f"pgo-wheel: {found[0].name} has no sts2_rs/{so_name} (the venv's extension name)")
PY
    rm -rf "$wheels"
}

# check_stamp DIR PGO SHA: the sts2_rs that PYTHONPATH=DIR imports (DIR empty:
# the venv's own installed one) must come from DIR and report build_info()
# pgo=PGO (1/0) and profdata_sha256=SHA (empty = none). Catches a build that
# did not pick up its stamp, which the trainer guard would then misjudge, and a
# PYTHONPATH that does not reach the staged copy the workloads are meant to run.
check_stamp() {
    local dir="$1"; shift
    local -a py=("$VENV/bin/python" -I)
    [[ -n $dir ]] && py=(env PYTHONPATH="$dir" "$VENV/bin/python" -s)
    "${py[@]}" - "$dir" "$@" >&2 <<'PY'
import sys

import sts2_rs

where, want_pgo, want_sha = sys.argv[1:]
if where and not sts2_rs.__file__.startswith(where + "/"):
    sys.exit(f"pgo-wheel: imported {sts2_rs.__file__}, not the staged copy under {where}")
info = sts2_rs.build_info()
want = (want_pgo == "1", want_sha or None)
if (info["pgo"], info["profdata_sha256"]) != want:
    sys.exit(f"pgo-wheel: build_info() = {info}; expected (pgo, profdata_sha256) = {want}")
print(f"pgo-wheel: build_info ✓ pgo={info['pgo']} instrumented={info['pgo_instrumented']}"
      f" ({sts2_rs.__file__})")
PY
}
declare -a CPU_STAMP=()
[[ -n $EXTRA_CPU ]] && CPU_STAMP=("STS2_WHEEL_TARGET_CPU=${EXTRA_CPU#-Ctarget-cpu=}")

declare -a AGENT_ARGS=(--agent "$AGENT")
if [[ $AGENT == ppo ]]; then
    AGENT_ARGS+=(--ascensions "$ASCENSIONS" --champions "$PYCLI_DIR/data/ppo/champions.toml")
fi

# run_shards NAME MOD_DIR SEEDS [TRACE_DIR]: one workload process per
# (character, seed) on the sts2_rs staged in MOD_DIR (PYTHONPATH), at most
# $JOBS at a time, each under $SHARD_LIMIT; with TRACE_DIR each writes
# TRACE_DIR/<character>_<seed>.json. Fails if any shard fails or times out.
run_shards() {
    local name="$1" mod_dir="$2" seeds="$3" trace_dir="${4:-}" ch seed
    local failed="$LOG_DIR/$name.failed"
    mkdir -p "$LOG_DIR/$name"
    rm -f "$failed"
    [[ -n $trace_dir ]] && mkdir -p "$trace_dir"
    for ch in $CHARACTERS; do
        for seed in $seeds; do
            while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
            local log="$LOG_DIR/$name/${ch}_$seed.log"
            local -a trace=()
            [[ -n $trace_dir ]] && trace=(--trace "$trace_dir/${ch}_$seed.json")
            {
                rc=0
                PYTHONPATH="$mod_dir:$PYCLI_DIR/python" "${SHARD_LIMIT[@]}" "$VENV/bin/python" \
                    "$SCRIPT_DIR/pgo_wheel_workload.py" "${AGENT_ARGS[@]}" \
                    --character "$ch" --seeds "$seed" --sims "$SIMS" \
                    --boss-sims "$BOSS_SIMS" --max-steps "$MAX_STEPS" "${trace[@]}" \
                    >"$log" 2>&1 || rc=$?
                # timeout exits 124 (SIGTERM did it) or 137 (needed SIGKILL).
                if (( ${#SHARD_LIMIT[@]} > 0 && (rc == 124 || rc == 137) )); then
                    echo "$ch $seed (TIMED OUT after ${SHARD_TIMEOUT}s — engine hang?)" >>"$failed"
                elif (( rc != 0 )); then
                    echo "$ch $seed (rc=$rc)" >>"$failed"
                fi
            } &
        done
    done
    wait
    grep -h ' seed=' "$LOG_DIR/$name"/*.log >&2 || true
    if [[ -s $failed ]]; then
        echo "pgo-wheel: $name: shard(s) failed: $(paste -sd ';' "$failed") — logs in $LOG_DIR/$name" >&2
        exit 1
    fi
}

# uv's freshness key for the source as it is now, before anything is built:
# the install writes this key, not the one current at install time, so a
# cache-key file that changes from here on makes uv rebuild (see the header).
# Taken with the venv's interpreter when there is one (the one the stamp runs
# on), else any Python >= 3.11. Without one, only the stamp is skipped.
uv_key() {
    local venv py
    local -a cands=()
    venv="$(cd "$SRC" && env -u STS2_RS_SRC "$WHEEL_VENV" path 2>/dev/null)" || venv=""
    [[ -n $venv && -x $venv/bin/python ]] && cands+=("$venv/bin/python")
    command -v python3 >/dev/null 2>&1 && cands+=(python3)
    if command -v uv >/dev/null 2>&1 && py="$(uv python find '>=3.11' 2>/dev/null)"; then
        cands+=("$py")
    fi
    for py in "${cands[@]}"; do
        "$py" -I -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null || continue
        "$py" -I "$PYCLI_DIR/scripts/uv_stamp_wheel.py" key "$SRC"
        return  # its status: another interpreter would fail the same way
    done
    return 1
}
if ! UV_KEY="$(uv_key)"; then
    UV_KEY=""
    echo "pgo-wheel: WARNING: could not take uv's freshness key (see any error above; it needs" \
        "Python >= 3.11: the venv's, python3, or one \`uv python find\` knows). The build goes" \
        "on, but the wheel will not be stamped for uv, so the next \`uv run\` / \`uv sync\`" \
        "replaces it with a plain build (which the trainer refuses)" >&2
fi

echo "== pgo-wheel phase 0: plain wheel + venv (wheel-venv.sh build) ==" >&2
# Unset the delegation env so wheel-venv.sh does a normal plain build here
# (STS2_WHEEL_PGO=1 makes it call back into this script — see wheel-venv.sh),
# and make it replace an installed PGO .so that it would otherwise keep: the
# identity baseline and the fallback have to be the plain build.
# STS2_RS_SRC unset: wheel-venv.sh prefers it over the cwd, so an exported one
# would build another checkout's venv than the one the staged builds come from.
VENV="$(cd "$SRC" && env -u STS2_RS_SRC STS2_WHEEL_PGO= WHEEL_VENV_REPLACE_PGO=1 \
    "$WHEEL_VENV" build)"
echo "pgo-wheel: venv = $VENV" >&2
check_stamp "" 0 ""

mapfile -t SOS < <(find "$VENV" -path '*/site-packages/sts2_rs/sts2_rs*.so')
if [[ ${#SOS[@]} -ne 1 ]]; then
    echo "pgo-wheel: expected one installed sts2_rs .so under $VENV, found ${#SOS[@]}: ${SOS[*]}" >&2
    exit 1
fi
INSTALLED_SO="${SOS[0]}"
SO_NAME="$(basename "$INSTALLED_SO")"
# Temp names an earlier run's install or rollback left behind if a SIGKILL
# skipped its EXIT trap. Nothing else uses them, and a pin would copy them.
rm -f "$INSTALLED_SO.pgo-tmp" "$INSTALLED_SO.tmp"
# The plain baseline every later step uses: identity A imports this copy, and a
# failure after the install puts it back.
rm -rf "$STAGE"
mkdir -p "$STAGE/plain/sts2_rs"
cp -p "$(dirname "$INSTALLED_SO")/__init__.py" "$INSTALLED_SO" "$STAGE/plain/sts2_rs/"
check_stamp "$STAGE/plain" 0 ""
PLAIN_SO="$STAGE/plain/sts2_rs/$SO_NAME"
clean_targets() {
    local dir
    for dir in "$SRC/target-pgo-wheel-gen" "$SRC/target-pgo-wheel-use"; do
        cargo clean --manifest-path "$MANIFEST" --target-dir "$dir" >&2 || true
    done
}
# Set right before the venv's .so is replaced; until then a failure leaves the
# venv alone (it still has the plain wheel, or whatever a concurrent build put
# there since, which a restore must not clobber).
INSTALLING=0
on_failure() {
    if (( INSTALLING )); then
        rm -f "$INSTALLED_SO.pgo-tmp"
        # Undo only our own install: before the rename, the venv still holds
        # what it held (perhaps a concurrent build's), and that stays.
        if cmp -s "$INSTALLED_SO" "$STAGE/use/sts2_rs/$SO_NAME"; then
            cp "$PLAIN_SO" "$INSTALLED_SO.tmp" && mv -f "$INSTALLED_SO.tmp" "$INSTALLED_SO"
            echo "pgo-wheel: FAILED — the plain wheel is back in $VENV" >&2
        else
            echo "pgo-wheel: FAILED before the PGO .so replaced $VENV's; left that as it is" >&2
        fi
    else
        echo "pgo-wheel: FAILED — nothing was installed; $VENV keeps its plain wheel" >&2
    fi
    rm -rf "$STAGE"
    clean_targets
}
trap 'on_failure; release_lock' EXIT

rm -rf "$IDENTITY_DIR" "$LOG_DIR"
if [[ $SKIP_IDENTITY -eq 0 ]]; then
    echo "== pgo-wheel: identity traces A (plain wheel; seeds: $IDENTITY_SEEDS) ==" >&2
    run_shards identity-plain "$STAGE/plain" "$IDENTITY_SEEDS" "$IDENTITY_DIR/plain"
fi

rm -rf "$PGO_DIR" "$PROFDATA"

echo "== pgo-wheel phase 1: instrumented build ==" >&2
build_stage gen "-Cprofile-generate=$PGO_DIR $EXTRA_CPU" "$SRC/target-pgo-wheel-gen" \
    STS2_WHEEL_PGO_INSTRUMENTED=1 "${CPU_STAMP[@]}"
check_stamp "$STAGE/gen" 0 ""
# RUSTFLAGS also instrumented the build scripts and proc-macros, and running
# them during the build (and check_stamp's import) already wrote ~50 .profraw
# files here. Drop them so the profile is the workload's alone (the runtime
# recreates the dir).
rm -rf "$PGO_DIR"

echo "== pgo-wheel phase 2: profile workload ($AGENT; seeds: $SEEDS, sims=$SIMS/$BOSS_SIMS) ==" >&2
run_shards profile "$STAGE/gen" "$SEEDS"

echo "== pgo-wheel phase 3: merge + optimized build ==" >&2
"$LLVM_PROFDATA" merge -o "$PROFDATA" "$PGO_DIR"
PROFDATA_SHA="$("$VENV/bin/python" -I -c \
    "import hashlib, sys; print(hashlib.file_digest(open(sys.argv[1], 'rb'), 'sha256').hexdigest())" \
    "$PROFDATA")"
build_stage use "-Cprofile-use=$PROFDATA $EXTRA_CPU" "$SRC/target-pgo-wheel-use" \
    STS2_WHEEL_PGO_PROFDATA_SHA256="$PROFDATA_SHA" "${CPU_STAMP[@]}"
check_stamp "$STAGE/use" 1 "$PROFDATA_SHA"

# Freshness gate — same guard wheel-venv.sh runs after its builds, on the
# staged copy. The gate is content-addressed on SOURCE, so a PGO .so from the
# same source passes.
if [ "${STS2_SKIP_WHEEL_CHECK:-0}" != "1" ]; then
    PYTHONPATH="$STAGE/use:$PYCLI_DIR/python" "$VENV/bin/python" -s -c \
        "import sys, sts2_rs, parity_audit
assert sts2_rs.__file__.startswith(sys.argv[2] + '/'), sts2_rs.__file__
parity_audit.assert_wheel_fresh(sys.argv[1])" "$SRC" "$STAGE/use" >&2
    echo "pgo-wheel: freshness verified ✓" >&2
fi

if [[ $SKIP_IDENTITY -eq 1 ]]; then
    rm -rf "$UNVERIFIED"
    mv "$STAGE/use" "$UNVERIFIED"
    trap release_lock EXIT
    rm -rf "$STAGE" "$LOG_DIR" "$PGO_DIR" "$PROFDATA"
    clean_targets
    echo "pgo-wheel: --skip-identity: NOT installed; $VENV keeps its plain wheel." >&2
    echo "pgo-wheel: the unverified PGO module is in $UNVERIFIED (PYTHONPATH it to measure; delete it after)" >&2
    echo "$UNVERIFIED"
    exit 0
fi

echo "== pgo-wheel: identity traces B (PGO wheel) ==" >&2
run_shards identity-pgo "$STAGE/use" "$IDENTITY_SEEDS" "$IDENTITY_DIR/pgo"
n_runs=$(find "$IDENTITY_DIR/plain" -name '*.json' | wc -l)
n_pgo=$(find "$IDENTITY_DIR/pgo" -name '*.json' | wc -l)
n_want=$(( $(wc -w <<<"$CHARACTERS") * $(wc -w <<<"$IDENTITY_SEEDS") ))
if (( n_want == 0 || n_runs != n_want || n_pgo != n_want )); then
    echo "pgo-wheel: the identity check has $n_runs plain / $n_pgo PGO traces, expected" \
        "$n_want (characters x identity seeds); not installing" >&2
    exit 1
fi
if diff -rq "$IDENTITY_DIR/plain" "$IDENTITY_DIR/pgo" >&2; then
    echo "pgo-wheel: identity ✓ — PGO wheel decision-identical to plain ($n_runs full runs)" >&2
    rm -rf "$IDENTITY_DIR"
else
    echo "pgo-wheel: IDENTITY MISMATCH — PGO wheel diverges from plain release!" >&2
    echo "pgo-wheel: diff -r $IDENTITY_DIR/plain $IDENTITY_DIR/pgo" >&2
    exit 1
fi

echo "== pgo-wheel: install the verified PGO .so ==" >&2
# The source must still be what the PGO build compiled: identity B took
# minutes, and an edit since (plus a `just sim-engine` into this venv) must not
# be overwritten by a PGO build of the older source. Not skippable: the install
# is the step it protects. INSTALLING is still 0, so a refusal leaves the venv.
if ! PYTHONPATH="$STAGE/use" "$VENV/bin/python" -s - "$SRC" "$STAGE/use" >&2 <<'PY'
import sys

import sts2_rs

src, stage = sys.argv[1:]
assert sts2_rs.__file__.startswith(stage + "/"), sts2_rs.__file__
built, now = sts2_rs.src_hash(), sts2_rs.recompute_src_hash(src)
if built != now:
    sys.exit(f"pgo-wheel: the source changed during the build (built {built}, now {now})")
PY
then
    echo "pgo-wheel: not installing; $VENV keeps what it has. Re-run \`just sim-engine-pgo\`" \
        "on the current source." >&2
    exit 1
fi
# Atomic: a copy under a temp name in the same dir, then a rename over the
# plain .so. A process importing sts2_rs meanwhile gets one or the other whole.
INSTALLING=1
cp "$STAGE/use/sts2_rs/$SO_NAME" "$INSTALLED_SO.pgo-tmp"
mv -f "$INSTALLED_SO.pgo-tmp" "$INSTALLED_SO"
check_stamp "" 1 "$PROFDATA_SHA"
# Give uv the key from before phase 0, so `uv run` / `uv sync` keep this wheel
# until a cache-key file changes (see the header).
if [[ -z $UV_KEY ]]; then
    echo "pgo-wheel: WARNING: not stamped for uv (no key, see above) — the next \`uv run\` /" \
        "\`uv sync\` in this venv will replace the PGO wheel with a plain build" >&2
elif ! "$VENV/bin/python" -I "$PYCLI_DIR/scripts/uv_stamp_wheel.py" stamp "$SRC" "$UV_KEY" \
    --check >&2; then
    echo "pgo-wheel: WARNING: uv's metadata could not be made to match (see above) — the next" \
        "\`uv run\` / \`uv sync\` in this venv will replace the PGO wheel with a plain build" \
        "(the trainer then refuses it; re-run \`just sim-engine-pgo\`)" >&2
fi

trap release_lock EXIT
rm -rf "$STAGE" "$LOG_DIR" "$PGO_DIR" "$PROFDATA"
clean_targets
echo "pgo-wheel: done — PGO${EXTRA_CPU:++v3} wheel (profdata sha256 ${PROFDATA_SHA:0:12}…) installed in $VENV" >&2
echo "$VENV"
