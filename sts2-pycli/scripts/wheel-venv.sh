#!/usr/bin/env bash
# wheel-venv.sh — per-checkout venv isolation for the sts2_rs PyO3 wheel.
#
# THE PROBLEM THIS SOLVES
# -----------------------
# The flywheel swarm runs several sts2-rs *worktree* agents at once, each at a
# different commit. When any of them needs the Python parity harness it runs
# `maturin develop --release` to build its `sts2_rs` extension — and they all
# install into the SINGLE shared `sts2-pycli/.venv`. Each build overwrites the
# venv's `sts2_rs*.so` AND rewrites `direct_url.json` to point at whichever
# checkout built last. The coordinator's main-checkout wheel silently becomes
# a sibling worktree's, two commits behind, and the next parity sweep buckets
# garbage (the recurring "stale wheel" pain). `build-mutex.sh` serializes the
# builds so they don't corrupt the cargo target dir, but serialization does
# NOT stop the clobber — whoever builds last still wins the one venv.
# `assert_wheel_fresh()` in parity_audit.py *detects* the clobber after the
# fact; this script *prevents* it.
#
# THE FIX
# -------
# Route each checkout's wheel into its OWN venv, keyed by the checkout:
#   * the canonical MAIN sts2-rs checkout  -> sts2-pycli/.venv   (unchanged;
#     so setup.sh / the coordinator / interactive use keep working as before)
#   * any linked WORKTREE                  -> sts2-pycli/.venv-h/<engine-hash>
# A worktree venv is lightweight: the parity harness imports only stdlib +
# `sts2_rs` + the local `python/` modules (no numpy/torch), so the venv just
# needs a Python and the worktree's own `.so`.
#
# WORKTREE VENVS ARE CONTENT-ADDRESSED (issue #15 Phase 2). The `<engine-hash>`
# is the 16-hex aggregate source hash of the worktree's engine crates — the
# SAME value the wheel bakes into `sts2_rs.src_hash()`, here computed from
# SOURCE (before any `.so` exists) by the dep-free `sts2-srchash` CLI. Two
# properties fall out for free:
#   * STALE IS UNREACHABLE — change the engine source and the venv path moves to
#     a hash that does not exist yet, forcing a fresh build; you can never run a
#     venv whose `.so` predates your edit.
#   * NO BASENAME COLLISIONS — the old `.venv-wt/<basename>` keying gave two
#     worktrees with the same dir name ONE shared venv, clobbering each other
#     (the very race this script exists to stop). Identical source now safely
#     SHARES a venv; divergent source never collides.
# `.venv-h/<hash>` dirs are reaped by `gc` (and after every worktree build) once
# no live checkout hashes to them — see `reachable_prune`.
#
# USAGE
#   scripts/wheel-venv.sh path  [SRC]            # print the venv dir for SRC
#   scripts/wheel-venv.sh build [SRC]            # build SRC's wheel into its venv; print the dir
#   scripts/wheel-venv.sh run   [SRC] -- CMD...  # build-if-stale then run CMD in SRC's venv
#   scripts/wheel-venv.sh gc                     # prune .venv-h/<hash> venvs no live checkout needs
#
# SRC is the sts2-rs checkout to build from. Default: the git toplevel that
# owns $PWD if it is an sts2-rs checkout, else the sibling ../sts2-rs. When you
# run this from inside a worktree, SRC auto-resolves to that worktree.
#
# `run` activates the resolved venv (VIRTUAL_ENV + PATH), puts sts2-pycli's
# `python/` on PYTHONPATH (the harness modules live there), and execs CMD from
# the sts2-pycli repo root — so `scripts/wheel-venv.sh run -- python
# python/parity_audit.py --coverage` Just Works from any worktree against that
# worktree's engine.
#
# PGO WHEELS. `STS2_WHEEL_PGO=1` (`just sim-engine-pgo`) builds the PGO wheel
# into the same venv via sts2-rs/scripts/pgo_wheel.sh; the PPO trainer refuses
# anything else. A plain `build` / `run` KEEPS an installed PGO .so whose
# `src_hash` matches the current source instead of reinstalling the plain one
# over it (maturin develop reinstalls even when nothing changed), and a PGO
# build is a no-op when such a wheel is already installed. A PGO .so that is
# stale for the source is rebuilt as usual (plain unless STS2_WHEEL_PGO=1).
# `src_hash` covers every workspace crate linked into sts2_rs, the workspace
# Cargo.toml, Cargo.lock and the toolchain pin. The PGO install also writes
# uv's freshness key, taken before that build started, into the dist-info
# (scripts/uv_stamp_wheel.py), so `uv run` / `uv sync` keep the PGO wheel until
# a file uv's cache-keys name changes after that point. Keeping a wheel here
# does not re-key uv. Plain builds are not stamped: uv still swaps them for its
# own cached plain build (harmless, same source).
#
# ENV
#   STS2_RS_SRC        override the source checkout (same as the SRC arg)
#   WHEEL_VENV_NO_BUILD=1  `run` skips the implicit build (use the venv as-is)
#   STS2_WHEEL_PGO=1   build the PGO wheel (sts2-rs/scripts/pgo_wheel.sh)
#   WHEEL_VENV_REPLACE_PGO=1  rebuild even over a fresh installed PGO .so
#                      (plain build, or a new PGO build with STS2_WHEEL_PGO=1)
#   WHEEL_VENV_NO_GC=1  skip the `.venv-h/<hash>` GC after a worktree build
set -euo pipefail

PYCLI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# The canonical main sts2-rs checkout is the sibling of this sts2-pycli dir
# (monorepo layout: <root>/sts2-rs + <root>/sts2-pycli). Derived from
# PYCLI_DIR so it is portable across clones — do NOT hardcode an absolute path.
CANONICAL_MAIN="$(cd "$PYCLI_DIR/../sts2-rs" 2>/dev/null && pwd -P || echo "$PYCLI_DIR/../sts2-rs")"

die() { echo "wheel-venv: $*" >&2; exit 1; }

# Resolve the sts2-rs checkout to operate on.
#   1. explicit SRC arg / $STS2_RS_SRC
#   2. the git toplevel owning $PWD, if that toplevel looks like sts2-rs
#   3. the canonical sibling main checkout
resolve_src() {
  local src="${1:-${STS2_RS_SRC:-}}"
  if [ -z "$src" ]; then
    local top
    top="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "$top" ] && [ -f "$top/crates/sts2-py/Cargo.toml" ]; then
      # $PWD is inside an sts2-rs checkout/worktree (its own toplevel).
      src="$top"
    elif [ -n "$top" ] && [ -f "$top/sts2-rs/crates/sts2-py/Cargo.toml" ]; then
      # Monorepo layout: the git toplevel is the umbrella repo and sts2-rs is a
      # subdirectory, so --show-toplevel does not point at the crate root.
      src="$top/sts2-rs"
    else
      src="$CANONICAL_MAIN"
    fi
  fi
  [ -f "$src/crates/sts2-py/Cargo.toml" ] || die "no sts2-py crate under SRC: $src"
  ( cd "$src" && pwd -P )
}

# Content-address an sts2-rs checkout by its engine source hash — the SAME
# 16-hex aggregate the wheel bakes into `sts2_rs.src_hash()`, here computed from
# SOURCE (no wheel needed) by the dep-free `sts2-srchash` CLI. Printed on
# stdout; returns non-zero (with a stderr note) if cargo can't produce it, so
# callers choose to fail hard (venv selection) or skip (best-effort GC).
engine_hash() {
  local src="$1" out
  if ! out="$(cd "$src" && cargo run --quiet --release -p sts2-srchash -- "$src")"; then
    echo "wheel-venv: could not compute engine hash for $src" >&2
    return 1
  fi
  printf '%s\n' "$out"
}

# Map a checkout to its dedicated venv dir.
#   main  -> $PYCLI/.venv                 (heavy; setup.sh / interactive)
#   wt    -> $PYCLI/.venv-h/<engine-hash> (content-addressed; see header)
# Worktree detection: a linked worktree's --git-dir differs from its
# --git-common-dir (the common dir is the main repo's .git).
venv_for() {
  local src="$1"
  local gitdir commondir
  gitdir="$(git -C "$src" rev-parse --absolute-git-dir 2>/dev/null || true)"
  commondir="$(cd "$src" && git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  if [ -n "$gitdir" ] && [ -n "$commondir" ] && [ "$gitdir" != "$commondir" ]; then
    # linked worktree → content-addressed venv (fail the caller if unhashable)
    local h
    h="$(engine_hash "$src")" || return 1
    echo "$PYCLI_DIR/.venv-h/$h"
  else
    echo "$PYCLI_DIR/.venv"
  fi
}

ensure_venv() {
  local venv="$1"
  if [ ! -x "$venv/bin/python" ]; then
    echo "wheel-venv: creating venv $venv" >&2
    if command -v uv >/dev/null 2>&1; then
      uv venv "$venv" >&2
    else
      python3 -m venv "$venv" >&2
    fi
  fi
}

# Provision the sts2-pycli APP dependencies (fastapi/uvicorn/pydantic/…) into a
# venv. `build_into` only installs the `sts2_rs` .so via maturin — it never
# installs the app stack — so a freshly-created content-addressed worktree venv
# can run parity sweeps but NOT `python -m workbench`. We `uv sync` the project
# (into this exact venv, by abs path so the `.venv` symlink is never rewritten
# by uv) the first time the app deps are missing. Cheap no-op once present.
# Caller MUST run the maturin engine build AFTER this — `uv sync` can install a
# stale CACHED editable `sts2_rs`, which the subsequent maturin+freshness gate
# then overwrites and verifies. uv-only (the dep set is uv-managed); warns if
# uv is absent rather than failing the engine build.
ensure_app_deps() {
  local venv="$1"
  ensure_venv "$venv"
  "$venv/bin/python" -c "import uvicorn, fastapi, pydantic" >/dev/null 2>&1 && return 0
  if ! command -v uv >/dev/null 2>&1; then
    echo "wheel-venv: uv not found — cannot provision app deps into $venv (workbench/simlab won't import)" >&2
    return 0
  fi
  echo "wheel-venv: provisioning app deps into $venv (uv sync)…" >&2
  ( cd "$PYCLI_DIR" && UV_PROJECT_ENVIRONMENT="$venv" uv sync >&2 ) \
    || echo "wheel-venv: uv sync failed — $venv may lack app deps" >&2
}

# Point the worktree's `$PYCLI/.venv` symlink at the freshly-built
# content-addressed venv, so the justfile's hardcoded `.venv/bin/python`
# launches THIS engine. Only ever rewrites a SYMLINK under a worktree checkout;
# the main checkout's `.venv` is a real heavy dir (not under `.venv-h/`) and is
# left untouched. Without this, a rebase that moves the engine hash leaves
# `.venv` pointing at the previous hash → the stale-wheel guard fires at launch.
relink_dot_venv() {
  local venv="$1"
  # Only content-addressed worktree venvs; main's `.venv` IS the venv → skip.
  case "$venv" in "$PYCLI_DIR/.venv-h/"*) ;; *) return 0 ;; esac
  local dot="$PYCLI_DIR/.venv"
  if [ -e "$dot" ] && [ ! -L "$dot" ]; then
    echo "wheel-venv: $dot is a real directory (not a symlink) — leaving as-is" >&2
    return 0
  fi
  local target=".venv-h/$(basename "$venv")"   # relative, matches existing convention
  ln -sfn "$target" "$dot"
  echo "wheel-venv: .venv -> $target" >&2
}

# Verify the installed extension actually matches the source: reuse the
# maintained parity_audit.assert_wheel_fresh() guard, passing the resolved
# source explicitly so the check is content-addressed against THIS checkout and
# never depends on the wheel's `direct_url.json` absolute path (issue #15).
# Returns 0 fresh, non-zero stale (with the reason on stdout). Honors
# STS2_SKIP_WHEEL_CHECK.
verify_fresh() {
  local venv="$1" src="$2"
  [ "${STS2_SKIP_WHEEL_CHECK:-0}" = "1" ] && return 0
  PYTHONPATH="$PYCLI_DIR/python" "$venv/bin/python" \
    -c "import sys, parity_audit; parity_audit.assert_wheel_fresh(sys.argv[1])" "$src" 2>&1
}

# 0 iff the venv's installed sts2_rs is a PGO build (`build_info()["pgo"]`)
# compiled from SRC's current source (`src_hash` == `recompute_src_hash`).
# Not verify_fresh: keeping a .so is only safe when it is PROVEN fresh, so
# STS2_SKIP_WHEEL_CHECK does not apply here, and any error (no venv, a wheel
# that predates build_info) means "no".
installed_pgo_fresh() {
  local venv="$1" src="$2"
  [ -x "$venv/bin/python" ] || return 1
  "$venv/bin/python" -I - "$src" >/dev/null 2>&1 <<'PY'
import sys

import sts2_rs

info = sts2_rs.build_info()
fresh = info["pgo"] and sts2_rs.src_hash() == sts2_rs.recompute_src_hash(sys.argv[1])
sys.exit(0 if fresh else 1)
PY
}

build_into() {
  local src="$1" venv="$2"
  ensure_venv "$venv"
  # Keep a PGO wheel that is fresh for this source. `maturin develop`
  # reinstalls its plain .so even when nothing changed, so without this every
  # `just sim*` / `workbench` call (via `_sim-engine-fresh`) silently swapped a
  # `just sim-engine-pgo` wheel for a plain one. A stale PGO .so falls through
  # to the rebuild below (plain unless STS2_WHEEL_PGO=1), and the PPO trainer
  # then refuses the plain wheel until `just sim-engine-pgo` is re-run.
  if [ "${WHEEL_VENV_REPLACE_PGO:-0}" != "1" ] && installed_pgo_fresh "$venv" "$src"; then
    echo "wheel-venv: keeping the installed PGO wheel — fresh for $src (WHEEL_VENV_REPLACE_PGO=1 rebuilds)" >&2
    return 0
  fi
  # PGO wheel (STS2_WHEEL_PGO=1, `just sim-engine-pgo`): delegate the whole
  # build to scripts/pgo_wheel.sh, which runs the two-phase PGO recipe into
  # this same venv (plain baseline build, instrumented build + profile
  # workload, optimized rebuild, freshness + build_info stamp gates,
  # decision-identity check). It calls back into THIS script with
  # STS2_WHEEL_PGO unset for the plain phase, so there is no recursion. Not the
  # default: ~2 extra cold LTO compiles per build plus the profile and identity
  # workloads. PPO training requires it (python/ppo/wheel_guard.py). The .so
  # stays portable (no -Ctarget-cpu unless pgo_wheel.sh gets --v3).
  if [ "${STS2_WHEEL_PGO:-0}" = "1" ]; then
    [ -x "$src/scripts/pgo_wheel.sh" ] || die "STS2_WHEEL_PGO=1 but $src/scripts/pgo_wheel.sh is missing"
    echo "wheel-venv: STS2_WHEEL_PGO=1 → PGO wheel via $src/scripts/pgo_wheel.sh" >&2
    STS2_WHEEL_PGO= "$src/scripts/pgo_wheel.sh" --src "$src" >/dev/null
    return 0
  fi
  local manifest="$src/crates/sts2-py/Cargo.toml"
  # Serialize against sibling builds (shared cargo target dir) via the
  # project mutex. build-mutex.sh lives in each sts2-rs checkout and they all
  # share the same default lock (/tmp/sts2-rs-build.lock), so any copy
  # serializes against every other build. Fall back to a bare maturin call if
  # the script is missing. Maturin installs the .so + dist-info directly into
  # VIRTUAL_ENV's site-packages; it does not need pip in the target venv.
  local mutex="$src/scripts/build-mutex.sh"
  local -a prefix=()
  [ -x "$mutex" ] && prefix=("$mutex")
  # The `build_info()` PGO stamps are pgo_wheel.sh's alone: strip any that
  # leaked into the caller's env so a plain build can never claim PGO.
  # Run from $src: rustup picks the toolchain from the cwd's
  # rust-toolchain.toml, and the caller's cwd (sts2-pycli) has none, so the
  # build otherwise compiles with the default toolchain instead of the pin.
  run_maturin() {
    echo "wheel-venv: maturin develop --release ($src) -> $venv" >&2
    ( cd "$src" && env -u STS2_WHEEL_PGO_PROFDATA_SHA256 -u STS2_WHEEL_PGO_INSTRUMENTED \
      -u STS2_WHEEL_TARGET_CPU VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH" \
      "${prefix[@]}" maturin develop --release -m "$manifest" >&2 )
  }
  run_maturin

  # Post-build freshness GATE — a green build must not lie. maturin can leave a
  # stale .so (incremental race) and `uv sync --reinstall-package` can reinstall
  # a cached wheel from an older commit; both have silently shipped outdated
  # engines for months. If the install is stale, self-heal once (rm the .so so
  # maturin is forced to relink) then re-verify; hard-fail if still stale.
  local out
  if ! out="$(verify_fresh "$venv" "$src")"; then
    echo "wheel-venv: post-build check says the wheel is STALE — self-healing (rm .so + recompile)…" >&2
    printf '%s\n' "$out" >&2
    rm -f "$venv"/lib/python*/site-packages/sts2_rs/*.so 2>/dev/null || true
    run_maturin
    if ! out="$(verify_fresh "$venv" "$src")"; then
      printf '%s\n' "$out" >&2
      die "installed sts2_rs is STILL stale after rm+recompile. The source crates may not have recompiled — try:
   cargo clean --manifest-path $src/Cargo.toml -p sts2-py && scripts/wheel-venv.sh build"
    fi
  fi
  echo "wheel-venv: freshness verified ✓ ($src)" >&2

  # Opportunistic GC: a content-addressed (re)build means this worktree's
  # PREVIOUS hash (if its source moved) is now orphaned — collect it and any
  # other `.venv-h/<H>` no live checkout needs. Only after a worktree build
  # (main lives at `.venv`, outside `.venv-h`). Never fail a good build over GC.
  # WHEEL_VENV_NO_GC=1 keeps them (GC also hashes every other live checkout).
  case "$venv" in
    "$PYCLI_DIR/.venv-h/"*) [ "${WHEEL_VENV_NO_GC:-0}" = "1" ] || reachable_prune || true ;;
  esac
}

# Resolve a `git worktree list` path to its sts2-rs checkout dir. This repo is a
# monorepo, so a worktree root usually holds sts2-rs in a `sts2-rs/` subdir;
# a bare sts2-rs clone holds the crate at the root. Non-zero if it is neither
# (a non-engine worktree, or a dir already torn down).
rs_checkout_of() {
  local wt="$1"
  if [ -f "$wt/crates/sts2-py/Cargo.toml" ]; then
    echo "$wt"
  elif [ -f "$wt/sts2-rs/crates/sts2-py/Cargo.toml" ]; then
    echo "$wt/sts2-rs"
  else
    return 1
  fi
}

# Remove every $hroot/<dir> whose basename is NOT in the reachable set passed as
# the remaining args. Pure set logic (no git/cargo) — the deletion-safety core,
# unit-tested via the `gc-prune` subcommand. A reachable venv is NEVER removed.
prune_unreachable() {
  local hroot="$1"; shift
  [ -d "$hroot" ] || return 0
  local -A keep=()
  local h d base
  for h in "$@"; do keep["$h"]=1; done
  for d in "$hroot"/*/; do
    [ -d "$d" ] || continue
    base="$(basename "$d")"
    [ -n "${keep[$base]:-}" ] && continue
    echo "wheel-venv: gc removing orphan venv $d" >&2
    rm -rf "$d"
  done
}

# Prune `.venv-h/<hash>` venvs that no currently-live checkout needs. Reachable
# = the engine hash of main + every linked worktree (`git worktree list`).
# CONSERVATIVE: if any live checkout refuses to hash, skip the whole sweep
# rather than risk deleting a venv still in use; likewise if there are no
# anchors at all. Best-effort; safe from the WorktreeRemove hook (exit ignored).
reachable_prune() {
  local hroot="$PYCLI_DIR/.venv-h"
  [ -d "$hroot" ] || return 0
  local list
  list="$(git -C "$CANONICAL_MAIN" worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}')" || return 0
  local -a hashes=()
  local wt rs h
  while IFS= read -r wt; do
    [ -n "$wt" ] || continue
    rs="$(rs_checkout_of "$wt")" || continue   # gone / non-engine worktree → unreachable
    if ! h="$(engine_hash "$rs")"; then
      echo "wheel-venv: gc skipped (couldn't hash live checkout $rs)" >&2
      return 0
    fi
    hashes+=("$h")
  done <<< "$list"
  if [ "${#hashes[@]}" -eq 0 ]; then
    echo "wheel-venv: gc skipped (no live checkout to anchor reachability)" >&2
    return 0
  fi
  prune_unreachable "$hroot" "${hashes[@]}"
}

# Guardrail (issue #15 Trap 1): refuse to SILENTLY auto-run the shared main
# `.venv`. The footgun: you `build` a worktree's engine (-> .venv-h/<hash>) but
# then `run` with no SRC from the canonical sts2-pycli dir, whose auto-resolve
# falls back to canonical main `.venv` — so the harness tests the CANONICAL
# engine, not your worktree's, and reports success. Fire ONLY in the genuinely
# ambiguous case: no explicit SRC, the resolved venv IS the shared main `.venv`,
# the caller is NOT standing inside the main checkout, AND isolated worktree
# venvs exist. Bypass: pass a SRC, set STS2_RS_SRC, or WHEEL_VENV_ALLOW_MAIN=1.
guard_ambiguous_run() {
  local src="$1" venv="$2" explicit="$3"
  [ -n "$explicit" ] && return 0                          # caller named a SRC → unambiguous
  [ "${WHEEL_VENV_ALLOW_MAIN:-0}" = "1" ] && return 0     # caller confirmed main
  [ "$venv" = "$PYCLI_DIR/.venv" ] || return 0            # not the shared main venv → fine
  case "$(pwd -P)/" in "$src/"*) return 0 ;; esac         # standing inside main → you mean main
  local wt found=0
  for wt in "$PYCLI_DIR"/.venv-h/*/; do
    [ -d "$wt" ] && { found=1; break; }
  done
  [ "$found" = "1" ] || return 0                          # no worktree venvs → no ambiguity
  die "ambiguous SRC — refusing to auto-run the shared main venv.
   cwd:  $(pwd -P)
   venv: $venv  (canonical main)
 No checkout was passed, you are not inside the main sts2-rs checkout, and
 isolated worktree venvs exist under $PYCLI_DIR/.venv-h/. 'run' would silently
 test the CANONICAL engine — not whatever worktree you just built (issue #15
 Trap 1). Say which engine you mean:
   scripts/wheel-venv.sh run /abs/path/to/worktree/sts2-rs -- <cmd>     # a worktree
   STS2_RS_SRC=/abs/path/to/worktree/sts2-rs scripts/wheel-venv.sh run -- <cmd>
   WHEEL_VENV_ALLOW_MAIN=1 scripts/wheel-venv.sh run -- <cmd>           # really main"
}

# Loud binding banner (issue #15): a wrong/stale engine is silent today. Print
# the exact engine we resolved — checkout, venv, and installed .so + build time
# — to stderr on every build/run so "I am testing exactly this engine" is
# verifiable at a glance (stdout stays reserved for `run`'s exec / `build`'s
# venv-path contract).
binding_banner() {
  local src="$1" venv="$2" so so_mtime="" build=""
  so="$(ls "$venv"/lib/python*/site-packages/sts2_rs/*.so 2>/dev/null | head -1)"
  if [ -n "$so" ] && [ -e "$so" ]; then
    so_mtime="$(date -r "$so" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
    build="$("$venv/bin/python" -I -c 'import sts2_rs; f = getattr(sts2_rs, "build_info", None); i = f() if f else {}; print("PGO, profdata sha256 " + i["profdata_sha256"][:12] if i.get("pgo") else "plain")' 2>/dev/null || true)"
  else
    so="(not built yet)"
  fi
  {
    echo "wheel-venv: ───── binding ─────"
    echo "wheel-venv:   checkout = $src"
    echo "wheel-venv:   venv     = $venv"
    echo "wheel-venv:   .so      = $so${so_mtime:+  (built $so_mtime)}"
    [ -z "$build" ] || echo "wheel-venv:   build    = $build"
  } >&2
}

# Guardrail: only build YOUR OWN checkout's venv, never another agent's.
#
# Each agent owns exactly one venv: a worktree agent owns its content-addressed
# `.venv-h/<hash>`; the main-checkout agent / coordinator owns the shared
# `.venv`. The footgun is building the SHARED main `.venv` from an EXPLICITLY
# named source — e.g. a worktree session (or anyone in the pycli dir) running
# `wheel-venv.sh build /…/sts2-rs`. That overwrites the `.so` +
# `direct_url.json` that every main-checkout agent's parity sweep depends on,
# silently swapping their wheel mid-run (the recurring "stale wheel" race).
#
# Auto-detection (no SRC arg, no $STS2_RS_SRC) is always allowed: that is how
# the main agent / coordinator legitimately builds main when main is genuinely
# their checkout. We block ONLY an explicit source that resolves to the shared
# main `.venv`. A worktree agent must instead build with NO src arg (-> its own
# `.venv-h`). The coordinator can opt in deliberately with
# WHEEL_VENV_ALLOW_MAIN=1.
guard_main_venv() {
  local src="$1" venv="$2" explicit="$3"
  [ "$venv" = "$PYCLI_DIR/.venv" ] || return 0           # not the shared main venv
  [ -n "$explicit" ] || return 0                          # auto-detected → allowed
  [ "${WHEEL_VENV_ALLOW_MAIN:-0}" = "1" ] && return 0     # coordinator opt-in
  die "refusing to build the SHARED main venv from an explicitly named source.
   src:  $src
   venv: $venv
 This clobbers the wheel every main-checkout agent's parity sweep depends on
 (the stale-wheel race). If you are a worktree/worker agent, build YOUR OWN
 checkout: run 'wheel-venv.sh build' with NO src arg from your worktree
 (-> .venv-h/<hash>). If you are the coordinator and really mean to rebuild
 main, re-run with WHEEL_VENV_ALLOW_MAIN=1."
}

cmd="${1:-}"; shift || true
case "$cmd" in
  path)
    src="$(resolve_src "${1:-}")"
    venv_for "$src" || die "could not resolve venv for $src"
    ;;
  build)
    explicit_src="${1:-${STS2_RS_SRC:-}}"
    src="$(resolve_src "${1:-}")"
    venv="$(venv_for "$src")" || die "could not resolve venv for $src"
    guard_main_venv "$src" "$venv" "$explicit_src"
    # Provision app deps FIRST (uv may install a stale cached sts2_rs), then the
    # maturin engine build + freshness gate overwrites & verifies it.
    ensure_app_deps "$venv"
    build_into "$src" "$venv"
    # Point the worktree's .venv symlink at this fresh venv so the justfile's
    # hardcoded `.venv/bin/python` launches it (no-op for the main checkout).
    relink_dot_venv "$venv"
    binding_banner "$src" "$venv"
    echo "$venv"
    ;;
  run)
    # Optional leading SRC before the `--` separator.
    src_arg=""
    if [ "${1:-}" != "--" ] && [ -n "${1:-}" ]; then
      src_arg="$1"; shift
    fi
    [ "${1:-}" = "--" ] || die "run: expected '--' before the command"
    shift
    [ "$#" -gt 0 ] || die "run: no command given after '--'"
    src="$(resolve_src "$src_arg")"
    venv="$(venv_for "$src")" || die "could not resolve venv for $src"
    guard_ambiguous_run "$src" "$venv" "${src_arg:-${STS2_RS_SRC:-}}"
    if [ "${WHEEL_VENV_NO_BUILD:-0}" != "1" ]; then
      guard_main_venv "$src" "$venv" "${src_arg:-${STS2_RS_SRC:-}}"
      build_into "$src" "$venv"
    else
      # Pinned to a prebuilt venv. With content addressing the venv path encodes
      # the source hash, so "missing" means the engine source changed (or was
      # never built) — fail loudly instead of silently running an empty venv.
      if [ ! -x "$venv/bin/python" ]; then
        die "WHEEL_VENV_NO_BUILD=1 but no prebuilt venv for this source:
   $venv
 The engine source hash moved (you edited the engine) or it was never built.
 Re-run WITHOUT WHEEL_VENV_NO_BUILD to build it."
      fi
    fi
    binding_banner "$src" "$venv"
    export VIRTUAL_ENV="$venv"
    export PATH="$venv/bin:$PATH"
    export PYTHONPATH="$PYCLI_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
    cd "$PYCLI_DIR"
    exec "$@"
    ;;
  gc)
    # Prune content-addressed venvs no live checkout needs (best-effort).
    reachable_prune || true
    ;;
  gc-prune)
    # Low-level / testing: remove $1/<dir> whose basename is NOT in the
    # reachable set given as the remaining args. `gc` computes that set from
    # live checkouts; this takes it explicitly. Pure set logic, no git/cargo.
    [ "$#" -ge 1 ] || die "gc-prune: usage: gc-prune <venv-h-root> [reachable-hash...]"
    root="$1"; shift
    prune_unreachable "$root" "$@"
    ;;
  ""|-h|--help|help)
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; s/^#$//'
    ;;
  *)
    die "unknown subcommand '$cmd' (expected: path | build | run | gc)"
    ;;
esac
