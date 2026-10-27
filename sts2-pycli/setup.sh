#!/bin/bash
# setup.sh — Copy game DLLs from Steam installation to lib/,
#            then build + install the sts2_rs PyO3 extension via maturin.
#
# Prerequisites:
#   - Slay the Spire 2 installed via Steam
#   - .NET 9+ SDK (ARM64 for Apple Silicon, x64 for Intel/Linux)
#   - Rust 1.99+ toolchain (https://rustup.rs; ../sts2-rs/rust-toolchain.toml pins the exact version)
#   - maturin (`cargo install maturin` or `pipx install maturin`)
#   - sts2-pycli's uv venv synced (`just install-dev` or `uv sync --dev`)
#     and ACTIVATED before running this script (we refuse to install
#     into a wrong/unset venv — see venv guard below).
#
# Usage:
#   ./setup.sh                    # Auto-detect Steam path
#   ./setup.sh /path/to/game      # Manual game directory
#
# Skip flags (env):
#   STS2_SKIP_RUST=1              # Skip the maturin step (DLL-only setup)
#   STS2_SKIP_DOTNET=1            # Skip the .NET copy + IL-patch + build steps

set -e

# Resolve repo root (directory of this script) so relative paths work
# regardless of $PWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── .NET DLL extraction + headless build ──
if [ "${STS2_SKIP_DOTNET:-0}" = "1" ]; then
    echo "⏭  STS2_SKIP_DOTNET=1 — skipping .NET copy + build."
else

# ── Locate game directory ──

GAME_DIR="$1"

if [ -z "$GAME_DIR" ]; then
    # Auto-detect based on platform
    case "$(uname -s)" in
        Darwin)
            GAME_DIR="$HOME/Library/Application Support/Steam/steamapps/common/Slay the Spire 2/SlayTheSpire2.app/Contents/Resources/data_sts2_macos_arm64"
            if [ ! -d "$GAME_DIR" ]; then
                # Try x86_64
                GAME_DIR="$HOME/Library/Application Support/Steam/steamapps/common/Slay the Spire 2/SlayTheSpire2.app/Contents/Resources/data_sts2_macos_x86_64"
            fi
            ;;
        Linux)
            GAME_DIR="$HOME/.steam/steam/steamapps/common/Slay the Spire 2"
            if [ ! -d "$GAME_DIR" ]; then
                GAME_DIR="$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            GAME_DIR="C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2"
            ;;
    esac
fi

# Copy the DLLs into lib/, IL-patch sts2.dll and build the harness: shared with
# `just import-game` (scripts/setup_oracle.py — cross-platform, stdlib only).
# No game dir + a vendored, pre-patched lib/sts2.dll (git-lfs) = build only.
ORACLE_ARGS=()
[ -d "$GAME_DIR" ] && ORACLE_ARGS=("$GAME_DIR")
[ -n "${STS2_INSTALL_ROOT:-}" ] && ORACLE_ARGS+=(--install-root "$STS2_INSTALL_ROOT")
uv run --no-project python "$SCRIPT_DIR/scripts/setup_oracle.py" "${ORACLE_ARGS[@]}"

fi  # STS2_SKIP_DOTNET

# ──────────────────────────────────────────────────────────────────────
# Rust / sts2_rs PyO3 extension (MCTS hint backend)
# ──────────────────────────────────────────────────────────────────────
#
# Builds the sts2_rs extension out of ../sts2-rs/crates/sts2-py and
# installs it editable-in-release-mode into the active venv via
# `maturin develop --release`. This is what powers the `sts2_hint`
# package's calls into `sts2_rs.state_from_dict`, `mcts_recommend`,
# etc. See:
#   docs/plans/.../integrate-sts2-rs-mcts-hint-system…md  (cross-repo plan)
#   python/sts2_hint/                                      (consumer)
#
if [ "${STS2_SKIP_RUST:-0}" = "1" ]; then
    echo ""
    echo "⏭  STS2_SKIP_RUST=1 — skipping maturin build of sts2_rs."
else

echo ""
echo "🦀 Building sts2_rs PyO3 extension via maturin..."

# ── Venv guard ──
# Refuse to install into the wrong venv or with no venv active. We match
# on the venv path containing "sts2-pycli" (uv's default venv lives at
# $REPO/.venv, so the realpath will always contain the repo name).
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "❌ No virtualenv active."
    echo ""
    echo "   sts2_rs must be installed into sts2-pycli's venv. Activate it first:"
    echo "     cd $SCRIPT_DIR && source .venv/bin/activate"
    echo "   (or run \`just install-dev\` first if .venv doesn't exist yet)."
    echo ""
    echo "   Override with STS2_SKIP_RUST=1 if you really want to skip this."
    exit 1
fi

# Resolve symlinks (e.g. macOS /private/var) for a stable comparison.
VENV_REAL="$(cd "$VIRTUAL_ENV" 2>/dev/null && pwd -P || echo "$VIRTUAL_ENV")"
if [[ "$VENV_REAL" != *sts2-pycli* ]]; then
    echo "❌ Active venv looks wrong: $VIRTUAL_ENV"
    echo ""
    echo "   Expected a venv whose path contains 'sts2-pycli' (typically"
    echo "   \$REPO/.venv). Activate the right one and rerun, or set"
    echo "   STS2_SKIP_RUST=1 to skip."
    exit 1
fi

echo "  ✓ venv: $VIRTUAL_ENV"

# ── Locate maturin ──
if ! command -v maturin &>/dev/null; then
    echo "❌ maturin not found on PATH."
    echo "   Install with one of:"
    echo "     cargo install maturin"
    echo "     pipx install maturin"
    echo "     pip install maturin   # (into the active venv)"
    exit 1
fi

# ── Pick an uninstall driver ──
# uv-managed venvs don't ship pip; fall back to `uv pip` when present.
# Order: `uv pip` (sts2-pycli is uv-based) > `pip` > `python -m pip`.
PIP_UNINSTALL=()
if command -v uv &>/dev/null; then
    PIP_UNINSTALL=(uv pip uninstall)
elif command -v pip &>/dev/null && python -c "import pip" &>/dev/null; then
    PIP_UNINSTALL=(pip uninstall -y)
elif python -m pip --version &>/dev/null; then
    PIP_UNINSTALL=(python -m pip uninstall -y)
fi

# ── Stale-wheel uninstall ──
# A previously-installed wheel named sts2_rs (or sts2-rs) — from pip,
# uv, or an earlier `maturin build`+`pip install` — will shadow the
# editable build produced by `maturin develop` and cause confusing
# import behavior. Uninstall both spellings unconditionally; ignore
# failures (nothing-to-uninstall is the common case).
if [ "${#PIP_UNINSTALL[@]}" -gt 0 ]; then
    echo "  🧹 Removing any stale sts2_rs install in the venv (via '${PIP_UNINSTALL[*]}')..."
    "${PIP_UNINSTALL[@]}" sts2_rs 2>/dev/null || true
    "${PIP_UNINSTALL[@]}" sts2-rs 2>/dev/null || true
else
    echo "  ⚠ No pip/uv pip available to run stale-wheel uninstall — skipping."
    echo "    (maturin will still install, but a pre-existing wheel could shadow it.)"
fi

# ── Resolve the sts2-rs PyO3 crate ──
# Relative to this script: ../sts2-rs/crates/sts2-py
STS2_PY_MANIFEST="$SCRIPT_DIR/../sts2-rs/crates/sts2-py/Cargo.toml"
if [ ! -f "$STS2_PY_MANIFEST" ]; then
    echo "❌ sts2-rs PyO3 manifest not found: $STS2_PY_MANIFEST"
    echo "   Expected the sts2-rs repo at ../sts2-rs (sibling of sts2-pycli)."
    echo "   Clone https://github.com/<...>/sts2-rs alongside this repo, or"
    echo "   set STS2_SKIP_RUST=1 to skip the hint backend."
    exit 1
fi

# ── Build + install ──
echo "  🔨 maturin develop --release -m $STS2_PY_MANIFEST"
maturin develop --release -m "$STS2_PY_MANIFEST"

# ── Smoke check ──
echo "  🔎 Verifying import + version..."
python -c "import sts2_rs; v = sts2_rs.version(); assert v, 'empty version'; print('     sts2_rs version:', v)"

fi  # STS2_SKIP_RUST

echo ""
echo "✅ Setup complete!"
echo ""
echo "To play:"
echo "  python3 python/play.py"
echo ""
echo "To run batch games:"
echo "  python3 python/play_full_run.py 10"
