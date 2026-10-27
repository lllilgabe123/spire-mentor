"""`scripts/uv_stamp_wheel.py`: write uv's freshness key for an installed
sts2_rs, so `uv run` keeps a PGO wheel until a Rust source changes.

The key has to be exactly the one uv computes (the newest ctime among the
`[tool.uv] cache-keys` files), taken before the build started, and it must
never be written for a wheel that is stale for the source or whose source
changed after the key was taken. Whether uv agrees with the mirror is checked
against uv itself by `--check` (and was, end to end, in T1b); these pin the
mirror's own rules on a synthetic tree.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import tomllib
import types
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "uv_stamp_wheel.py"
_spec = importlib.util.spec_from_file_location("uv_stamp_wheel", _SCRIPT)
assert _spec is not None and _spec.loader is not None
uv_stamp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uv_stamp)

# The real config (sts2-rs/crates/sts2-py/pyproject.toml; a test below keeps
# the two equal).
_REAL_PKG = Path(__file__).resolve().parents[2] / "sts2-rs" / "crates" / "sts2-py"
_PYPROJECT = """
[tool.uv]
cache-keys = [
    { file = "pyproject.toml" },
    { file = "Cargo.toml" },
    { file = "**/*.rs" },
    { file = "../**/*.rs" },
    { file = "../**/Cargo.toml" },
    { file = "../*/src/**/*" },
    { file = "../../data/runai/heuristic_v.toml" },
    { file = "../../data/prior_nets/*.json" },
    { file = "../../Cargo.toml" },
    { file = "../../Cargo.lock" },
    { file = "../../rust-toolchain.toml" },
]
"""


def _bump(path: Path) -> None:
    """Make `path` the most recently changed file: chmod moves its ctime (not
    its mtime). File timestamps come from a coarse clock, so wait a tick."""
    time.sleep(0.02)
    path.chmod(path.stat().st_mode)


# `_bump` moves a unix ctime; Windows has none (st_ctime_ns is the creation
# time there, and uv keys on the mtime: `test_off_unix_the_key_is_the_mtime`).
_CTIME_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="no unix ctime on Windows")


def _tree(root: Path) -> Path:
    pkg = root / "crates" / "sts2-py"
    for rel, text in {
        "crates/sts2-py/pyproject.toml": _PYPROJECT,
        "crates/sts2-py/Cargo.toml": "",
        "crates/sts2-py/src/lib.rs": "",
        "crates/sts2-core/Cargo.toml": "",
        "crates/sts2-core/src/a/b.rs": "",
        "crates/sts2-core/.hegel/gen.rs": "",
        "crates/sts2-core/notes.txt": "",
        "crates/sts2-runai/src/policy/mod.rs": "",
        "crates/sts2-vocab/src/vocab_pins/card.txt": "",
        "data/runai/heuristic_v.toml": "",
        "data/runai/other.toml": "",
        "data/prior_nets/ironclad.json": "",
        "data/prior_nets/README.md": "",
        "Cargo.toml": "",
        "Cargo.lock": "",
        "rust-toolchain.toml": "",
        "README.md": "",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return pkg


@_CTIME_ONLY
def test_the_key_is_the_newest_ctime_among_the_cache_key_files(tmp_path):
    pkg = _tree(tmp_path)
    for rel in (
        "crates/sts2-core/src/a/b.rs",
        "Cargo.lock",
        # The non-Rust inputs src_hash covers too.
        "crates/sts2-vocab/src/vocab_pins/card.txt",
        "data/runai/heuristic_v.toml",
        "data/prior_nets/ironclad.json",
        "rust-toolchain.toml",
        "crates/sts2-core/.hegel/gen.rs",
    ):
        _bump(tmp_path / rel)
        info, newest = uv_stamp.cache_info(pkg)
        assert newest.resolve() == (tmp_path / rel).resolve()
        ns = (tmp_path / rel).stat().st_ctime_ns
        assert info == {
            "timestamp": {"secs_since_epoch": ns // 10**9, "nanos_since_epoch": ns % 10**9},
            "commit": None,
            "tags": None,
            "env": {},
            "directories": {},
        }
    # Files no key names do not move it.
    for rel in (
        "crates/sts2-core/notes.txt",
        "data/runai/other.toml",
        "data/prior_nets/README.md",
        "README.md",
    ):
        _bump(tmp_path / rel)
    hegel = (tmp_path / "crates/sts2-core/.hegel/gen.rs").resolve()
    assert uv_stamp.cache_info(pkg)[1].resolve() == hegel


@pytest.mark.skipif(not _REAL_PKG.is_dir(), reason="no sts2-rs checkout next to sts2-pycli")
def test_the_synthetic_config_is_the_real_one():
    """The tree above must exercise the keys uv really uses, and the mirror
    must be able to compute every one of them (it refuses kinds it cannot)."""
    with open(_REAL_PKG / "pyproject.toml", "rb") as f:
        real = tomllib.load(f)["tool"]["uv"]["cache-keys"]
    assert tomllib.loads(_PYPROJECT)["tool"]["uv"]["cache-keys"] == real
    uv_stamp.cache_info(_REAL_PKG)


def test_off_unix_the_key_is_the_mtime(monkeypatch, tmp_path):
    """uv's Windows key: the newest mtime, which a ctime-only change (chmod on
    unix) does not move."""
    monkeypatch.setattr(uv_stamp, "KEY_ON_CTIME", False)
    pkg = _tree(tmp_path)
    lock = tmp_path / "Cargo.lock"
    later = (time.time_ns() // 10**9 + 3600) * 10**9  # whole seconds: NTFS keeps 100 ns
    os.utime(lock, ns=(later, later))
    _bump(tmp_path / "crates/sts2-core/src/a/b.rs")
    info, newest = uv_stamp.cache_info(pkg)
    assert newest.resolve() == lock.resolve()
    assert info["timestamp"] == {
        "secs_since_epoch": later // 10**9,
        "nanos_since_epoch": later % 10**9,
    }


@pytest.mark.parametrize(
    "key", ['{ dir = "src" }', "{ git = true }", '{ env = "X" }', '{ file = "src/{a,b}.rs" }']
)
def test_keys_it_cannot_mirror_refuse_instead_of_guessing(tmp_path, key):
    pkg = _tree(tmp_path)
    (pkg / "pyproject.toml").write_text(f"[tool.uv]\ncache-keys = [{key}]\n")
    with pytest.raises(SystemExit):
        uv_stamp.cache_info(pkg)


def _installed(monkeypatch, tmp_path, stamped: str, current: str) -> Path:
    """A fake installed sts2_rs whose src_hash is `stamped` while the source
    hashes to `current`; returns its dist-info dir."""
    site = tmp_path / "site-packages"
    (site / "sts2_rs").mkdir(parents=True)
    (site / "sts2_rs" / "__init__.py").write_text("")
    dist = site / "sts2_rs-0.0.1.dist-info"
    dist.mkdir()
    (dist / uv_stamp.CACHE_RECORD).write_text('{"timestamp": "the wheel file"}')
    mod = types.ModuleType("sts2_rs")
    mod.__file__ = str(site / "sts2_rs" / "__init__.py")
    mod.src_hash = lambda: stamped  # type: ignore[attr-defined]
    mod.recompute_src_hash = lambda root: current  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sts2_rs", mod)
    return dist


def _key(capsys, rs: Path) -> str:
    """What `pgo_wheel.sh` captures before it builds anything."""
    assert uv_stamp.main(["key", str(rs)]) == 0
    return capsys.readouterr().out.strip()


def test_a_fresh_wheel_gets_the_key_taken_before_the_build(monkeypatch, tmp_path, capsys):
    _tree(tmp_path / "rs")
    taken = _key(capsys, tmp_path / "rs")
    assert json.loads(taken) == uv_stamp.cache_info(tmp_path / "rs" / "crates" / "sts2-py")[0]
    dist = _installed(monkeypatch, tmp_path, stamped="aa", current="aa")
    assert uv_stamp.main(["stamp", str(tmp_path / "rs"), taken]) == 0
    assert json.loads((dist / uv_stamp.CACHE_RECORD).read_text()) == json.loads(taken)


@_CTIME_ONLY
@pytest.mark.parametrize(
    "rel", ["crates/sts2-runai/src/policy/mod.rs", "Cargo.toml", "crates/sts2-py/pyproject.toml"]
)
def test_a_change_after_the_key_was_taken_is_never_stamped(monkeypatch, tmp_path, capsys, rel):
    """A cache-key file that changes during the build -- even one the wheel's
    src_hash does not cover, as here, where src_hash agrees -- must leave uv's
    key ahead of the stamp, so the next `uv run` rebuilds from the current
    source. A key read at stamp time would have blessed the edit."""
    _tree(tmp_path / "rs")
    taken = _key(capsys, tmp_path / "rs")
    _bump(tmp_path / "rs" / rel)
    dist = _installed(monkeypatch, tmp_path, stamped="aa", current="aa")
    assert uv_stamp.main(["stamp", str(tmp_path / "rs"), taken]) == 1
    assert json.loads((dist / uv_stamp.CACHE_RECORD).read_text()) == {"timestamp": "the wheel file"}
    assert "changed after the build started" in capsys.readouterr().err


def test_a_stale_wheel_is_never_stamped(monkeypatch, tmp_path, capsys):
    """Otherwise uv would keep an engine built from older source: the stale-wheel
    trap the cache-keys exist to prevent."""
    _tree(tmp_path / "rs")
    taken = _key(capsys, tmp_path / "rs")
    dist = _installed(monkeypatch, tmp_path, stamped="old", current="new")
    with pytest.raises(SystemExit, match="not stamping"):
        uv_stamp.main(["stamp", str(tmp_path / "rs"), taken])
    assert json.loads((dist / uv_stamp.CACHE_RECORD).read_text()) == {"timestamp": "the wheel file"}


@pytest.mark.parametrize("plan", [None, " ~ sts2-rs==0.0.1 (from file:///x)"])
def test_check_fails_the_stamp_when_uv_would_still_replace_it(monkeypatch, tmp_path, capsys, plan):
    """`--check` asks uv; a stamp uv disagrees with (a uv upgrade that changed
    its key) must fail loudly rather than claim the wheel survives `uv run`."""
    _tree(tmp_path / "rs")
    taken = _key(capsys, tmp_path / "rs")
    _installed(monkeypatch, tmp_path, stamped="aa", current="aa")
    asked = []
    monkeypatch.setattr(uv_stamp, "uv_would_reinstall", lambda prefix: asked.append(prefix) or plan)
    rc = uv_stamp.main(["stamp", str(tmp_path / "rs"), taken, "--check"])
    assert rc == (0 if plan is None else 1)
    assert asked == [sys.prefix]
    err = capsys.readouterr().err
    assert ("uv still plans to replace it" in err) is (plan is not None)


def test_the_uv_cross_check_reads_both_plan_shapes():
    """The two plans `uv sync --dry-run` printed in T1b's end-to-end run: a
    reinstall of a cached build, and a build anew after a source change."""
    src = "file:///x/sts2-rs/crates/sts2-py"
    reinstall = f"Would uninstall 1 package\n ~ sts2-rs==0.0.1 (from {src})\n"
    rebuild = f" - sts2-rs==0.0.1 (from {src})\n + sts2-rs @ {src}\n"
    assert uv_stamp.planned_sts2_rs_change(reinstall) == f" ~ sts2-rs==0.0.1 (from {src})"
    assert uv_stamp.planned_sts2_rs_change(rebuild) is not None
    assert uv_stamp.planned_sts2_rs_change(f" + sts2-rs @ {src}\n") is not None
    assert uv_stamp.planned_sts2_rs_change("Checked 134 packages\nWould make no changes\n") is None
    # Another package that merely starts with the name is not sts2-rs.
    other = "Would install 1 package\n + sts2-rs-extras==1.0\n"
    assert uv_stamp.planned_sts2_rs_change(other) is None


def test_a_colored_plan_still_reads_as_a_reinstall():
    """FORCE_COLOR in the caller's env makes uv wrap the markers in ANSI codes
    (uv 0.12.23); the plan must not read as agreement then."""
    colored = " \x1b[32m+\x1b[39m \x1b[1msts2-rs\x1b[0m\x1b[2m @ file:///x\x1b[0m\n"
    assert uv_stamp.planned_sts2_rs_change(colored) == " + sts2-rs @ file:///x"


@pytest.mark.parametrize(
    "plan",
    [
        "",
        "Would reinstall: sts2-rs 0.0.1\n",
        "Would uninstall 1 package\n  ~ sts2-rs==0.0.1 (from file:///x)\n",
        "Would uninstall 1 package\n ~ sts2_rs==0.0.1 (from file:///x)\n",
        "Checked 134 packages\n",
    ],
    ids=["empty", "reworded", "indent", "underscore", "no-summary"],
)
def test_a_plan_it_does_not_recognise_fails_the_check(plan):
    """No recognised summary, or sts2-rs named in a shape the parser does not
    match: the cross-check must fail loudly, not pass vacuously."""
    why = uv_stamp.planned_sts2_rs_change(plan)
    assert why is not None
    assert why.startswith("unrecognised") or "sts2" in why


def test_the_environment_line_is_not_a_plan_entry():
    plan = "Would use project environment at: /x/sts2-rs/.venv\nWould make no changes\n"
    assert uv_stamp.planned_sts2_rs_change(plan) is None


class _Ran:
    def __init__(self, rc: int, out: str) -> None:
        self.returncode, self.stdout, self.stderr = rc, out, ""


def test_the_dry_run_asks_uv_in_plain_text_against_the_venv(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr(uv_stamp.shutil, "which", lambda name: "/bin/uv")
    seen = {}

    def run(cmd, **kw):
        seen.update(cmd=cmd, env=kw["env"])
        return _Ran(0, "Checked 134 packages\nWould make no changes\n")

    monkeypatch.setattr(uv_stamp.subprocess, "run", run)
    assert uv_stamp.uv_would_reinstall("/venv") is None
    assert seen["cmd"] == ["/bin/uv", "sync", "--frozen", "--inexact", "--dry-run"]
    assert seen["env"]["UV_PROJECT_ENVIRONMENT"] == "/venv"
    assert seen["env"]["NO_COLOR"] == "1"
    assert "FORCE_COLOR" not in seen["env"]


def test_a_failed_dry_run_fails_the_check(monkeypatch):
    monkeypatch.setattr(uv_stamp.shutil, "which", lambda name: "/bin/uv")
    monkeypatch.setattr(uv_stamp.subprocess, "run", lambda cmd, **kw: _Ran(2, "error: no lock"))
    why = uv_stamp.uv_would_reinstall("/venv")
    assert why is not None and "rc=2" in why


def test_without_uv_the_check_is_skipped(monkeypatch, capsys):
    monkeypatch.setattr(uv_stamp.shutil, "which", lambda name: None)
    assert uv_stamp.uv_would_reinstall("/venv") is None
    assert "skipped the cross-check" in capsys.readouterr().err


@_CTIME_ONLY
def test_a_directory_ctime_never_becomes_the_key(tmp_path):
    """`../*/src/**/*` also matches directories, whose ctime moves when a file
    inside is created or deleted. uv keys on files only, and so must the
    mirror, or it would write a key uv never computes."""
    pkg = _tree(tmp_path)
    before = uv_stamp.cache_info(pkg)
    time.sleep(0.02)
    scratch = tmp_path / "crates/sts2-core/src/a/scratch.rs"
    scratch.write_text("")
    scratch.unlink()
    after = uv_stamp.cache_info(pkg)
    assert after == before
    assert after[1].is_file()
