"""Replace the absolute paths of the private run in the rows and analysis files (2026-10-26).

The sweep wrote absolute paths of the private host into its outputs: every row's
`wheel.so_path` (the engine `.so` in the private worktree's venv; also in the rows of the two
12-seed replays, replay/<commit>/rows.jsonl), the experiment folder
(`inputs.sweep_dir` in analysis/primary.json, `meta.inputs` in analysis/blind/blind.json) and
the private copy of the 2026-10-11 rows (`inputs.old_runs_csv`, `meta.inputs`). This script
replaces

    <anything>/.venv-h/<16-hex src_hash>/                              ->  <venv>/
    <anything>/bench-archive/2026-10-26-ironclad-release-winrates      ->  <experiment>
    <anything>/sts2-mono/logs/2026-10-11-release-winrates              ->  <private-repo>/logs/2026-10-11-release-winrates

and changes no other byte. Applied once, to the files below; the write-up lists each file's
sha256 before and after. Re-running it is a no-op.

Usage: python redact_paths.py [EXPERIMENT_DIR]   (default: the folder above this script)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FILES = (
    *(f"cells/ironclad_A{a}.jsonl" for a in range(11)),
    "analysis/primary.json",
    "analysis/blind/blind.json",
    "analysis/reconcile.json",
    "replay/b0c3d6d60/rows.jsonl",
    "replay/78e23253d/rows.jsonl",
)
# A path segment run cannot cross a JSON string boundary: no quote, backslash or whitespace.
SUBS = (
    (re.compile(r'/[^"\\\s]*/\.venv-h/[0-9a-f]{16}/'), "<venv>/"),
    (re.compile(r'/[^"\\\s]*/bench-archive/2026-10-26-ironclad-release-winrates'), "<experiment>"),
    (
        re.compile(r'/[^"\\\s]*/sts2-mono/logs/2026-10-11-release-winrates'),
        "<private-repo>/logs/2026-10-11-release-winrates",
    ),
)


def main() -> int:
    exp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    for rel in FILES:
        path = exp / rel
        text = path.read_text(encoding="utf-8")
        counts = []
        for pattern, repl in SUBS:
            text, n = pattern.subn(repl, text)
            counts.append(n)
        path.write_text(text, encoding="utf-8", newline="")
        print(
            f"{rel}: {counts[0]} venv, {counts[1]} experiment, {counts[2]} old-rows path(s) replaced"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
