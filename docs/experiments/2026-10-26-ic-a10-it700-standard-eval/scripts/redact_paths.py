"""Replace the two absolute paths of the private run in the rows and analysis files (2026-10-26).

The run wrote absolute paths into its outputs: every row's `wheel.so_path` (the engine `.so` in
the private worktree's venv), and the experiment folder in `analysis/primary.json` (`prereg`) and
`analysis/recompute_blind.json` (`experiment`, `inputs`). This script replaces

    <anything>/.venv-h/<16-hex src_hash>/                     ->  <venv>/
    <anything>/bench-archive/2026-10-26-ic-a10-it700-standard-eval  ->  <experiment>

and changes no other byte. Applied once, to the five files below; the write-up lists each file's
sha256 before and after. Re-running it is a no-op.

Usage: python redact_paths.py [EXPERIMENT_DIR]   (default: the folder above this script)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FILES = (
    "runs/D.jsonl",
    "runs/P.jsonl",
    "analysis/primary.json",
    "analysis/recompute_blind.json",
    "analysis/reconcile.json",
)
# A path segment run cannot cross a JSON string boundary: no quote, backslash or whitespace.
SUBS = (
    (re.compile(r'/[^"\\\s]*/\.venv-h/[0-9a-f]{16}/'), "<venv>/"),
    (re.compile(r'/[^"\\\s]*/bench-archive/2026-10-26-ic-a10-it700-standard-eval'), "<experiment>"),
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
        print(f"{rel}: {counts[0]} venv path(s), {counts[1]} experiment path(s) replaced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
