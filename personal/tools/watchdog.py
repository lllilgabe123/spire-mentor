"""Spire Mentor update watchdog: is the AI still able to read the game?

Checks, all read-only and local:
  1. Game version (release_info.json) against the versions Spire Mentor
     supports -- a game update can break the mod or the AI.
  2. Whether the installed AI engine is still the fixed build (a Workshop
     re-install of Spire Mentor would bring back the original, which cannot
     read multiplayer-only beta cards such as Midnight).
  3. Every "cannot read this" problem the AI has hit: from Spire Mentor's
     watchdog.jsonl (written in game since the watchdog update) and from the
     AI program's own advisor_stdout.log.

Usage:  python watchdog.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

GAME = Path(r"E:\Steam\steamapps\common\Slay the Spire 2")
SUPPORTED = ("v0.107.1", "v0.111.0")
APPDATA = Path(os.environ.get("LOCALAPPDATA", "")) / "SpireMentor"
ENGINE = GAME / "mods" / "SpireMentor" / "SimAdvisor" / "_internal" / "sts2_rs" / "sts2_rs.pyd"
ORIGINAL_ENGINE = GAME / "mods_backup" / "sts2_rs.pyd.original"
# "...[advisor error: $.deck.hand[2]: MIDNIGHT]" -> "$.deck.hand[2]: MIDNIGHT"
ERROR_RE = re.compile(r"advisor error: (.*?)\]?\s*$")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ok = True
    print("Spire Mentor watchdog\n=====================")

    try:
        version = json.loads((GAME / "release_info.json").read_text(encoding="utf-8"))["version"]
        if version in SUPPORTED:
            print(f"OK    Game version {version} is supported.")
        else:
            ok = False
            print(f"WARN  Game version {version} is NOT in Spire Mentor's supported list {SUPPORTED}. "
                  "Expect broken advice until the mod is updated.")
    except (OSError, KeyError, ValueError) as e:
        print(f"WARN  Could not read the game version: {e}")

    if not ENGINE.exists():
        ok = False
        print("WARN  The AI engine file is missing -- is Spire Mentor still installed?")
    elif ORIGINAL_ENGINE.exists() and sha256(ENGINE) == sha256(ORIGINAL_ENGINE):
        ok = False
        print("WARN  The AI engine is the ORIGINAL again (the Midnight fix is gone, e.g. after a "
              "Workshop update). Multiplayer-only beta cards will stop fight advice.")
    else:
        print("OK    The fixed AI engine is installed.")

    # The AI program's own log is the full record; Spire Mentor's
    # watchdog.jsonl only adds what that log may have rotated away.
    problems: Counter[str] = Counter()
    last_seen: dict[str, str] = {}
    out = APPDATA / "advisor_stdout.log"
    for log in (out.with_suffix(".log.1"), out):
        if not log.exists():
            continue
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = ERROR_RE.search(line) if "advice #" in line else None
            if not m:
                continue
            detail = m.group(1).strip()
            what = detail.rsplit(":", 1)[-1].strip() if ":" in detail else detail
            problems[what] += 1
            last_seen[what] = line[:19]
    wd = APPDATA / "watchdog.jsonl"
    if wd.exists():
        for line in wd.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            what = row.get("unreadable") or "?"
            if what not in last_seen:
                problems[what] += 1
            if row.get("time", "") > last_seen.get(what, ""):
                last_seen[what] = row.get("time", "")
    if problems:
        print("\nThings the AI could not read (advice was paused while they were in play):")
        for what, n in problems.most_common():
            print(f"  {what:<30} {n:>5}x   last seen {last_seen.get(what, '?')}")
        print("If one of these appears AFTER the date you installed a fix, tell Claude.")
    else:
        print("OK    The AI has not reported anything it could not read.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
