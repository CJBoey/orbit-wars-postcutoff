#!/usr/bin/env python3
"""Trim a full raw crawl into the committed ``data/`` snapshot.

``crawl/fetch_data.py`` writes a large raw archive: full episode records for
the *thousands* of submissions its BFS touches (~1.7 GB). The committed
snapshot only needs the fields the analysis reads, for the top-N teams'
submissions — that trims to ~60 MB and reproduces the dashboard deterministically.

Pipeline position::

    crawl/fetch_data.py  --out-dir raw      # full, gitignored cache
    scripts/make_snapshot.py --raw raw --out data --top-n 100
    analysis/postcutoff_analysis.py --archive-dir data ...

Keeps in ``data/``: trimmed ``submission_episodes/<sid>.json`` for the top-N
teams' subs, ``submission_summary.json`` verbatim, and only the *latest*
``leaderboard_*.json`` (historical snapshots feed only the unused rank-evolution
panel, so we don't accumulate them in git).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

# Reuse the analysis's canonical top-team selection.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import postcutoff_analysis as P  # noqa: E402

KEEP_EP = ("id", "create_time", "state")
KEEP_AG = ("index", "submission_id", "team_id", "reward")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="full crawl dir (input)")
    ap.add_argument("--out", default="data", help="committed snapshot dir (output)")
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    ep_out = out / "submission_episodes"
    ep_out.mkdir(parents=True, exist_ok=True)

    teams, summary = P._load_top_teams_and_summary(raw, args.top_n)
    sub_ids = [int(s["submission_id"]) for t in teams
               for s in (summary.get(str(t.team_id), {}).get("submissions") or [])]

    keep: set[str] = set()
    n_files = n_eps = 0
    for sid in sub_ids:
        f = raw / "submission_episodes" / f"{sid}.json"
        if not f.exists():
            continue
        rows = []
        for ep in json.loads(f.read_text()):
            row = {k: ep.get(k) for k in KEEP_EP}
            row["agents"] = [{k: a.get(k) for k in KEEP_AG}
                             for a in (ep.get("agents") or [])]
            rows.append(row)
            n_eps += 1
        (ep_out / f"{sid}.json").write_text(json.dumps(rows, separators=(",", ":")))
        keep.add(f"{sid}.json")
        n_files += 1

    # Drop episode files for subs that dropped out of the top-N this refresh.
    for f in ep_out.glob("*.json"):
        if f.name not in keep:
            f.unlink()

    # submission_summary.json verbatim (the analysis reads several fields).
    shutil.copy2(raw / "submission_summary.json", out / "submission_summary.json")

    # Keep only the latest leaderboard snapshot.
    for f in out.glob("leaderboard_*.json"):
        f.unlink()
    latest = sorted(glob.glob(str(raw / "leaderboard_*.json")))[-1]
    shutil.copy2(latest, out / os.path.basename(latest))

    print(f"[snapshot] {n_files} subs, {n_eps} episodes, "
          f"LB {os.path.basename(latest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
