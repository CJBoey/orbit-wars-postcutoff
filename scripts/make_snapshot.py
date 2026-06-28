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
teams' subs, ``submission_summary.json`` verbatim, only the *latest*
``leaderboard_*.json``, a compact ``lb_history.json`` — the post-cutoff
per-team LB-score timeline distilled from every historical snapshot (feeds the
team-rating trajectory) so we don't have to accumulate the raw snapshots in git —
and ``sub_score_history.json``, an append-only per-submission ``public_score``
time series we grow one observation per refresh (feeds the per-sub trajectory).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime, timezone
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
    ap.add_argument("--cutoff-utc", default="2026-06-24T00:00:00Z",
                    help="only bundle leaderboard history at/after this instant "
                         "(feeds the actual-rating trajectory)")
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff_utc.replace("Z", "+00:00"))

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

    # Compact post-cutoff leaderboard history for the actual-rating trajectory.
    # Each snapshot's `top` rows carry the team's *real* live LB score and the
    # `submission_date` of the submission producing it; we keep only top-N teams
    # and drop consecutive duplicate (score, submission_date) readings.
    team_ids = {t.team_id for t in teams}
    hist: dict[str, list[dict]] = {}
    n_pts = 0
    for fp in sorted(glob.glob(str(raw / "leaderboard_*.json"))):
        snap = json.loads(Path(fp).read_text())
        stamp = snap.get("fetched_at_utc")
        if not stamp:
            continue
        ts = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if ts < cutoff:
            continue
        ts_iso = ts.isoformat()
        for r in snap.get("top") or []:
            tid = r.get("team_id")
            if tid not in team_ids:
                continue
            rows = hist.setdefault(str(tid), [])
            row = {"ts": ts_iso, "score": r.get("score"),
                   "submission_date": r.get("submission_date")}
            if rows and rows[-1]["score"] == row["score"] \
                    and rows[-1]["submission_date"] == row["submission_date"]:
                continue
            rows.append(row)
            n_pts += 1
    # NB: kept out of the `leaderboard_*.json` namespace so it isn't mistaken
    # for a raw snapshot by the analysis's snapshot glob.
    (out / "lb_history.json").write_text(
        json.dumps(hist, separators=(",", ":")))

    # Accumulate a real PER-SUBMISSION rating time series. The crawler fetches
    # each sub's own `public_score` every pass but overwrites it; here we append
    # the latest observation to a persistent, committed history keyed by
    # submission_id so an actual per-sub trajectory builds up over time. We key
    # on each sub's `last_seen` and skip when it hasn't advanced (no new
    # observation), which faithfully records flats and changes alike.
    sh_path = out / "sub_score_history.json"
    sub_hist: dict[str, list[dict]] = (
        json.loads(sh_path.read_text()) if sh_path.exists() else {})
    n_new = 0
    for t in teams:
        for s in (summary.get(str(t.team_id), {}).get("submissions") or []):
            sid = str(s.get("submission_id"))
            score, ts = s.get("public_score"), s.get("last_seen")
            if score is None or ts is None:
                continue
            pts = sub_hist.setdefault(sid, [])
            if pts and pts[-1]["ts"] == ts:
                continue  # no new observation since last append
            pts.append({"ts": ts, "score": score, "team_id": t.team_id})
            n_new += 1
    sh_path.write_text(json.dumps(sub_hist, separators=(",", ":")))

    print(f"[snapshot] {n_files} subs, {n_eps} episodes, "
          f"LB {os.path.basename(latest)}, "
          f"team-history {n_pts} pts/{len(hist)} teams, "
          f"sub-history +{n_new} pts ({len(sub_hist)} subs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
