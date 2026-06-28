"""Refresh the active submissions for the current top-N. Post-submission-
cutoff this is the *only* thing that matters: no new subs can appear, and
every team's set of active subs is fixed for the remaining 2 weeks of the
convergence window.

What it does each pass (~100 Kaggle API calls total — no BFS):

1. Snapshot the top-N leaderboard (1 call, writes ``leaderboard_<ts>.json``).
2. For each top-N team, look up the team's active submissions:
   * Use ``competition_team_submissions(team_id)`` to fetch the team's current
     active subs from Kaggle (1 call per team).
   * Fall back to the team's two most-recently-active subs from the existing
     summary if the team-submissions endpoint returns nothing.
3. For each active sub, **always** re-fetch the episode list (1 call per
   active sub). Post-cutoff there is no static "old data" to preserve — every
   sub is still accumulating games.
4. Re-roll up per-sub stats and write ``submission_summary.json`` for the
   top-N teams.

We deliberately skip the BFS-discovery used by ``archive.run_archive``: post-
cutoff there are no new subs to discover, and the BFS tail (chasing 6-week-old
historical subs) was the only thing tripping Kaggle's 429 burst limit.

Replay downloads are NOT triggered (that endpoint is rate-limited harder; use
``crawl_runner --download`` for those).

Usage::

    python crawl/refresh_recent.py --top-n 100 --out-dir raw
    python crawl/refresh_recent.py --refresh-rank-start 1 --refresh-rank-end 30 --out-dir raw
    python crawl/refresh_recent.py --dry-run --out-dir raw

Normally you run it via ``scripts/refresh.sh`` (fast slice every pass, slow
slice every ~90 min), not directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

# fetch_data.py (the full crawler) lives alongside this file and provides the
# shared helpers (Paths, snapshot_leaderboard, build_summary, rollup_stats, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_data as archive  # type: ignore  # noqa: E402


_RATE_LIMIT_BACKOFF_SEC = 30.0


# Number of active subs per team to refresh. Post-cutoff, Kaggle keeps each
# team's top-2 by score active; 2 covers every active sub. Bump only if you
# see leaderboard rows referencing a sub outside the team's top-2.
ACTIVE_SUBS_PER_TEAM = 2


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _active_subs_for_team(
    api: KaggleApi,
    team_id: int,
    team_blob: dict,
    log: list[str],
) -> list[tuple[int, Optional[float]]]:
    """Return [(sub_id, public_score), ...] for the team's active subs.

    Source of truth is ``competition_team_submissions(team_id)`` — Kaggle's
    own view of "which subs are still being scored". The SDK exposes
    ``sub.id`` and ``sub.public_score`` directly. Retries once with 30s
    backoff on 429 (Kaggle's burst limit); falls back to summary's
    most-recently-active subs (with score=None) on persistent failure."""
    api_subs = []
    for attempt in (1, 2):
        try:
            api_subs = api.competition_team_submissions(int(team_id)) or []
            break
        except Exception as ex:
            if attempt == 1 and _is_rate_limit_error(ex):
                log.append(f"[active] team {team_id} 429, backing off "
                           f"{_RATE_LIMIT_BACKOFF_SEC:.0f}s and retrying")
                time.sleep(_RATE_LIMIT_BACKOFF_SEC)
                continue
            log.append(f"[active] team {team_id} team_submissions FAILED: "
                       f"{type(ex).__name__}: {ex}")
            api_subs = []
            break

    out: list[tuple[int, Optional[float]]] = []
    seen: set[int] = set()
    for api_sub in api_subs:
        sid = getattr(api_sub, "id", None)
        if sid is None:
            continue
        sid = int(sid)
        if sid in seen:
            continue
        seen.add(sid)
        score_raw = getattr(api_sub, "public_score", None)
        try:
            score = float(score_raw) if score_raw not in (None, "") else None
        except (TypeError, ValueError):
            score = None
        out.append((sid, score))

    if not out:
        summary_subs = team_blob.get("submissions", []) or []
        ranked = sorted(
            (s for s in summary_subs if _parse_iso(s.get("last_seen"))),
            key=lambda s: _parse_iso(s["last_seen"]),
            reverse=True,
        )
        out = [(int(s["submission_id"]), None) for s in ranked[:ACTIVE_SUBS_PER_TEAM]]

    return out[:ACTIVE_SUBS_PER_TEAM]


def _is_rate_limit_error(ex: Exception) -> bool:
    """Detect 429 from the SDK's wrapped requests.HTTPError."""
    if isinstance(ex, requests.exceptions.HTTPError):
        resp = getattr(ex, "response", None)
        if resp is not None and resp.status_code == 429:
            return True
    msg = str(ex)
    return "429" in msg and "Too Many Requests" in msg


def _refresh_episode_cache(
    api: KaggleApi, sub_id: int, paths: archive.Paths, log: list[str],
) -> int:
    """Force-refresh one submission's episode cache; return episode count.

    On HTTP 429 (Kaggle's burst-rate ceiling) we sleep
    ``_RATE_LIMIT_BACKOFF_SEC`` and retry once. Other errors fail fast.

    Writes via the same path as :func:`archive._load_or_fetch_episodes` so
    other tooling reads the same cache file."""
    cache = paths.episode_cache(sub_id)
    eps = None
    for attempt in (1, 2):
        try:
            eps = api.competition_list_episodes(sub_id) or []
            break
        except Exception as ex:
            if attempt == 1 and _is_rate_limit_error(ex):
                log.append(f"[refresh] sub {sub_id} 429, backing off "
                           f"{_RATE_LIMIT_BACKOFF_SEC:.0f}s and retrying")
                time.sleep(_RATE_LIMIT_BACKOFF_SEC)
                continue
            log.append(f"[refresh] sub {sub_id} list_episodes FAILED: "
                       f"{type(ex).__name__}: {ex}")
            return -1
    if eps is None:
        return -1
    data = [archive._episode_to_dict(ep) for ep in eps]
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(cache)
    return len(data)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--competition", default="orbit-wars")
    p.add_argument("--top-n", type=int, default=100,
                   help="Total teams to TRACK in the summary. The summary "
                        "always covers top-N regardless of which slice is "
                        "refreshed this iteration.")
    p.add_argument("--refresh-rank-start", type=int, default=1,
                   help="1-based rank: refresh active subs starting here.")
    p.add_argument("--refresh-rank-end", type=int, default=None,
                   help="1-based rank: refresh through this rank (inclusive). "
                        "Defaults to --top-n. Use with --refresh-rank-start to "
                        "process a slice (e.g. fast loop: 1-30; slow loop: 31-100).")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--per-call-delay", type=float, default=0.4,
                   help="Seconds to sleep after each Kaggle API call. Tiny "
                        "throttle to keep us well below Kaggle's burst "
                        "limit even at minimum iteration cadence.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print which subs would be refreshed and stop.")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    paths = archive.Paths(out_dir)
    paths.init()

    api = KaggleApi()
    api.authenticate()
    print(f"[auth] method={api.config_values.get(api.CONFIG_NAME_AUTH_METHOD, '<unset>')}")

    log: list[str] = [f"=== refresh_recent start {datetime.now(timezone.utc).isoformat()} ==="]

    # 1. Fresh leaderboard snapshot.
    leaderboard, _ = archive.snapshot_leaderboard(
        api, args.competition, args.top_n, paths, log,
    )
    print(f"[refresh] leaderboard top-{args.top_n}: {len(leaderboard)} teams")

    # Determine the rank slice to actively refresh.
    refresh_end = args.refresh_rank_end or args.top_n
    refresh_start = max(1, args.refresh_rank_start)
    refresh_end = min(refresh_end, args.top_n)
    refresh_slice = leaderboard[refresh_start - 1: refresh_end]
    print(f"[refresh] refreshing rank slice {refresh_start}..{refresh_end} "
          f"({len(refresh_slice)} teams); summary covers full top-{args.top_n}")

    # 2. Load existing summary (or start fresh).
    summary: dict = {}
    if paths.submission_summary.exists():
        summary = json.loads(paths.submission_summary.read_text())
    by_team = {int(b["team_id"]): b for b in summary.values()}

    # 3. Resolve active subs per team — only for the rank slice.
    team_to_subs_full: dict[int, list[tuple[int, Optional[float]]]] = {}
    for row in refresh_slice:
        tid = int(row["team_id"])
        team_blob = by_team.get(tid, {})
        team_to_subs_full[tid] = _active_subs_for_team(api, tid, team_blob, log)
        if args.per_call_delay > 0:
            time.sleep(args.per_call_delay)

    team_to_subs: dict[int, list[int]] = {
        tid: [sid for sid, _ in subs]
        for tid, subs in team_to_subs_full.items()
    }
    sub_scores: dict[int, Optional[float]] = {
        sid: score for subs in team_to_subs_full.values() for sid, score in subs
    }

    total_subs = sum(len(s) for s in team_to_subs.values())
    print(f"[refresh] active subs to refresh: {total_subs} "
          f"(avg {total_subs / max(1, len(team_to_subs)):.1f}/team)")

    if args.dry_run:
        for tid, subs in team_to_subs_full.items():
            name = by_team.get(tid, {}).get("team_name", f"team_{tid}")
            pairs = ", ".join(f"{sid}({score})" for sid, score in subs)
            print(f"  team {tid} {name}: [{pairs}]")
        return 0

    # 4. Re-fetch each active sub's episode list.
    t0 = time.time()
    fetched = 0
    failed = 0
    total_eps = 0
    for tid, sids in team_to_subs.items():
        for sid in sids:
            n = _refresh_episode_cache(api, sid, paths, log)
            if n < 0:
                failed += 1
            else:
                fetched += 1
                total_eps += n
            if args.per_call_delay > 0:
                time.sleep(args.per_call_delay)
    print(f"[refresh] episode lists: fetched={fetched} failed={failed} "
          f"total_episodes={total_eps} in {time.time() - t0:.1f}s")

    # 5. Roll up + write summary. The summary always covers the full top-N
    # even if we only refreshed a slice this iteration. For non-refreshed
    # teams we reuse the existing summary's sub IDs (from previous iters);
    # missing teams get an empty sub list.
    full_team_to_subs: dict[int, list[int]] = {}
    for row in leaderboard[: args.top_n]:
        tid = int(row["team_id"])
        if tid in team_to_subs:
            full_team_to_subs[tid] = team_to_subs[tid]
        else:
            # Reuse previously-known subs for this team
            prev = (by_team.get(tid) or {}).get("submissions", []) or []
            full_team_to_subs[tid] = [int(s["submission_id"]) for s in prev]

    team_stats = archive.rollup_stats(full_team_to_subs, paths)
    new_summary = archive.build_summary(leaderboard[: args.top_n], team_stats)
    # Inject per-sub public_score from this iter's refresh slice; preserve
    # previously-captured scores for teams outside the slice.
    prev_sub_score: dict[int, float] = {}
    for team_blob in by_team.values():
        for sub_dict in team_blob.get("submissions", []) or []:
            ps = sub_dict.get("public_score")
            if ps is not None:
                try:
                    prev_sub_score[int(sub_dict["submission_id"])] = float(ps)
                except (TypeError, ValueError):
                    pass
    for tid_str, team_blob in new_summary.items():
        for sub_dict in team_blob.get("submissions", []) or []:
            sid = int(sub_dict["submission_id"])
            if sid in sub_scores and sub_scores[sid] is not None:
                sub_dict["public_score"] = sub_scores[sid]
            elif sid in prev_sub_score:
                sub_dict["public_score"] = prev_sub_score[sid]
    archive._write_json(paths.submission_summary, new_summary)
    log.append(f"[refresh] wrote {paths.submission_summary.name} "
               f"({len(new_summary)} teams)")
    paths.crawl_log.write_text("\n".join(log) + "\n")
    print(log[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
