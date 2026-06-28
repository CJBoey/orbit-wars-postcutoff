"""Fetch the raw inputs the post-cutoff analysis consumes, from Kaggle.

Snapshots the public Orbit Wars leaderboard, crawls the episode graph to map
``team_id -> submission_id(s)``, and rolls up per-submission win/loss /
opponent stats from cached episode metadata. Writes three things into
``--out-dir`` (default ``data/``), which is exactly what
``analysis/postcutoff_analysis.py`` reads:

* ``leaderboard_<utc>.json`` — the public leaderboard snapshot (team rankings).
* ``submission_episodes/<sub_id>.json`` — per-submission episode metadata
  (id, times, state, per-agent reward / team / submission).
* ``submission_summary.json`` — per-team / per-submission rollup.

Everything is derived from *public* episode metadata returned by
``competition_list_episodes`` (the same data the Kaggle episode viewer shows),
so the snapshot contains nothing private.

Requires Kaggle API credentials (``~/.kaggle/kaggle.json`` or the
``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables). See the README.

Usage::

    python crawl/fetch_data.py --top-n 100 --out-dir data

Re-runs are idempotent: cached episode lists in
``<out_dir>/submission_episodes/<sub_id>.json`` are reused.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kaggle.api.kaggle_api_extended import KaggleApi


# ---- helpers --------------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _to_iso(dt) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return str(dt)


def _read_json(p: Path) -> Any:
    with p.open("r") as f:
        return json.load(f)


def _write_json(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(p)


# ---- output paths ---------------------------------------------------------


@dataclass
class Paths:
    root: Path

    @property
    def episodes_dir(self) -> Path:
        return self.root / "submission_episodes"

    @property
    def submission_index(self) -> Path:
        return self.root / "submission_index.json"

    @property
    def submission_summary(self) -> Path:
        return self.root / "submission_summary.json"

    @property
    def crawl_log(self) -> Path:
        return self.root / "crawl_log.txt"

    def episode_cache(self, sub_id: int) -> Path:
        return self.episodes_dir / f"{sub_id}.json"

    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(exist_ok=True)
        gi = self.root / ".gitignore"
        if not gi.exists():
            gi.write_text("*\n")


# ---- step 1: leaderboard snapshot -----------------------------------------


def snapshot_leaderboard(
    api: KaggleApi, competition: str, top_n: int, paths: Paths, log: list[str]
) -> tuple[list[dict], set[int]]:
    """Fetch the top page (page_size=top_n, capped at 200) and persist."""
    page_size = min(max(top_n, 20), 200)
    rows = api.competition_leaderboard_view(competition, page_size=page_size)
    rows = rows or []
    top = sorted(
        ({
            "team_id": r.team_id,
            "team_name": r.team_name,
            "submission_date": _to_iso(r.submission_date),
            "score": float(r.score) if r.score is not None else None,
        } for r in rows),
        key=lambda x: (x["score"] is None, -(x["score"] or 0.0)),
    )[:top_n]

    payload = {
        "fetched_at_utc": _now_utc(),
        "competition": competition,
        "top_n": top_n,
        "page_size_requested": page_size,
        "raw_count": len(rows),
        "top": top,
    }
    out = paths.root / f"leaderboard_{payload['fetched_at_utc']}.json"
    _write_json(out, payload)
    log.append(f"[leaderboard] saved {out.name} top_n={len(top)} raw={len(rows)}")
    return top, {row["team_id"] for row in top}


# ---- step 2: episode crawl ------------------------------------------------


def _episode_to_dict(ep) -> dict:
    return {
        "id": ep.id,
        "create_time": _to_iso(ep.create_time),
        "end_time": _to_iso(ep.end_time),
        "state": str(ep.state) if ep.state is not None else None,
        "type": str(ep.type) if ep.type is not None else None,
        "agents": [
            {
                "index": a.index,
                "submission_id": a.submission_id,
                "team_id": a.team_id,
                "team_name": a.team_name,
                "reward": a.reward,
                "state": str(a.state) if a.state is not None else None,
            }
            for a in (ep.agents or [])
        ],
    }


def _load_or_fetch_episodes(api: KaggleApi, sub_id: int, paths: Paths) -> list[dict]:
    cache = paths.episode_cache(sub_id)
    if cache.exists():
        return _read_json(cache)
    eps = api.competition_list_episodes(sub_id) or []
    data = [_episode_to_dict(ep) for ep in eps]
    _write_json(cache, data)
    return data


def crawl_episodes(
    api: KaggleApi,
    seed_sub_ids: list[int],
    target_team_ids: set[int],
    paths: Paths,
    log: list[str],
    max_crawl: int,
) -> dict[int, list[int]]:
    """BFS over `competition_list_episodes`. Prefer expanding submissions whose
    team is in target_team_ids; fall back to others only after the prioritized
    queue drains, since lower-ranked teams' episodes can still chain to top
    teams we haven't found yet.

    Returns: {team_id: [submission_id, ...]} (ordered by first-seen).
    """
    team_to_subs: dict[int, list[int]] = collections.defaultdict(list)
    crawled: set[int] = set()

    prio: collections.deque[int] = collections.deque()
    fallback: collections.deque[int] = collections.deque()
    seen_in_queue: set[int] = set()

    def enqueue(sub_id: int, team_id: Optional[int]) -> None:
        if sub_id in crawled or sub_id in seen_in_queue:
            return
        seen_in_queue.add(sub_id)
        if team_id is not None and team_id in target_team_ids:
            prio.append(sub_id)
        else:
            fallback.append(sub_id)

    for sid in seed_sub_ids:
        enqueue(sid, None)

    crawl_count = 0
    cache_hits = 0
    network_fetches = 0

    while crawl_count < max_crawl:
        if prio:
            sid = prio.popleft()
        elif fallback:
            covered = sum(1 for tid in target_team_ids if team_to_subs.get(tid))
            if covered >= len(target_team_ids):
                log.append(f"[crawl] all {len(target_team_ids)} target teams covered, stopping")
                break
            sid = fallback.popleft()
        else:
            break

        if sid in crawled:
            continue
        crawled.add(sid)
        crawl_count += 1

        cache = paths.episode_cache(sid)
        had_cache = cache.exists()
        try:
            episodes = _load_or_fetch_episodes(api, sid, paths)
        except Exception as ex:
            log.append(f"[crawl] sub {sid} list_episodes FAILED: {type(ex).__name__}: {ex}")
            continue
        if had_cache:
            cache_hits += 1
        else:
            network_fetches += 1

        for ep in episodes:
            for agent in ep["agents"]:
                tid = agent["team_id"]
                asid = agent["submission_id"]
                if asid is None:
                    continue
                if asid not in team_to_subs.get(tid, []):
                    team_to_subs.setdefault(tid, []).append(asid)
                if tid in target_team_ids and asid not in crawled:
                    enqueue(asid, tid)

        if crawl_count % 25 == 0:
            covered = sum(1 for tid in target_team_ids if team_to_subs.get(tid))
            log.append(
                f"[crawl] processed={crawl_count} cache_hits={cache_hits} "
                f"net_fetches={network_fetches} target_coverage={covered}/{len(target_team_ids)} "
                f"queues prio={len(prio)} fallback={len(fallback)}"
            )

    covered = sum(1 for tid in target_team_ids if team_to_subs.get(tid))
    log.append(
        f"[crawl] DONE processed={crawl_count} cache_hits={cache_hits} "
        f"net_fetches={network_fetches} target_coverage={covered}/{len(target_team_ids)}"
    )

    _write_json(paths.submission_index, {str(tid): sids for tid, sids in team_to_subs.items()})
    return dict(team_to_subs)


# ---- step 3: per-submission stats -----------------------------------------


@dataclass
class SubStats:
    submission_id: int
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    reward_sum: float = 0.0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    opponents: dict[int, dict] = field(default_factory=dict)

    def add(self, reward: Optional[float], create_time: Optional[str], opponents: list[dict]) -> None:
        self.games += 1
        if reward is None:
            pass
        elif reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1
        self.reward_sum += float(reward) if reward is not None else 0.0
        if create_time:
            if self.first_seen is None or create_time < self.first_seen:
                self.first_seen = create_time
            if self.last_seen is None or create_time > self.last_seen:
                self.last_seen = create_time
        for opp in opponents:
            tid = opp.get("team_id")
            if tid is None:
                continue
            slot = self.opponents.setdefault(tid, {"team_name": opp.get("team_name"), "games": 0, "wins": 0})
            slot["games"] += 1
            if reward is not None and reward > 0:
                slot["wins"] += 1

    def to_dict(self) -> dict:
        opps = {}
        for tid, s in self.opponents.items():
            g = s["games"]
            opps[str(tid)] = {
                "team_name": s["team_name"],
                "games": g,
                "win_rate": (s["wins"] / g) if g else 0.0,
            }
        return {
            "submission_id": self.submission_id,
            "stats": {
                "games": self.games,
                "wins": self.wins,
                "losses": self.losses,
                "draws": self.draws,
                "mean_reward": (self.reward_sum / self.games) if self.games else 0.0,
            },
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "opponents": opps,
        }


def rollup_stats(team_to_subs: dict[int, list[int]], paths: Paths) -> dict[int, dict[int, SubStats]]:
    """For each (team, submission), aggregate stats from cached episodes."""
    by_team: dict[int, dict[int, SubStats]] = {}
    for tid, sids in team_to_subs.items():
        for sid in sids:
            cache = paths.episode_cache(sid)
            if not cache.exists():
                continue
            stats = SubStats(submission_id=sid)
            for ep in _read_json(cache):
                self_agent = next((a for a in ep["agents"] if a["submission_id"] == sid), None)
                if self_agent is None:
                    continue
                opps = [a for a in ep["agents"] if a["submission_id"] != sid]
                stats.add(self_agent.get("reward"), ep.get("create_time"), opps)
            by_team.setdefault(tid, {})[sid] = stats
    return by_team


# ---- summary --------------------------------------------------------------


def build_summary(
    leaderboard: list[dict],
    team_stats: dict[int, dict[int, SubStats]],
) -> dict:
    out: dict[str, dict] = {}
    for row in leaderboard:
        tid = row["team_id"]
        subs_map = team_stats.get(tid, {})
        subs = []
        for sid, st in sorted(
            subs_map.items(), key=lambda kv: (kv[1].last_seen or "", kv[0]), reverse=True
        ):
            subs.append(st.to_dict())
        out[str(tid)] = {
            "team_id": tid,
            "team_name": row["team_name"],
            "leaderboard_score": row["score"],
            "leaderboard_submission_date": row["submission_date"],
            "submissions": subs,
        }
    return out


# ---- programmatic entry (called by crawl_runner) -------------------------


def run_archive(
    competition: str,
    top_n: int,
    out_dir: Path,
    max_crawl: int,
    seed_page_size: int,
    api: Optional[KaggleApi] = None,
    log: Optional[list[str]] = None,
) -> dict:
    """Run the leaderboard snapshot + episode crawl + stats rollup.

    Returns the summary dict; also persists `submission_summary.json`,
    `submission_index.json`, and `leaderboard_<ts>.json`. Suitable for
    calling from another script (e.g. crawl_runner).
    """
    paths = Paths(out_dir.resolve())
    paths.init()

    if api is None:
        api = KaggleApi()
        api.authenticate()
    if log is None:
        log = []

    log.append(f"=== run start {_now_utc()} competition={competition} top_n={top_n} ===")

    leaderboard, target_team_ids = snapshot_leaderboard(api, competition, top_n, paths, log)

    own = api.competition_submissions(competition, page_size=seed_page_size) or []
    seed_ids = [s.ref for s in own]
    log.append(f"[seed] own_submissions={len(seed_ids)}")
    team_to_subs = crawl_episodes(api, seed_ids, target_team_ids, paths, log, max_crawl)

    target_team_to_subs = {tid: sids for tid, sids in team_to_subs.items() if tid in target_team_ids}
    team_stats = rollup_stats(target_team_to_subs, paths)

    summary = build_summary(leaderboard, team_stats)
    _write_json(paths.submission_summary, summary)

    covered = sum(1 for tid in target_team_ids if team_stats.get(tid))
    log.append(
        f"=== archive done | leaderboard={len(leaderboard)} "
        f"target_coverage={covered}/{len(target_team_ids)} "
        f"submissions_with_stats={sum(len(v) for v in team_stats.values())} "
        f"summary={paths.submission_summary} ==="
    )

    paths.crawl_log.write_text("\n".join(log) + "\n")
    return summary


# ---- main -----------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--competition", default="orbit-wars")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--out-dir", default="data")
    p.add_argument("--max-crawl", type=int, default=1500)
    p.add_argument("--seed-page-size", type=int, default=200,
                   help="Page size when fetching own submissions to seed crawl")
    args = p.parse_args()

    t0 = time.time()
    api = KaggleApi()
    api.authenticate()
    auth_method = api.config_values.get(api.CONFIG_NAME_AUTH_METHOD, "<unset>")
    print(f"[auth] method={auth_method}")

    log: list[str] = []
    run_archive(
        competition=args.competition,
        top_n=args.top_n,
        out_dir=Path(args.out_dir),
        max_crawl=args.max_crawl,
        seed_page_size=args.seed_page_size,
        api=api,
        log=log,
    )

    elapsed = time.time() - t0
    print(f"[archive_top100] done in {elapsed:.1f}s")
    print("\n".join(log[-5:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
