"""Post-submission-cutoff analysis at the *per-submission* level.

Each top-N team typically has up to ~2 active submissions; we rate each
submission independently because (a) Kaggle treats them as separate agents on
the leaderboard, and (b) the two subs from the same team can have different
strengths. Kaggle does not expose per-match rating updates (we checked the
SDK: ``competition_list_episodes`` only returns ``{index, submission_id,
team_id, team_name, reward, state}``), so we compute our own Glicko-2 from
the post-cutoff episode outcomes.

What this module produces (``postcutoff_analysis.json``):

* ``submissions``: the set of "active" subs for top-N teams (filtered to
  subs with ≥ ``min_games`` post-cutoff episodes).
* ``ratings_pure`` / ``ratings_anchored``: end-of-window Glicko-2 ratings per
  sub. Pure is from-scratch; anchored uses the team's current LB score as
  prior with tight RD.
* ``trajectory``: per-sub rating over time. We process episodes in
  chronological order in fixed-length **time buckets** (default 2h). At the
  end of each bucket we run one Glicko-2 update per sub using the games it
  played in that bucket. The output is a list of
  ``(sub_id, bucket_end_ts, rating, rd, games_so_far)``.
* ``bootstrap``: resampled rank distribution per sub.
* ``seat_stats``: 4p seating + kingmaker stats. Still grouped per-team for
  intuition ("when team A loses, who wins"); also breaks out per-sub winrate.
* ``rank_evolution``: Kaggle's own LB-score timeline per team (from
  ``leaderboard_*.json`` snapshots, sparse — see refresh_recent / cron).

Usage::

    python -m kaggle_submission.archive_pipeline.postcutoff_analysis \\
        --top-n 30 --bootstrap-reps 200
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve()
# glicko2.py lives alongside this module; make it importable whether the file is
# run as a script (`python analysis/postcutoff_analysis.py`) or as a module
# (`python -m analysis.postcutoff_analysis`).
sys.path.insert(0, str(HERE.parent))
from glicko2 import update_glicko2  # type: ignore  # noqa: E402


# ---- constants ------------------------------------------------------------

DEFAULT_CUTOFF_UTC = datetime(2026, 6, 24, 0, 0, tzinfo=timezone.utc)
PRIOR_RD_PURE = 200.0
PRIOR_RD_ANCHORED = 50.0
PRIOR_VOLATILITY = 0.06
GLICKO_TAU = 0.5

DEFAULT_BOOTSTRAP_REPS = 200
DEFAULT_TRAJECTORY_BUCKET_HOURS = 2.0
DEFAULT_MIN_GAMES = 50  # filter dud / just-deadline subs

# Kaggle-Elo calibration (reverse-engineered from observed per-game rating
# deltas in the orbit-wars leaderboard).
#   - 200-scale: rating diff of 200 → ~76% expected win
#   - K=10 per game per player; for 4p we split into K_pair = K / (N-1)
#   - Formula matches winner-take-all + losers-draw (== _episode_pairwise)
# Note: real Kaggle uses higher K for the first ~tens of games (uncertainty),
# producing ~100-120 swings. Top-30 post-cutoff subs all have thousands of
# pre-cutoff games so they're at the steady-state K and this constant is
# accurate. New / dud subs would be mis-predicted at the start.
KAGGLE_ELO_K = 10.0
KAGGLE_ELO_SCALE = 200.0

# Rating-gap noise at K=10 Elo steady state. Each sub's rating has σ ≈ 20
# around its true skill; comparing two subs, the rating-difference noise is
# σ × √2 ≈ 28 rating points. Tier boundaries use this:
#   - LOCKED: gap > 3σ (~ 60 pts)  — virtually no swap risk
#   - STABLE: gap > 2σ (~ 30 pts)  — rare swap
#   - CONTESTED: gap ≤ 2σ          — frequent rank swaps
ELO_NOISE_RANK_DIFF_SIGMA = 28.0  # rating-diff std under K=10 Elo at convergence
TIER_GAP_STABLE = 30.0  # ~ 2σ
TIER_GAP_LOCKED = 60.0  # ~ 3σ


# ---- helpers --------------------------------------------------------------


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_snapshot_ts(fp: str) -> Optional[datetime]:
    name = Path(fp).stem
    parts = name.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        return datetime.strptime(parts[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ---- data structures -----------------------------------------------------


@dataclass
class TeamInfo:
    team_id: int
    team_name: str
    rank: int
    score: float


@dataclass
class SubInfo:
    submission_id: int
    team_id: int
    team_name: str
    team_rank: int
    team_score: float  # team's leaderboard score (max of sub scores)
    sub_score: float  # this submission's own public_score (Kaggle's per-sub)
    n_games: int  # post-cutoff games for this sub


@dataclass
class Episode:
    episode_id: int
    create_time: datetime
    num_agents: int
    # Per agent: (submission_id, team_id, index, reward).
    rows: list[tuple[int, int, int, int]]


# ---- ingest --------------------------------------------------------------


def _load_top_teams_and_summary(
    archive_dir: Path, top_n: int
) -> tuple[list[TeamInfo], dict]:
    summary = json.loads((archive_dir / "submission_summary.json").read_text())
    lbs = sorted(glob.glob(str(archive_dir / "leaderboard_*.json")))
    if not lbs:
        raise SystemExit(f"no leaderboard snapshots in {archive_dir}")
    latest = json.loads(Path(lbs[-1]).read_text())
    teams: list[TeamInfo] = []
    for rank, row in enumerate(latest["top"][:top_n], start=1):
        tid = int(row["team_id"])
        teams.append(TeamInfo(
            team_id=tid,
            team_name=row.get("team_name") or f"team_{tid}",
            rank=rank,
            score=float(row["score"]) if row.get("score") is not None else float("nan"),
        ))
    return teams, summary


def _scan_episodes(
    archive_dir: Path,
    summary: dict,
    teams: list[TeamInfo],
    cutoff: datetime,
    min_games: int,
) -> tuple[list[Episode], list[SubInfo]]:
    """Walk per-sub episode caches for top-N teams. Returns (episodes,
    sub_list) where sub_list is filtered to subs with >= min_games."""
    ep_dir = archive_dir / "submission_episodes"
    top_team_ids = {t.team_id for t in teams}
    team_by_id = {t.team_id: t for t in teams}

    seen: set[int] = set()
    episodes: list[Episode] = []
    sub_game_count: dict[int, int] = collections.Counter()

    for t in teams:
        tb = summary.get(str(t.team_id), {}) or {}
        for s in tb.get("submissions", []) or []:
            sid = int(s["submission_id"])
            cache = ep_dir / f"{sid}.json"
            if not cache.exists():
                continue
            for ep in json.loads(cache.read_text()):
                eid = int(ep["id"])
                if eid in seen:
                    continue
                seen.add(eid)
                ct = _parse_iso(ep.get("create_time"))
                if ct is None or ct < cutoff:
                    continue
                state = ep.get("state")
                if state and "COMPLETE" not in str(state).upper():
                    continue
                agents = ep.get("agents") or []
                n = len(agents)
                if n not in (2, 4):
                    continue
                rows: list[tuple[int, int, int, int]] = []
                all_top = True
                for a in agents:
                    a_tid = a.get("team_id")
                    a_sid = a.get("submission_id")
                    if a_tid is None or a_sid is None or a_tid not in top_team_ids:
                        all_top = False
                        break
                    rows.append((
                        int(a_sid), int(a_tid),
                        int(a.get("index", 0)),
                        int(a.get("reward") or 0),
                    ))
                if not all_top:
                    continue
                episodes.append(Episode(
                    episode_id=eid,
                    create_time=ct,
                    num_agents=n,
                    rows=rows,
                ))
                for sid_row, _, _, _ in rows:
                    sub_game_count[sid_row] += 1

    episodes.sort(key=lambda e: e.create_time)

    # Resolve each sub_id we've seen back to a team (via episode rows is most
    # robust, since summary may be missing entries for newly-discovered subs).
    sub_to_team: dict[int, int] = {}
    for ep in episodes:
        for sid, tid, _, _ in ep.rows:
            sub_to_team.setdefault(sid, tid)

    # Per-sub public_score from refresh_recent. Falls back to team_score
    # if missing (e.g. before refresh_recent was upgraded to capture it).
    sub_score_map: dict[int, float] = {}
    for team_blob in summary.values():
        for sub_dict in team_blob.get("submissions", []) or []:
            sid_int = int(sub_dict["submission_id"])
            ps = sub_dict.get("public_score")
            if ps is not None:
                try:
                    sub_score_map[sid_int] = float(ps)
                except (TypeError, ValueError):
                    pass

    subs: list[SubInfo] = []
    for sid, ng in sub_game_count.items():
        if ng < min_games:
            continue
        tid = sub_to_team.get(sid)
        if tid is None or tid not in team_by_id:
            continue
        t = team_by_id[tid]
        subs.append(SubInfo(
            submission_id=sid, team_id=tid, team_name=t.team_name,
            team_rank=t.rank, team_score=t.score,
            sub_score=sub_score_map.get(sid, t.score),
            n_games=ng,
        ))
    # Sort: by team rank then by sub_score descending (truer per-sub ranking)
    subs.sort(key=lambda s: (s.team_rank, -s.sub_score))
    return episodes, subs


# ---- Glicko-2 fits --------------------------------------------------------


def _episode_pairwise(
    rows: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, float]]:
    """Per-episode pairwise scores keyed by submission_id."""
    out = []
    n = len(rows)
    for i in range(n):
        si, _, _, ri = rows[i]
        for j in range(i + 1, n):
            sj, _, _, rj = rows[j]
            if si == sj:
                continue
            if ri > rj:
                s = 1.0
            elif ri < rj:
                s = 0.0
            else:
                s = 0.5
            out.append((si, sj, s))
    return out


def _init_rating_map(
    subs: list[SubInfo], *, use_anchor: bool,
) -> tuple[dict[int, float], dict[int, float]]:
    if use_anchor:
        rating = {s.submission_id: (s.sub_score if not math.isnan(s.sub_score) else 1500.0)
                  for s in subs}
        rd = {s.submission_id: PRIOR_RD_ANCHORED for s in subs}
    else:
        rating = {s.submission_id: 1500.0 for s in subs}
        rd = {s.submission_id: PRIOR_RD_PURE for s in subs}
    return rating, rd


def _fit_glicko_oneshot(
    subs: list[SubInfo],
    episodes: list[Episode],
    *,
    use_anchor: bool,
) -> dict[int, dict]:
    """Single-rating-period Glicko-2 fit using all post-cutoff games."""
    rating, rd = _init_rating_map(subs, use_anchor=use_anchor)
    per_sub_opps: dict[int, list[tuple[float, float, float]]] = {
        s.submission_id: [] for s in subs
    }
    for ep in episodes:
        for sa, sb, score_a in _episode_pairwise(ep.rows):
            if sa not in rating or sb not in rating:
                continue
            per_sub_opps[sa].append((rating[sb], rd[sb], score_a))
            per_sub_opps[sb].append((rating[sa], rd[sa], 1.0 - score_a))

    out: dict[int, dict] = {}
    for s in subs:
        opps = per_sub_opps[s.submission_id]
        r0, rd0 = rating[s.submission_id], rd[s.submission_id]
        r1, rd1, v1 = update_glicko2(r0, rd0, PRIOR_VOLATILITY, opps, tau=GLICKO_TAU)
        out[s.submission_id] = {
            "rating": r1, "rd": rd1, "volatility": v1,
            "games": len(opps), "rating_prior": r0, "rd_prior": rd0,
        }
    return out


# ---- online (bucketed) trajectory ----------------------------------------


def _fit_glicko_trajectory(
    subs: list[SubInfo],
    episodes: list[Episode],
    cutoff: datetime,
    *,
    bucket_hours: float,
    use_anchor: bool,
) -> list[dict]:
    """Run Glicko-2 over fixed-length time buckets. At each bucket boundary,
    update every sub that played at least one game in that bucket. Returns a
    flat list of checkpoint rows (one per sub per bucket).

    Opponent (rating, rd) used inside a bucket is the rating *at the start of
    the bucket* — i.e., we use the same prior across the whole bucket. That's
    standard Glicko-2 semantics."""
    if not episodes:
        return []
    rating, rd = _init_rating_map(subs, use_anchor=use_anchor)
    volatility: dict[int, float] = {s.submission_id: PRIOR_VOLATILITY for s in subs}
    games_so_far: dict[int, int] = {s.submission_id: 0 for s in subs}

    bucket_delta = timedelta(hours=bucket_hours)
    start = cutoff
    end = max(e.create_time for e in episodes) + timedelta(minutes=1)

    # Index episodes by bucket
    out: list[dict] = []
    t = start
    ep_iter = iter(episodes)
    pending: Episode | None = None
    while t < end:
        bucket_end = t + bucket_delta
        bucket_eps: list[Episode] = []
        # Drain episodes whose create_time falls in [t, bucket_end)
        while True:
            if pending is None:
                try:
                    pending = next(ep_iter)
                except StopIteration:
                    break
            if pending.create_time >= bucket_end:
                break
            if pending.create_time >= t:
                bucket_eps.append(pending)
            pending = None
        # Collect each sub's opponent list for this bucket
        per_sub_opps: dict[int, list[tuple[float, float, float]]] = {
            sid: [] for sid in rating
        }
        for ep in bucket_eps:
            for sa, sb, score_a in _episode_pairwise(ep.rows):
                if sa not in rating or sb not in rating:
                    continue
                per_sub_opps[sa].append((rating[sb], rd[sb], score_a))
                per_sub_opps[sb].append((rating[sa], rd[sa], 1.0 - score_a))
        # Update each sub that played
        for s in subs:
            sid = s.submission_id
            opps = per_sub_opps[sid]
            if not opps:
                continue
            r1, rd1, v1 = update_glicko2(
                rating[sid], rd[sid], volatility[sid], opps, tau=GLICKO_TAU,
            )
            rating[sid] = r1
            rd[sid] = rd1
            volatility[sid] = v1
            games_so_far[sid] += len(opps)
            out.append({
                "submission_id": sid,
                "team_id": s.team_id,
                "team_name": s.team_name,
                "bucket_end_utc": bucket_end.isoformat(),
                "rating": r1,
                "rd": rd1,
                "games_in_bucket": len(opps),
                "games_so_far": games_so_far[sid],
            })
        t = bucket_end
    return out


# ---- Kaggle-Elo (200-scale, K=10) trajectory ----------------------------


def _kaggle_elo_endstate(
    subs: list[SubInfo],
    episodes: list[Episode],
    *,
    K: float = KAGGLE_ELO_K,
    scale: float = KAGGLE_ELO_SCALE,
) -> dict[int, dict]:
    """Online Elo over all post-cutoff games, in Kaggle's scale (200) and
    K (10). Returns each sub's end-of-window rating + games processed.

    Anchored at each sub's **own** current public_score (from
    competition_team_submissions, not the team's leaderboard row). The
    trajectory function emits per-bucket checkpoints from this same loop;
    this function just returns the final state for the prediction table."""
    rating = {
        s.submission_id: (s.sub_score if not math.isnan(s.sub_score) else 1500.0)
        for s in subs
    }
    prior_by_sid = {s.submission_id: s.sub_score for s in subs}
    games = {s.submission_id: 0 for s in subs}
    for ep in episodes:
        n_pairs_per_player = ep.num_agents - 1
        if n_pairs_per_player <= 0:
            continue
        K_pair = K / n_pairs_per_player
        for sa, sb, score_a in _episode_pairwise(ep.rows):
            if sa not in rating or sb not in rating:
                continue
            E_a = 1.0 / (1.0 + 10 ** ((rating[sb] - rating[sa]) / scale))
            delta = K_pair * (score_a - E_a)
            rating[sa] += delta
            rating[sb] -= delta
            games[sa] += 1
            games[sb] += 1
    return {
        sid: {"rating": rating[sid], "games": games[sid],
              "rating_prior": prior_by_sid.get(sid)}
        for sid in rating
    }


def _kaggle_elo_trajectory(
    subs: list[SubInfo],
    episodes: list[Episode],
    cutoff: datetime,
    *,
    bucket_hours: float,
    K: float = KAGGLE_ELO_K,
    scale: float = KAGGLE_ELO_SCALE,
) -> list[dict]:
    """Per-bucket Kaggle-Elo trajectory: one update per pair per game,
    chronological, ratings anchored at each sub's own public_score."""
    rating = {
        s.submission_id: (s.sub_score if not math.isnan(s.sub_score) else 1500.0)
        for s in subs
    }
    games_so_far = {s.submission_id: 0 for s in subs}
    bucket_played = {s.submission_id: 0 for s in subs}
    name_by_sid = {s.submission_id: s.team_name for s in subs}
    team_by_sid = {s.submission_id: s.team_id for s in subs}

    bucket_delta = timedelta(hours=bucket_hours)
    bucket_end = cutoff + bucket_delta
    out: list[dict] = []

    def _emit(ts: datetime) -> None:
        for sid in rating:
            if bucket_played[sid] == 0:
                continue
            out.append({
                "submission_id": sid,
                "team_id": team_by_sid[sid],
                "team_name": name_by_sid[sid],
                "bucket_end_utc": ts.isoformat(),
                "rating": rating[sid],
                "games_in_bucket": bucket_played[sid],
                "games_so_far": games_so_far[sid],
            })

    for ep in episodes:
        while ep.create_time >= bucket_end:
            _emit(bucket_end)
            for sid in bucket_played:
                bucket_played[sid] = 0
            bucket_end += bucket_delta
        n_pairs_per_player = ep.num_agents - 1
        if n_pairs_per_player <= 0:
            continue
        K_pair = K / n_pairs_per_player
        for sa, sb, score_a in _episode_pairwise(ep.rows):
            if sa not in rating or sb not in rating:
                continue
            E_a = 1.0 / (1.0 + 10 ** ((rating[sb] - rating[sa]) / scale))
            delta = K_pair * (score_a - E_a)
            rating[sa] += delta
            rating[sb] -= delta
            games_so_far[sa] += 1
            games_so_far[sb] += 1
            bucket_played[sa] += 1
            bucket_played[sb] += 1
    # Final bucket
    _emit(bucket_end)
    return out


# ---- bootstrap rank ------------------------------------------------------


def _bootstrap_rank(
    subs: list[SubInfo],
    episodes: list[Episode],
    reps: int,
    seed: int,
) -> tuple[dict[int, dict], dict[int, dict]]:
    """Resample post-cutoff episodes with replacement; re-fit Kaggle-Elo from
    each sub's current public_score anchor; track BOTH sub-rank and
    team-rank per replicate.

    Team rank = rank when teams are sorted by their best sub's rating (this
    is what Kaggle's leaderboard shows). Returns (per_sub, per_team) where
    per_team is keyed by team_id.
    """
    rng = random.Random(seed)
    n_subs = len(subs)
    team_ids = sorted({s.team_id for s in subs})
    n_teams = len(team_ids)
    sub_to_team: dict[int, int] = {s.submission_id: s.team_id for s in subs}

    ranks_by_sub: dict[int, list[int]] = {s.submission_id: [] for s in subs}
    ranks_by_team: dict[int, list[int]] = {tid: [] for tid in team_ids}
    n_ep = len(episodes)
    for _ in range(reps):
        sample = [episodes[rng.randrange(n_ep)] for _ in range(n_ep)]
        fit = _kaggle_elo_endstate(subs, sample)

        # Per-sub rank
        sub_scored = sorted(fit.items(), key=lambda kv: -kv[1]["rating"])
        for rank, (sid, _) in enumerate(sub_scored, start=1):
            ranks_by_sub[sid].append(rank)

        # Per-team rank: each team's score = max of its subs' ratings
        team_max: dict[int, float] = {}
        for sid, info in fit.items():
            tid = sub_to_team[sid]
            r = info["rating"]
            if r > team_max.get(tid, -1e18):
                team_max[tid] = r
        team_scored = sorted(team_max.items(), key=lambda kv: -kv[1])
        for rank, (tid, _) in enumerate(team_scored, start=1):
            ranks_by_team[tid].append(rank)

    def _summarise(ranks: list[int], universe_size: int) -> dict:
        if not ranks:
            return {}
        ranks_sorted = sorted(ranks)
        n = len(ranks_sorted)
        # Bins of interest: Prize=top10, Gold=top19, Silver=top32, Bronze=top49
        # (Kaggle competition medal bands for ≥250 teams).
        return {
            "mean_rank": sum(ranks_sorted) / n,
            "median_rank": ranks_sorted[n // 2],
            "rank_p05": ranks_sorted[int(0.05 * n)],
            "rank_p95": ranks_sorted[int(0.95 * n)],
            "p_top1": sum(1 for r in ranks_sorted if r == 1) / n,
            "p_top3": sum(1 for r in ranks_sorted if r <= 3) / n,
            "p_top5": sum(1 for r in ranks_sorted if r <= 5) / n,
            "p_top6": sum(1 for r in ranks_sorted if r <= 6) / n,
            "p_top10": sum(1 for r in ranks_sorted if r <= 10) / n,
            "p_top15": sum(1 for r in ranks_sorted if r <= 15) / n,
            "p_top19": sum(1 for r in ranks_sorted if r <= 19) / n,
            "p_top30": sum(1 for r in ranks_sorted if r <= 30) / n,
            "rank_hist": [ranks_sorted.count(r)
                          for r in range(1, universe_size + 1)],
        }

    out_sub = {sid: _summarise(rs, n_subs) for sid, rs in ranks_by_sub.items()}
    out_team = {tid: _summarise(rs, n_teams) for tid, rs in ranks_by_team.items()}
    return out_sub, out_team


def _compute_team_kaggle_elo(
    ratings_kaggle: dict[int, dict], subs: list[SubInfo],
) -> dict[int, dict]:
    """Per-team: best Kaggle-Elo rating across the team's subs, plus which
    sub achieved it. This is the rank-determining score Kaggle uses."""
    out: dict[int, dict] = {}
    for s in subs:
        info = ratings_kaggle.get(s.submission_id, {})
        r = info.get("rating")
        if r is None:
            continue
        slot = out.setdefault(s.team_id, {
            "team_id": s.team_id,
            "team_name": s.team_name,
            "team_rank_now": s.team_rank,
            "team_score_now": s.team_score,
            "best_rating": -1e18,
            "best_sub_id": None,
            "subs": [],
        })
        slot["subs"].append({
            "submission_id": s.submission_id,
            "sub_score": s.sub_score,
            "kaggle_elo": r,
            "games": info.get("games", 0),
        })
        if r > slot["best_rating"]:
            slot["best_rating"] = r
            slot["best_sub_id"] = s.submission_id
    return out


def _compute_tiers(
    team_kelo: dict[int, dict],
    bootstrap_team: dict[int, dict],
    *,
    lock_threshold: float = 0.95,
    likely_threshold: float = 0.50,
    contested_threshold: float = 0.10,
) -> list[dict]:
    """Cluster teams into **bootstrap-based lock-in tiers**.

    For each team, we find the smallest K such that p_top_K ≥
    ``lock_threshold`` — that's the tier they're "locked into". Within a
    bucket of {locked top 3, locked top 5, locked top 10}, we sort by
    bootstrap mean rank. Teams that don't lock anywhere fall into "strong
    top 10" (≥ likely_threshold), "contested top 10" (≥ contested), or
    "longshot".

    Returns: list of tier dicts sorted from strongest to weakest.

    Bootstrap-based tiers are tighter than rating-gap tiers because the
    bootstrap captures correlated movements (when A's rating goes up, B's
    relative position drops) and actual signal (not just stationary noise).
    """
    # Tier order: best→worst. Prize zone = top 10 (slots 1-5), Gold Medal
    # zone = top 19 for teams that miss top-10 lock (slots 6-8), then
    # longshot at the tail.
    members_by_tier: dict[str, list[dict]] = {
        "locked_top3": [], "locked_top5": [], "locked_top10": [],
        "strong_top10": [], "contested_top10": [],
        "locked_top19": [], "strong_top19": [], "contested_top19": [],
        "longshot": [],
    }
    tier_meta = [
        ("locked_top3", "🥇 Locked top 3", "P(top 3) ≥ 95%", "#1f7a1f"),
        ("locked_top5", "🥇 Locked top 5", "P(top 5) ≥ 95%", "#3aa53a"),
        ("locked_top10", "💰 Locked top 10 (Prize)", "P(top 10) ≥ 95%", "#6cc46c"),
        ("strong_top10", "💰 Strong Prize contender", "P(top 10) ≥ 50%", "#a8d96a"),
        ("contested_top10", "💰 Contested for Prize", "P(top 10) ≥ 10%", "#d8a93f"),
        ("locked_top19", "🥇 Locked Gold Medal (top 19)", "P(top 19) ≥ 95%, missed Prize", "#e0a040"),
        ("strong_top19", "🥇 Strong Gold contender", "P(top 19) ≥ 50%, missed Prize", "#e8b865"),
        ("contested_top19", "🥇 Contested for Gold", "P(top 19) ≥ 10%, missed Prize", "#d88a1f"),
        ("longshot", "Longshot (no medal)", "P(top 19) < 10%", "#a0a0a0"),
    ]

    for tid, info in team_kelo.items():
        bt = bootstrap_team.get(tid, {})
        p3 = bt.get("p_top3", 0.0) or 0.0
        p5 = bt.get("p_top5", 0.0) or 0.0
        p10 = bt.get("p_top10", 0.0) or 0.0
        p19 = bt.get("p_top19", 0.0) or 0.0
        if p3 >= lock_threshold:
            slot = "locked_top3"
        elif p5 >= lock_threshold:
            slot = "locked_top5"
        elif p10 >= lock_threshold:
            slot = "locked_top10"
        elif p10 >= likely_threshold:
            slot = "strong_top10"
        elif p10 >= contested_threshold:
            slot = "contested_top10"
        elif p19 >= lock_threshold:
            slot = "locked_top19"
        elif p19 >= likely_threshold:
            slot = "strong_top19"
        elif p19 >= contested_threshold:
            slot = "contested_top19"
        else:
            slot = "longshot"
        members_by_tier[slot].append({
            "team_id": tid,
            "team_name": info["team_name"],
            "team_rank_now": info["team_rank_now"],
            "team_score_now": info["team_score_now"],
            "best_rating": info["best_rating"],
            "best_sub_id": info["best_sub_id"],
            "p_top3": p3, "p_top5": p5, "p_top10": p10,
            "p_top15": bt.get("p_top15", 0.0) or 0.0,
            "p_top19": p19,
            "p_top30": bt.get("p_top30", 0.0) or 0.0,
            "bs_mean_rank": bt.get("mean_rank"),
            "bs_p05": bt.get("rank_p05"),
            "bs_p95": bt.get("rank_p95"),
        })

    # Sort each tier's members by bootstrap mean rank
    for slot in members_by_tier:
        members_by_tier[slot].sort(
            key=lambda m: (m.get("bs_mean_rank") or 99, -m["best_rating"]),
        )

    out: list[dict] = []
    for i, (slot, label, criterion, color) in enumerate(tier_meta, start=1):
        members = members_by_tier[slot]
        if not members and slot != "longshot":
            # Always emit the tier even if empty, except longshot (UI clutter)
            pass
        out.append({
            "tier": i,
            "slot": slot,
            "label": label,
            "criterion": criterion,
            "color": color,
            "count": len(members),
            "members": members,
        })
    return out


# ---- match volume & winrates --------------------------------------------


def _empty_volume() -> dict:
    """Mutable accumulator for one sub/team's match volume + W/D/L."""
    return {
        "games_total": 0, "games_2p": 0, "games_4p": 0,
        "wins_total": 0, "draws_total": 0, "losses_total": 0,
        "wins_2p": 0, "draws_2p": 0, "losses_2p": 0,
        "wins_4p": 0, "draws_4p": 0, "losses_4p": 0,
    }


def _finalize_volume(acc: dict, *, extra: dict) -> dict:
    """Add ratios + winrates to a raw accumulator. ``extra`` carries identity
    fields (submission_id / team_name / …). Winrate counts a draw as 0; in 2p
    draws are rare, in 4p only the sole winner scores +1 so winrate is the
    fraction of games this entity finished first."""
    gt, g2, g4 = acc["games_total"], acc["games_2p"], acc["games_4p"]
    return {
        **extra,
        **acc,
        "pct_4p": (g4 / gt) if gt else None,
        "winrate_total": (acc["wins_total"] / gt) if gt else None,
        "winrate_2p": (acc["wins_2p"] / g2) if g2 else None,
        "winrate_4p": (acc["wins_4p"] / g4) if g4 else None,
    }


def _match_volume_stats(
    teams: list[TeamInfo],
    subs: list[SubInfo],
    episodes: list[Episode],
) -> dict:
    """Per-sub and per-team post-cutoff match volume + winrates, split by
    game format (2p / 4p). Reward convention: +1 win, -1 loss, 0 draw.

    Returns ``{"per_sub": {sid: {...}}, "per_team": {tid: {...}}}``. Team rows
    aggregate over the team's subs (the same episode is counted once per
    distinct sub that played it, matching ``n_games`` semantics). The dashboard
    renders team rows with their subs nested underneath.
    """
    sub_lookup = {s.submission_id: s for s in subs}
    sub_acc: dict[int, dict] = {s.submission_id: _empty_volume() for s in subs}
    team_acc: dict[int, dict] = {t.team_id: _empty_volume() for t in teams}

    for ep in episodes:
        fmt = "2p" if ep.num_agents == 2 else "4p"
        # Aggregate to the team once per distinct team in the episode so a team
        # fielding two subs in the same game (it never does in practice) is not
        # double-counted; per-sub is one row per sub.
        teams_in_ep: dict[int, int] = {}  # team_id -> best reward seen
        for sid, tid, _idx, rew in ep.rows:
            if sid in sub_acc:
                a = sub_acc[sid]
                a["games_total"] += 1
                a[f"games_{fmt}"] += 1
                bucket = ("wins" if rew == 1 else
                          "losses" if rew == -1 else "draws")
                a[f"{bucket}_total"] += 1
                a[f"{bucket}_{fmt}"] += 1
            if tid in team_acc:
                teams_in_ep[tid] = max(teams_in_ep.get(tid, -2), rew)
        for tid, rew in teams_in_ep.items():
            a = team_acc[tid]
            a["games_total"] += 1
            a[f"games_{fmt}"] += 1
            bucket = ("wins" if rew == 1 else
                      "losses" if rew == -1 else "draws")
            a[f"{bucket}_total"] += 1
            a[f"{bucket}_{fmt}"] += 1

    team_by_id = {t.team_id: t for t in teams}
    per_sub = {
        str(sid): _finalize_volume(acc, extra={
            "submission_id": sid,
            "team_id": sub_lookup[sid].team_id,
            "team_name": sub_lookup[sid].team_name,
            "sub_score": sub_lookup[sid].sub_score,
        })
        for sid, acc in sub_acc.items()
    }
    # Which subs belong to each team (only subs that cleared min_games).
    team_subs: dict[int, list[int]] = collections.defaultdict(list)
    for s in subs:
        team_subs[s.team_id].append(s.submission_id)
    per_team = {
        str(tid): _finalize_volume(acc, extra={
            "team_id": tid,
            "team_name": team_by_id[tid].team_name,
            "team_rank": team_by_id[tid].rank,
            "team_score": team_by_id[tid].score,
            "sub_ids": sorted(team_subs.get(tid, [])),
        })
        for tid, acc in team_acc.items()
        if acc["games_total"] > 0
    }
    return {"per_sub": per_sub, "per_team": per_team}


# ---- 4p seating / kingmaker ---------------------------------------------


def _seat_stats(
    teams: list[TeamInfo],
    subs: list[SubInfo],
    episodes: list[Episode],
) -> dict:
    """Per-sub seat winrate + per-team direction-conditional + per-team
    kingmaker."""
    sub_seat_games: dict[int, list[int]] = {s.submission_id: [0, 0, 0, 0] for s in subs}
    sub_seat_wins: dict[int, list[int]] = {s.submission_id: [0, 0, 0, 0] for s in subs}
    sub_lookup = {s.submission_id: s for s in subs}

    pair_delta_games: dict[tuple[int, int], list[int]] = collections.defaultdict(
        lambda: [0, 0, 0, 0]
    )
    pair_delta_wins: dict[tuple[int, int], list[int]] = collections.defaultdict(
        lambda: [0, 0, 0, 0]
    )
    king_losses: dict[int, int] = {t.team_id: 0 for t in teams}
    king_neighbour_won: dict[int, list[int]] = {t.team_id: [0, 0, 0] for t in teams}

    for ep in episodes:
        if ep.num_agents != 4:
            continue
        by_idx = {idx: (sid, tid, rew) for (sid, tid, idx, rew) in ep.rows}
        if set(by_idx) != {0, 1, 2, 3}:
            continue
        # Per-sub seat win rate
        for sid, tid, idx, rew in ep.rows:
            if sid in sub_seat_games:
                sub_seat_games[sid][idx] += 1
                if rew == 1:
                    sub_seat_wins[sid][idx] += 1
        # Per (team_A, team_B) direction asymmetry
        for sid_i, tid_i, idx_i, ri in ep.rows:
            if tid_i not in king_losses:
                continue
            for sid_j, tid_j, idx_j, rj in ep.rows:
                if tid_i == tid_j or tid_j not in king_losses:
                    continue
                delta = (idx_j - idx_i) % 4
                if delta == 0:
                    continue
                pair_delta_games[(tid_i, tid_j)][delta] += 1
                if ri == 1:
                    pair_delta_wins[(tid_i, tid_j)][delta] += 1
        # Kingmaker
        for sid, tid, idx, rew in ep.rows:
            if tid not in king_losses or rew != -1:
                continue
            king_losses[tid] += 1
            for d in (1, 2, 3):
                _, _, neigh_rew = by_idx[(idx + d) % 4]
                if neigh_rew == 1:
                    king_neighbour_won[tid][d - 1] += 1

    out: dict = {"per_sub_seat": {}, "pair_delta": {}, "kingmaker": {}}
    for s in subs:
        wins = sub_seat_wins[s.submission_id]
        games = sub_seat_games[s.submission_id]
        wr = [(wins[i] / games[i]) if games[i] else None for i in range(4)]
        out["per_sub_seat"][str(s.submission_id)] = {
            "submission_id": s.submission_id,
            "team_id": s.team_id,
            "team_name": s.team_name,
            "seat_games": games,
            "seat_wins": wins,
            "seat_winrate": wr,
            "total_games": sum(games),
            "overall_winrate": (sum(wins) / sum(games)) if sum(games) else None,
        }
    MIN_PAIR_PER_DELTA = 8
    for (ti, tj), games in pair_delta_games.items():
        if games[1] < MIN_PAIR_PER_DELTA or games[3] < MIN_PAIR_PER_DELTA:
            continue
        wins = pair_delta_wins[(ti, tj)]
        out["pair_delta"][f"{ti}_{tj}"] = {
            "team_a": ti,
            "team_b": tj,
            "delta_games": games,
            "delta_wins": wins,
            "delta_winrate": [
                (wins[d] / games[d]) if games[d] else None for d in range(4)
            ],
        }
    for t in teams:
        losses = king_losses[t.team_id]
        nw = king_neighbour_won[t.team_id]
        out["kingmaker"][str(t.team_id)] = {
            "team_id": t.team_id,
            "team_name": t.team_name,
            "losses": losses,
            "neighbour_won_d1": nw[0],
            "neighbour_won_d2": nw[1],
            "neighbour_won_d3": nw[2],
            "neighbour_wr_d1": (nw[0] / losses) if losses else None,
            "neighbour_wr_d2": (nw[1] / losses) if losses else None,
            "neighbour_wr_d3": (nw[2] / losses) if losses else None,
        }
    return out


# ---- LB-snapshot rank evolution (sparse but free) ------------------------


def _build_rank_evolution(
    archive_dir: Path, team_ids: set[int], cutoff: datetime,
) -> list[dict]:
    rows = []
    for fp in sorted(glob.glob(str(archive_dir / "leaderboard_*.json"))):
        ts = _parse_snapshot_ts(fp)
        if ts is None or ts < cutoff:
            continue
        try:
            d = json.loads(Path(fp).read_text())
        except Exception:
            continue
        for rank, row in enumerate(d.get("top", []), start=1):
            tid = int(row.get("team_id") or -1)
            if tid not in team_ids:
                continue
            rows.append({
                "ts": ts.isoformat(),
                "team_id": tid,
                "team_name": row.get("team_name"),
                "rank": rank,
                "score": row.get("score"),
            })
    rows.sort(key=lambda r: r["ts"])
    return rows


def _build_actual_trajectory(
    archive_dir: Path, teams: list[TeamInfo],
) -> list[dict]:
    """Per-submission **actual** rating over time, straight from the bundled
    leaderboard history (``lb_history.json``) — NOT recomputed from
    episodes. Each top-team row is the live LB score Kaggle reported at that
    instant; we segment a team's series into one trace per distinct
    ``submission_date`` (a new submission ⇒ a new segment). ``seg`` is a
    1-based per-team submission ordinal so the renderer can draw one line per
    submission."""
    p = archive_dir / "lb_history.json"
    if not p.exists():
        return []
    hist = json.loads(p.read_text())
    name_by_tid = {t.team_id: t.team_name for t in teams}
    out: list[dict] = []
    for tid_str, rows in hist.items():
        tid = int(tid_str)
        if tid not in name_by_tid:
            continue
        seg = 0
        prev_sd = object()
        for r in sorted(rows, key=lambda r: r["ts"]):
            sd = r.get("submission_date")
            if sd != prev_sd:
                seg += 1
                prev_sd = sd
            out.append({
                "team_id": tid,
                "team_name": name_by_tid[tid],
                "submission_date": sd,
                "seg": seg,
                "ts_utc": r["ts"],
                "score": r["score"],
            })
    return out


def _build_sub_trajectory(
    archive_dir: Path, subs: list[SubInfo],
) -> list[dict]:
    """Per-submission **actual** rating over time from the append-only
    ``sub_score_history.json`` — each point is a real ``public_score`` Kaggle
    reported for that submission, logged once per refresh. Unlike the recomputed
    Kaggle-Elo, nothing here is inferred from games; it's a genuinely observed
    series that fills in as the refresh loop runs. Restricted to subs still in
    the current top-N."""
    p = archive_dir / "sub_score_history.json"
    if not p.exists():
        return []
    hist = json.loads(p.read_text())
    meta = {s.submission_id: s for s in subs}
    out: list[dict] = []
    for sid_str, pts in hist.items():
        sid = int(sid_str)
        info = meta.get(sid)
        if info is None:
            continue
        for r in sorted(pts, key=lambda r: r["ts"]):
            out.append({
                "submission_id": sid,
                "team_id": info.team_id,
                "team_name": info.team_name,
                "ts_utc": r["ts"],
                "score": r["score"],
            })
    return out


# ---- driver --------------------------------------------------------------


def run(
    archive_dir: Path,
    top_n: int,
    cutoff: datetime,
    bootstrap_reps: int,
    bucket_hours: float,
    min_games: int,
    seed: int,
) -> dict:
    teams, summary = _load_top_teams_and_summary(archive_dir, top_n)
    episodes, subs = _scan_episodes(archive_dir, summary, teams, cutoff, min_games)
    print(f"[postcutoff] teams={len(teams)} subs={len(subs)} episodes={len(episodes)} "
          f"2p={sum(1 for e in episodes if e.num_agents == 2)} "
          f"4p={sum(1 for e in episodes if e.num_agents == 4)}")
    if not episodes or not subs:
        raise SystemExit("no post-cutoff episodes/subs found")

    ratings_pure = _fit_glicko_oneshot(subs, episodes, use_anchor=False)
    ratings_anchored = _fit_glicko_oneshot(subs, episodes, use_anchor=True)
    print(f"[postcutoff] kaggle-elo (K={KAGGLE_ELO_K}, scale={KAGGLE_ELO_SCALE})...")
    ratings_kaggle = _kaggle_elo_endstate(subs, episodes)
    print(f"[postcutoff] trajectory (bucket={bucket_hours}h)...")
    trajectory_glicko = _fit_glicko_trajectory(
        subs, episodes, cutoff, bucket_hours=bucket_hours, use_anchor=True,
    )
    trajectory_kaggle = _kaggle_elo_trajectory(
        subs, episodes, cutoff, bucket_hours=bucket_hours,
    )
    print(f"[postcutoff] bootstrap (reps={bootstrap_reps}, Kaggle-Elo)...")
    bootstrap, bootstrap_team = _bootstrap_rank(
        subs, episodes, bootstrap_reps, seed,
    )
    seat = _seat_stats(teams, subs, episodes)
    match_volume = _match_volume_stats(teams, subs, episodes)
    rank_evo = _build_rank_evolution(archive_dir, {t.team_id for t in teams}, cutoff)
    trajectory_actual = _build_actual_trajectory(archive_dir, teams)
    trajectory_sub_actual = _build_sub_trajectory(archive_dir, subs)
    print(f"[postcutoff] actual trajectory: team={len(trajectory_actual)} "
          f"sub={len(trajectory_sub_actual)} pts")

    # Team-level aggregation + bootstrap-based tier clustering.
    team_kelo = _compute_team_kaggle_elo(ratings_kaggle, subs)
    tiers = _compute_tiers(team_kelo, bootstrap_team)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "top_n": top_n,
        "bootstrap_reps": bootstrap_reps,
        "trajectory_bucket_hours": bucket_hours,
        "min_games_per_sub": min_games,
        "teams": [
            {
                "team_id": t.team_id, "team_name": t.team_name,
                "current_rank": t.rank, "current_score": t.score,
            } for t in teams
        ],
        "submissions": [
            {
                "submission_id": s.submission_id, "team_id": s.team_id,
                "team_name": s.team_name, "team_rank": s.team_rank,
                "team_score": s.team_score, "sub_score": s.sub_score,
                "n_games": s.n_games,
            } for s in subs
        ],
        "episode_counts": {
            "total": len(episodes),
            "two_p": sum(1 for e in episodes if e.num_agents == 2),
            "four_p": sum(1 for e in episodes if e.num_agents == 4),
            "earliest": min(e.create_time for e in episodes).isoformat(),
            "latest": max(e.create_time for e in episodes).isoformat(),
        },
        "kaggle_elo": {
            "K": KAGGLE_ELO_K, "scale": KAGGLE_ELO_SCALE,
            "note": "200-scale Elo, K=10/(N-1) per pair, anchored at "
                    "current public_score. Matches Kaggle's per-game Δ "
                    "within ±0.5 for established subs.",
        },
        "convergence": {
            "rank_diff_sigma": ELO_NOISE_RANK_DIFF_SIGMA,
            "tier_gap_stable": TIER_GAP_STABLE,
            "tier_gap_locked": TIER_GAP_LOCKED,
            "note": "At K=10 Elo steady state, rating diff between two subs "
                    "has std ~28. Gaps above 30 (~2σ) make rank swaps rare; "
                    "above 60 (~3σ) make them virtually impossible.",
        },
        "ratings_pure": {str(sid): info for sid, info in ratings_pure.items()},
        "ratings_anchored": {str(sid): info for sid, info in ratings_anchored.items()},
        "ratings_kaggle": {str(sid): info for sid, info in ratings_kaggle.items()},
        "team_kaggle_elo": {str(tid): info for tid, info in team_kelo.items()},
        "tiers": tiers,
        "trajectory": trajectory_glicko,
        "trajectory_kaggle": trajectory_kaggle,
        "trajectory_actual": trajectory_actual,
        "trajectory_sub_actual": trajectory_sub_actual,
        "bootstrap": {str(sid): info for sid, info in bootstrap.items()},
        "bootstrap_team": {str(tid): info for tid, info in bootstrap_team.items()},
        "seat_stats": seat,
        "match_volume": match_volume,
        "rank_evolution": rank_evo,
    }
    out_path = archive_dir / "postcutoff_analysis.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[postcutoff] wrote {out_path}")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-dir", default="data")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--cutoff-utc", default=DEFAULT_CUTOFF_UTC.isoformat())
    p.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    p.add_argument("--bucket-hours", type=float, default=DEFAULT_TRAJECTORY_BUCKET_HOURS)
    p.add_argument("--min-games", type=int, default=DEFAULT_MIN_GAMES)
    p.add_argument("--seed", type=int, default=20260625)
    args = p.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff_utc.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    run(Path(args.archive_dir).resolve(), args.top_n, cutoff,
        args.bootstrap_reps, args.bucket_hours, args.min_games, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
