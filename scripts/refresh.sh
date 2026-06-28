#!/usr/bin/env bash
# Crawl (recent-refresh) → snapshot → analyze → build page → commit → push.
# Designed to run unattended from a single cron entry every 30 minutes.
#
# It mirrors the proven rate-limit budget:
#   * FAST slice (ranks 1..FAST_END) is refreshed EVERY pass        (~30 min)
#   * SLOW slice (ranks FAST_END+1..TOP_N) at most every ~90 min
#   * cold start (no summary yet) does one full 1..TOP_N pass
# Per pass that's roughly: fast ~90 Kaggle calls; slow adds ~210. A 0.4s
# throttle + built-in 429 backoff keep it under Kaggle's burst ceiling.
#
# Post-cutoff there are NO new submissions, so we only re-fetch known active
# subs (refresh_recent) — no BFS discovery is needed.
#
# Needs in the cron environment:
#   * Kaggle creds  (~/.kaggle/kaggle.json, or KAGGLE_USERNAME/KAGGLE_KEY)
#   * non-interactive git push auth (credential store / PAT-in-URL / SSH key)
#
# Overridable via env: PY TOP_N FAST_END SLOW_EVERY_MIN CUTOFF PER_CALL_DELAY
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# cron starts bare — make tools findable and HOME sane.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

PY="${PY:-python3}"
TOP_N="${TOP_N:-100}"
FAST_END="${FAST_END:-30}"
SLOW_EVERY_MIN="${SLOW_EVERY_MIN:-85}"     # ~90 min, allowing for cron jitter
CUTOFF="${CUTOFF:-2026-06-24T00:00:00Z}"
PER_CALL_DELAY="${PER_CALL_DELAY:-0.4}"
EP_EVERY_MIN="${EP_EVERY_MIN:-1440}"       # commit the heavy episode snapshot ≤ once/day
RAW="$REPO/raw"
LOG="$REPO/refresh.log"
STAMP="$RAW/.slow_stamp"
EP_STAMP="$RAW/.episodes_stamp"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Single instance: skip if a previous pass is still running.
exec 9>"$REPO/.refresh.lock"
if ! flock -n 9; then
  echo "$(date -u +%FT%TZ) [skip] another refresh is running" >>"$LOG"
  exit 0
fi

refresh_slice() {  # $1=rank start  $2=rank end
  "$PY" crawl/refresh_recent.py --top-n "$TOP_N" \
    --refresh-rank-start "$1" --refresh-rank-end "$2" \
    --out-dir "$RAW" --per-call-delay "$PER_CALL_DELAY"
}

{
  echo "==== refresh $(date -u +%FT%TZ) (branch $BRANCH) ===="
  mkdir -p "$RAW"

  if [ ! -f "$RAW/submission_summary.json" ]; then
    echo "[cold start] full slice 1..$TOP_N"
    refresh_slice 1 "$TOP_N"
    touch "$STAMP"
  else
    echo "[fast] slice 1..$FAST_END"
    refresh_slice 1 "$FAST_END"
    if [ ! -f "$STAMP" ] || [ -n "$(find "$STAMP" -mmin +"$SLOW_EVERY_MIN" 2>/dev/null)" ]; then
      echo "[slow] slice $((FAST_END + 1))..$TOP_N"
      refresh_slice "$((FAST_END + 1))" "$TOP_N"
      touch "$STAMP"
    else
      echo "[slow] skipped (still fresh)"
    fi
  fi

  # Trim the raw cache into the small committed snapshot, then analyze + build.
  "$PY" scripts/make_snapshot.py --raw "$RAW" --out data --top-n "$TOP_N" \
      --cutoff-utc "$CUTOFF"
  "$PY" -m analysis.postcutoff_analysis --archive-dir data --top-n "$TOP_N" \
      --bootstrap-reps 200 --min-games 50 --cutoff-utc "$CUTOFF"
  "$PY" site/build_static.py --input data/postcutoff_analysis.json \
      --output docs/index.html

  # Commit + push only if something changed.
  #
  # The episode snapshot (data/submission_episodes/) is ~55 MB and changes every
  # pass; committing it each cron run would balloon git history. The analysis JSON
  # + page are tiny, so we commit those every pass (the live dashboard tracks each
  # refresh) and the heavy episodes at most once/day — recent enough for anyone to
  # re-run the pipeline, without per-pass blob churn. See README "Commit cadence".
  git add -A data docs
  if [ -f "$EP_STAMP" ] && [ -z "$(find "$EP_STAMP" -mmin +"$EP_EVERY_MIN" 2>/dev/null)" ]; then
    git reset -q -- data/submission_episodes   # hold episodes back (still on disk, uncommitted)
    echo "[episodes] held — committed ≤ once / $EP_EVERY_MIN min"
  else
    touch "$EP_STAMP"
    echo "[episodes] committing daily snapshot"
  fi
  if git diff --cached --quiet; then
    echo "no changes — nothing to push"
  else
    git -c user.name="orbit-wars-bot" \
        -c user.email="orbit-wars-bot@users.noreply.github.com" \
        commit -q -m "data refresh $(date -u +%FT%TZ)"
    git push -q origin "$BRANCH"
    echo "pushed $(git rev-parse --short HEAD)"
  fi
  echo "==== done $(date -u +%FT%TZ) ===="
} >>"$LOG" 2>&1
