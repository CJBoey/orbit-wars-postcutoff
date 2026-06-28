# Orbit Wars — Post-Cutoff Final-Rank Prediction

A small, fully reproducible pipeline that predicts where teams will **finish** on
the [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)
leaderboard, using only the games played **after the submission cutoff**.

After the deadline, no new submissions are accepted but the leaderboard keeps
churning as agents keep playing. The public score is a noisy, still-moving Elo.
This project recomputes each submission's rating from the post-cutoff games
alone, bootstraps the final-rank distribution, and renders it as a static
dashboard.

**▶ Live dashboard:** https://cjboey.github.io/orbit-wars-postcutoff/
*(update the URL to your fork's GitHub Pages address)*

The page is a single self-contained HTML file (Plotly.js from CDN, vanilla JS,
auto light/dark with a manual toggle) with four tabs:

| Tab | What it shows |
| --- | --- |
| **Final-rank tiers (team)** | Teams bucketed by bootstrap lock-in probability (Locked Top 3 / Prize / Gold / longshot). |
| **Predicted final rank (per sub)** | Each submission's recomputed Kaggle-Elo, current LB delta, and bootstrap rank CI. |
| **Per-sub rating trajectory (actual LB)** | The real Kaggle leaderboard score over post-cutoff time, one line per submission — straight from the LB snapshots, not recomputed. |
| **Match volume & winrates** | Per-team game counts, 2p/4p split and ratio, and winrates — each team row expands to its submissions. |

## Why this is reproducible

Everything the dashboard shows is derived from **public** Kaggle episode
metadata (the same data the Kaggle episode viewer exposes for every game). This
repo bundles a trimmed snapshot of that data, so you can rebuild the exact
dashboard with **no Kaggle account and no third-party packages** — the analysis
and the site builder are pure Python standard library.

```
data/
├── submission_summary.json          # per-team / per-submission rollup
├── leaderboard_<utc>.json           # public leaderboard snapshot (team ranks)
├── submission_episodes/<sub_id>.json# per-sub episode metadata (trimmed)
└── postcutoff_analysis.json         # ← computed analysis (the dashboard's input)
```

## Quickstart

### 1. Rebuild the page from the bundled analysis (seconds, zero deps)

```bash
python site/build_static.py --input data/postcutoff_analysis.json --output docs/index.html
# preview:
python -m http.server -d docs    # → http://localhost:8000
```

### 2. Re-run the analysis from the bundled raw data (zero deps)

This regenerates `data/postcutoff_analysis.json` from the committed episodes —
proving the snapshot end-to-end. It is deterministic (fixed bootstrap seed).

```bash
python -m analysis.postcutoff_analysis \
    --archive-dir data --top-n 100 \
    --bootstrap-reps 200 --min-games 50 \
    --cutoff-utc 2026-06-24T00:00:00Z
python site/build_static.py --input data/postcutoff_analysis.json --output docs/index.html
```

### 3. Crawl fresh data from Kaggle (needs the Kaggle API)

To run the whole thing against the *current* leaderboard:

```bash
pip install -r requirements.txt          # installs the kaggle client
# credentials: put kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME / KAGGLE_KEY
python crawl/fetch_data.py --top-n 100 --out-dir data
python -m analysis.postcutoff_analysis --archive-dir data --top-n 100
python site/build_static.py --input data/postcutoff_analysis.json --output docs/index.html
```

A `Makefile` wraps these: `make page`, `make analyze`, `make crawl`.

## Method, briefly

- **Kaggle-Elo recompute** — each post-cutoff game updates every player's rating
  by `Δ = K·(S − E)` on Kaggle's 200-scale, `K = 10` (split across pairs in 4p,
  winner-take-all + losers-draw), anchored at each submission's current public
  score. This tracks Kaggle's own per-match delta to within ±0.5 for established
  submissions.
- **Bootstrap final-rank distribution** — resample the post-cutoff episodes with
  replacement, re-fit, and track each submission's end-of-window rank. Repeating
  this yields `P(top K)` per submission/team.
- **Team tiers** — Kaggle ranks *teams* (a team's score = the max of its two
  submissions). Teams are bucketed by the smallest `K` for which
  `P(top K) ≥ 95%` (locked), then by `≥ 50%` / `≥ 10%`.
- **Match volume & winrates** — straight counts from the episode rewards
  (`+1` win, `−1` loss, `0` draw), split by 2p vs 4p so you can see how much data
  backs each rating and how a team does in each format (2p winrate baseline 0.5,
  4p baseline 0.25).

See the docstrings in [`analysis/postcutoff_analysis.py`](analysis/postcutoff_analysis.py)
for the full detail.

## Repo layout

```
analysis/   postcutoff_analysis.py   # the analysis (stdlib only)
            glicko2.py               # Glicko-2 (vendored, stdlib only)
crawl/      fetch_data.py            # Kaggle crawler → produces data/ inputs
site/       build_static.py          # analysis JSON → self-contained index.html
data/       bundled snapshot (see above)
docs/       index.html               # prebuilt page (GitHub Pages serves this)
.github/    workflows/pages.yml      # rebuild + deploy to Pages on push
```

## Deploying your own

1. Push this repo to GitHub.
2. **Settings → Pages → Source: GitHub Actions.**
3. The included workflow rebuilds `docs/index.html` from the committed analysis
   JSON and deploys it. The prebuilt `docs/index.html` also works immediately if
   you instead point Pages at the `/docs` folder on `main`.

To refresh with newer games: re-run step 3 (crawl) above, commit the updated
`data/`, and push — the workflow redeploys.

## Automated refresh (cron)

[`scripts/refresh.sh`](scripts/refresh.sh) runs the whole loop unattended:
**refresh active subs → trim snapshot → analyze → rebuild page → commit → push**
(pushing triggers the Pages workflow). It only commits when something changed,
and a lock file skips overlapping runs.

Post-cutoff **no new submissions appear**, so it doesn't BFS-discover — it just
re-fetches each top team's active subs (via
[`crawl/refresh_recent.py`](crawl/refresh_recent.py)) to pick up new games. To
respect Kaggle's burst limit it splits the work by rank, exactly like the
original loop:

| Slice | Ranks | Cadence | ≈ Kaggle calls / pass |
| --- | --- | --- | --- |
| Fast | 1–30 | every pass (~30 min) | ~90 |
| Slow | 31–100 | every ~90 min | +~210 |

(A 0.4s throttle between calls + automatic 429 backoff keep it well under the
ceiling.) It writes into a gitignored `raw/` cache and trims a ~60 MB `data/`
snapshot from it via [`scripts/make_snapshot.py`](scripts/make_snapshot.py), so
commits stay small. Tunable via env: `TOP_N`, `FAST_END`, `SLOW_EVERY_MIN`,
`PER_CALL_DELAY`, `EP_EVERY_MIN`, `PY`.

### Commit cadence (why episodes lag the analysis)

The refresh splits what it commits, on purpose:

| What | Size | Committed |
| --- | --- | --- |
| `postcutoff_analysis.json` + `docs/index.html` + leaderboard/summary | small | **every pass** (the live dashboard tracks each refresh) |
| `submission_episodes/` (the per-match snapshot) | ~55 MB | **at most once/day** (`EP_EVERY_MIN`, default 1440 min) |

The episode snapshot is large and changes every pass, so committing it each cron
run (every ~30 min) would balloon git history by tens of MB per pass. The
dashboard inputs are tiny, so we push those continuously and let the heavy
episodes lag — a once-a-day snapshot is recent enough for anyone to re-run the
full pipeline, without the per-pass blob churn. (Held-back episodes still sit in
the working tree; the next daily window simply commits their latest state.) The
evaluation window is short — ~10 days — so a daily episode snapshot keeps the
whole post-cutoff record without ever needing history surgery. Set
`EP_EVERY_MIN=0` to commit episodes every pass instead.

**Prerequisites in the (cron) environment:**

1. **Kaggle creds** — `~/.kaggle/kaggle.json` (chmod 600) or `KAGGLE_USERNAME`
   / `KAGGLE_KEY`.
2. **Non-interactive git push auth** — a credential store seeded once:
   ```bash
   git config --global credential.helper store
   git push            # enter username + a PAT once; saved to ~/.git-credentials
   ```
   (or bake a PAT into the remote URL, or use a passphrase-less SSH key). cron
   must run as the **same user** so it reads the same `~/.git-credentials`.

**Run it once by hand first** (the cold-start pass fetches all top-100 active
subs and takes a couple of minutes; later passes are fast):

```bash
./scripts/refresh.sh && tail -n 30 refresh.log
```

**Schedule it.** A single entry every 30 min gives fast-every-pass and
slow-every-~90-min automatically. WSL doesn't start cron on its own:

```bash
sudo service cron start                       # start now
# auto-start on WSL launch — add to /etc/wsl.conf:
#   [boot]
#   command = "service cron start"
```

Then `crontab -e` and add:

```cron
*/30 * * * * /path/to/orbit-wars-postcutoff/scripts/refresh.sh
```

Watch progress with `tail -f refresh.log`. To go gentler on the API, raise the
cron interval and/or `SLOW_EVERY_MIN`.

## Data provenance & privacy

The bundled data is a snapshot of the public Orbit Wars leaderboard and episode
metadata, trimmed to the fields the analysis reads
(`id, create_time, state`, and per agent `index, submission_id, team_id,
reward`). No private submission code, file names, or descriptions are included
or fetched — the crawler is deliberately limited to public episode metadata.

## License

[MIT](LICENSE). This is an independent analysis tool and is not affiliated with
or endorsed by Kaggle.
