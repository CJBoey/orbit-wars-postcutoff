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
| **Per-sub rating trajectory** | Kaggle-Elo over post-cutoff time, game-by-game, anchored at the live LB score. |
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

## Data provenance & privacy

The bundled data is a snapshot of the public Orbit Wars leaderboard and episode
metadata, trimmed to the fields the analysis reads
(`id, create_time, state`, and per agent `index, submission_id, team_id,
reward`). No private submission code, file names, or descriptions are included
or fetched — the crawler is deliberately limited to public episode metadata.

## License

[MIT](LICENSE). This is an independent analysis tool and is not affiliated with
or endorsed by Kaggle.
