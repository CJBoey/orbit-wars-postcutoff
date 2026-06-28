#!/usr/bin/env python3
"""Render the post-cutoff prediction into a self-contained static HTML page.

The Streamlit dashboard (``app.py``) needs a running server + local SQLite /
replay files, so it can't be hosted statically. The **post-cutoff prediction**
section, however, is computed entirely from a single artifact
(``postcutoff_analysis.json``). This script trims that artifact to the
prediction-relevant keys, inlines them into one ``index.html`` (Plotly.js from
CDN, vanilla-JS tabs), and writes it out. The result is a single file that can
be opened locally or uploaded to S3 / CloudFront.

Usage::

    python -m dashboard.build_static \
        --input top100_archive/postcutoff_analysis.json \
        --output dist_site/index.html

``publish_dashboard.py`` wraps this (optionally re-running the analysis) and
uploads the output to S3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Keys the static page actually renders. Everything else (glicko trajectories,
# seat_stats, ratings_pure/anchored, rank_evolution) is dropped to keep the
# inlined payload small (~2 MB vs 8.5 MB).
_KEEP_KEYS = (
    "generated_at_utc", "cutoff_utc", "top_n", "bootstrap_reps",
    "trajectory_bucket_hours", "min_games_per_sub",
    "teams", "submissions", "episode_counts", "kaggle_elo",
    "ratings_kaggle", "tiers", "trajectory_actual", "trajectory_sub_actual",
    "bootstrap", "match_volume",
)


def trim_payload(pc: dict) -> dict:
    """Return a minimal payload with only the keys the static page renders."""
    return {k: pc[k] for k in _KEEP_KEYS if k in pc}


# Both actual-rating trajectories can be the bulk of the payload; thin each to
# the top-K traces by latest score so the charts stay readable and the inlined
# file stays small. `key_fn` maps a row to its trace identity (team or sub).
def _thin_traj(pc: dict, traj_key: str, key_fn, top_k: int) -> None:
    traj = pc.get(traj_key, [])
    last: dict = {}  # trace identity -> latest row, by ts
    for r in traj:
        k = key_fn(r)
        cur = last.get(k)
        if cur is None or r["ts_utc"] >= cur["ts_utc"]:
            last[k] = r
    ranked = sorted(last.items(), key=lambda kv: -(kv[1].get("score") or 0))
    keep = {k for k, _ in ranked[:top_k]}
    pc[traj_key] = [r for r in traj if key_fn(r) in keep]


def thin_trajectory(pc: dict, top_k: int) -> None:
    """In-place: thin the team trajectory to the top-K teams, and the per-sub
    trajectory to the top-K submissions, both by latest score."""
    _thin_traj(pc, "trajectory_actual", lambda r: r["team_id"], top_k)
    _thin_traj(pc, "trajectory_sub_actual", lambda r: r["submission_id"], top_k)
    pc["_traj_top_k"] = top_k


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Orbit Wars — Post-cutoff final-rank prediction</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  /* Theme: dark is the default; light follows the OS preference unless the
     user forces a theme via the toggle (sets <html data-theme>). The explicit
     data-theme rules have higher specificity than the media-query :root, so a
     manual choice always wins over the OS preference. */
  :root { --maxw:1640px; --gutter:32px;
           --bg:#0e1117; --panel:#161b22; --fg:#e6edf3; --muted:#8b949e;
           --border:#30363d; --accent:#58a6ff; --hover:#1c2230; }
  @media (prefers-color-scheme: light) {
    :root { --bg:#ffffff; --panel:#f6f8fa; --fg:#1f2328; --muted:#656d76;
             --border:#d0d7de; --accent:#0969da; --hover:#eef1f4; }
  }
  :root[data-theme="light"] {
    --bg:#ffffff; --panel:#f6f8fa; --fg:#1f2328; --muted:#656d76;
    --border:#d0d7de; --accent:#0969da; --hover:#eef1f4; }
  :root[data-theme="dark"] {
    --bg:#0e1117; --panel:#161b22; --fg:#e6edf3; --muted:#8b949e;
    --border:#30363d; --accent:#58a6ff; --hover:#1c2230; }
  .theme-btn { float:right; background:var(--panel); color:var(--fg);
    border:1px solid var(--border); border-radius:6px; padding:6px 12px;
    cursor:pointer; font-size:0.85em; }
  .theme-btn:hover { border-color:var(--accent); }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
          font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
          font-size:14px; line-height:1.5; }
  /* Full-bleed bars; their inner .wrap centres content at --maxw. */
  .wrap { max-width:var(--maxw); margin:0 auto; padding:0 var(--gutter); width:100%; }
  header { padding:20px 0; border-bottom:1px solid var(--border); }
  h1 { font-size:1.5em; margin:0 0 6px; }
  h2 { font-size:1.15em; margin:22px 0 8px; }
  .muted { color:var(--muted); font-size:0.9em; }
  .kpis { display:flex; gap:14px; flex-wrap:wrap; margin:14px 0 4px; }
  .kpi { background:var(--panel); border:1px solid var(--border); border-radius:8px;
          padding:10px 16px; min-width:120px; flex:1 1 auto; }
  .kpi .v { font-size:1.6em; font-weight:600; }
  .kpi .l { color:var(--muted); font-size:0.82em; }
  .navbar { border-bottom:1px solid var(--border); position:sticky; top:0;
            background:var(--bg); z-index:10; }
  nav { display:flex; gap:4px; flex-wrap:wrap; }
  nav button { background:none; border:none; color:var(--muted); padding:12px 16px;
                cursor:pointer; font-size:0.95em; border-bottom:2px solid transparent; }
  nav button:hover { color:var(--fg); }
  nav button.active { color:var(--accent); border-bottom-color:var(--accent); }
  main { padding:8px 0 60px; }
  .tab { display:none; }
  .tab.active { display:block; }
  table { border-collapse:collapse; width:100%; margin:8px 0 18px; font-size:0.88em; }
  th, td { text-align:right; padding:5px 9px; border-bottom:1px solid var(--border);
            white-space:nowrap; }
  th { color:var(--muted); font-weight:600; position:sticky; top:46px; background:var(--bg); }
  td.l, th.l { text-align:left; }
  tr:hover td { background:var(--hover); }
  .tier { border-left:5px solid #888; padding:7px 12px; margin:10px 0 4px;
           border-radius:4px; }
  .tier b { font-size:1.05em; }
  details { background:var(--panel); border:1px solid var(--border);
             border-radius:6px; margin:6px 0; padding:2px 12px; }
  summary { cursor:pointer; padding:8px 0; font-size:0.95em; }
  .pill { display:inline-block; padding:1px 7px; border-radius:10px; font-size:0.8em; }
  .chart { width:100%; height:560px; }
  .note { color:var(--muted); font-size:0.85em; margin:4px 0 14px; }
</style>
</head>
<body>
<header>
  <div class="wrap">
    <button class="theme-btn" id="themeBtn" title="Toggle theme (Auto / Light / Dark)"></button>
    <h1>Orbit Wars — Post-cutoff final-rank prediction</h1>
    <div class="muted" id="meta"></div>
    <div class="kpis" id="kpis"></div>
  </div>
</header>
<div class="navbar"><nav class="wrap" id="nav"></nav></div>
<main class="wrap" id="main"></main>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const PC = JSON.parse(document.getElementById('payload').textContent);
const teamById = {};
(PC.teams||[]).forEach(t => teamById[t.team_id] = t);

// Theme: 'auto' (follow OS), 'light', or 'dark'. Persisted in localStorage.
// Plotly colours are recomputed from the *effective* theme so charts re-theme
// in lock-step with the CSS vars.
const THEMES = ['auto', 'light', 'dark'];
function storedTheme() {
  try { const v = localStorage.getItem('ow-theme'); return THEMES.includes(v) ? v : 'auto'; }
  catch (e) { return 'auto'; }
}
function effectiveTheme() {
  const t = storedTheme();
  if (t !== 'auto') return t;
  return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
    ? 'dark' : 'light';
}
function themeColors() {
  return effectiveTheme() === 'light'
    ? { paper:'#ffffff', grid:'#d0d7de', fg:'#1f2328' }
    : { paper:'#0e1117', grid:'#30363d', fg:'#e6edf3' };
}
let TH = themeColors();

const fmtPct = v => (v==null||isNaN(v)) ? '—' : (v*100).toFixed(1)+'%';
const fmtPct0 = v => (v==null||isNaN(v)) ? '—' : Math.round(v*100)+'%';
const fmt1 = v => (v==null||isNaN(v)) ? '—' : Number(v).toFixed(1);
const fmt2 = v => (v==null||isNaN(v)) ? '—' : Number(v).toFixed(2);
const esc = s => String(s==null?'':s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function table(cols, rows) {
  let h = '<table><thead><tr>';
  cols.forEach(c => h += `<th class="${c.l?'l':''}">${esc(c.t)}</th>`);
  h += '</tr></thead><tbody>';
  rows.forEach(r => {
    h += '<tr>';
    cols.forEach(c => {
      const v = c.f ? c.f(r[c.k], r) : r[c.k];
      h += `<td class="${c.l?'l':''}">${v==null?'—':esc(v)}</td>`;
    });
    h += '</tr>';
  });
  return h + '</tbody></table>';
}

// ---- header -------------------------------------------------------------
function renderHeader() {
  const ec = PC.episode_counts||{};
  document.getElementById('meta').innerHTML =
    `Generated: ${esc(PC.generated_at_utc)} · cutoff: ${esc(PC.cutoff_utc)} ` +
    `· window: ${esc(ec.earliest)} → ${esc(ec.latest)} ` +
    `· bootstrap reps: ${esc(PC.bootstrap_reps)} · top-${esc(PC.top_n)} teams`;
  const kpis = [
    ['Teams', (PC.teams||[]).length],
    ['Submissions', (PC.submissions||[]).length],
    ['Episodes', ec.total||0],
    ['2p games', ec.two_p||0],
    ['4p games', ec.four_p||0],
  ];
  document.getElementById('kpis').innerHTML = kpis.map(
    ([l,v]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`
  ).join('');
}

// ---- tab: tiers ---------------------------------------------------------
function renderTiers(el) {
  el.innerHTML = '<p class="note">Team-level final-rank tiers from the bootstrap '+
    'rank distribution on Kaggle-Elo end-states. Kaggle ranks <b>teams</b> '+
    '(team score = max of its 2 subs). <b>team_score_now</b> = current Kaggle '+
    'LB score; <b>best_rating</b> = max recomputed Kaggle-Elo across the team\\'s '+
    'subs (drives the bootstrap). <b>p_top_K</b> = bootstrap probability of '+
    'finishing in the top K.</p>';
  (PC.tiers||[]).forEach(t => {
    if (!t.count) return;
    const c = t.color||'#888';
    el.innerHTML += `<div class="tier" style="border-left-color:${c};background:${c}18">`+
      `<b style="color:${c}">Tier ${t.tier} — ${esc(t.label)}</b> `+
      `<span class="muted">(${esc(t.criterion)}, ${t.count} team${t.count!=1?'s':''})</span></div>`;
    const cols = [
      {k:'team_rank_now', t:'rank'},
      {k:'team_name', t:'team', l:true},
      {k:'team_score_now', t:'score_now', f:fmt1},
      {k:'best_rating', t:'best_rating', f:fmt1},
      {k:'bs_mean_rank', t:'bs_mean', f:fmt1},
      {k:'bs_p05', t:'p05', f:v=>v==null?'—':Math.round(v)},
      {k:'bs_p95', t:'p95', f:v=>v==null?'—':Math.round(v)},
      {k:'p_top3', t:'P(top3)', f:fmtPct0},
      {k:'p_top10', t:'P(top10)', f:fmtPct0},
      {k:'p_top19', t:'P(top19)', f:fmtPct0},
    ];
    el.innerHTML += table(cols, t.members||[]);
  });
}

// ---- tab: predicted final rank (per sub) --------------------------------
function renderPredicted(el) {
  const kelo = PC.ratings_kaggle||{}, boot = PC.bootstrap||{};
  let rows = (PC.submissions||[]).map(s => {
    const sid = String(s.submission_id);
    const k = kelo[sid]||{}, b = boot[sid]||{};
    const anchor = s.sub_score!=null ? s.sub_score : s.team_score;
    return {
      team_rank: s.team_rank, team_name: s.team_name, submission_id: s.submission_id,
      lb_sub_score: anchor, kaggle_elo: k.rating,
      lb_delta: (k.rating!=null) ? k.rating-anchor : null,
      games: k.games||b.games||0,
      bs_mean_rank: b.mean_rank, bs_p05: b.rank_p05, bs_p95: b.rank_p95,
      p_top10: b.p_top10, p_top15: b.p_top15,
    };
  });
  rows.sort((a,b)=>(a.bs_mean_rank??999)-(b.bs_mean_rank??999));
  rows.forEach((r,i)=>r.pred=i+1);
  el.innerHTML = '<p class="note">End-of-window <b>Kaggle-Elo</b> over post-cutoff '+
    'games only, per submission. The bootstrap resamples episodes, re-fits, and '+
    'tracks each sub\\'s rank to give a final-rank distribution.</p>';
  el.innerHTML += table([
    {k:'pred', t:'pred_rank'},
    {k:'team_rank', t:'team_rank'},
    {k:'team_name', t:'team', l:true},
    {k:'submission_id', t:'sub_id'},
    {k:'lb_sub_score', t:'lb_score', f:fmt1},
    {k:'kaggle_elo', t:'kaggle_elo', f:fmt1},
    {k:'lb_delta', t:'Δ vs lb', f:v=>v==null?'—':(v>=0?'+':'')+v.toFixed(1)},
    {k:'games', t:'games'},
    {k:'bs_mean_rank', t:'bs_mean', f:fmt1},
    {k:'bs_p05', t:'p05', f:v=>v==null?'—':Math.round(v)},
    {k:'bs_p95', t:'p95', f:v=>v==null?'—':Math.round(v)},
    {k:'p_top10', t:'P(top10)', f:fmtPct0},
  ], rows);
}

// ---- tab: rating trajectory ---------------------------------------------
// Shared line-chart helper for both actual-rating tabs. `rows` carry
// {ts_utc, score} plus whatever grouping/label the caller resolves.
function plotTrajectory(divId, traces) {
  Plotly.newPlot(divId, traces, {
    paper_bgcolor:TH.paper, plot_bgcolor:TH.paper, font:{color:TH.fg},
    margin:{t:10,r:10,b:40,l:50}, showlegend:true,
    legend:{font:{size:10}},
    xaxis:{gridcolor:TH.grid, title:'time (UTC)'},
    yaxis:{gridcolor:TH.grid, title:'LB score'},
  }, {responsive:true, displayModeBar:false});
}

function renderTeamTrajectory(el) {
  el.innerHTML = '<p class="note"><b>Actual</b> Kaggle leaderboard score per '+
    '<b>team</b> over post-cutoff time — straight from the LB snapshots, '+
    '<b>not</b> recomputed from episodes. The team score is its leading '+
    'submission, so this is one line per team. Top '+(PC._traj_top_k||'')+
    ' by latest score.</p><div id="teamTrajChart" class="chart"></div>';
  const traj = PC.trajectory_actual||[];
  const byTeam = {};
  traj.forEach(r => { (byTeam[r.team_id] = byTeam[r.team_id]||[]).push(r); });
  const traces = Object.values(byTeam).map(pts => {
    pts.sort((a,b)=>new Date(a.ts_utc)-new Date(b.ts_utc));
    const p0 = pts[0];
    return {
      x: pts.map(p=>p.ts_utc), y: pts.map(p=>p.score),
      mode:'lines', name: (p0.team_name||p0.team_id).slice(0,18),
      hovertemplate:'%{y:.1f}<extra>'+(p0.team_name||p0.team_id)+'</extra>',
    };
  });
  plotTrajectory('teamTrajChart', traces);
}

function renderSubTrajectory(el) {
  const traj = PC.trajectory_sub_actual||[];
  el.innerHTML = '<p class="note"><b>Actual</b> per-<b>submission</b> '+
    'public_score over time — each point is a real score Kaggle reported for '+
    'that submission, logged once per refresh (<b>not</b> recomputed from '+
    'games). Two lines per team. This series starts when logging was switched '+
    'on and <b>fills in as the refresh loop runs</b>, so it is sparse at '+
    'first. Top '+(PC._traj_top_k||'')+' subs by latest score.</p>'+
    '<div id="subTrajChart" class="chart"></div>';
  if (!traj.length) {
    el.innerHTML += '<p class="note">No per-sub history captured yet — check '+
      'back after the next few refreshes.</p>';
    return;
  }
  const bySub = {};
  traj.forEach(r => { (bySub[r.submission_id] = bySub[r.submission_id]||[]).push(r); });
  const traces = Object.entries(bySub).map(([sid, pts]) => {
    pts.sort((a,b)=>new Date(a.ts_utc)-new Date(b.ts_utc));
    const p0 = pts[0];
    return {
      x: pts.map(p=>p.ts_utc), y: pts.map(p=>p.score),
      mode: pts.length>1 ? 'lines+markers' : 'markers',
      name: (p0.team_name||p0.team_id).slice(0,14)+' · '+sid,
      hovertemplate:'%{y:.1f}<extra>'+(p0.team_name||p0.team_id)+'<br>sub '+sid+'</extra>',
    };
  });
  plotTrajectory('subTrajChart', traces);
}

// ---- tab: match volume & winrates ---------------------------------------
function renderVolume(el) {
  const mv = PC.match_volume||{};
  const perTeam = Object.values(mv.per_team||{});
  const perSub = mv.per_sub||{};
  if (!perTeam.length) { el.innerHTML='<p class="note">No match-volume data.</p>'; return; }
  perTeam.sort((a,b)=>(a.team_rank??9999)-(b.team_rank??9999));
  const totAll = perTeam.reduce((s,r)=>s+r.games_total,0);
  const tot2 = perTeam.reduce((s,r)=>s+r.games_2p,0);
  const tot4 = perTeam.reduce((s,r)=>s+r.games_4p,0);
  el.innerHTML = '<p class="note">How many post-cutoff games back each team\\'s '+
    'rating, and how they\\'re doing — split by format. Winrate counts only '+
    'outright wins (reward +1): 2p baseline 0.5, 4p baseline 0.25 (fraction of '+
    'games finished sole first).</p>';
  el.innerHTML += '<div class="kpis">'+
    `<div class="kpi"><div class="v">${totAll}</div><div class="l">sub-games</div></div>`+
    `<div class="kpi"><div class="v">${tot2}</div><div class="l">2p</div></div>`+
    `<div class="kpi"><div class="v">${tot4}</div><div class="l">4p</div></div>`+
    `<div class="kpi"><div class="v">${totAll?Math.round(tot4/totAll*100):0}%</div><div class="l">4p share</div></div>`+
    '</div>';

  el.innerHTML += '<h2>Per-team summary</h2>';
  el.innerHTML += table([
    {k:'team_rank', t:'rank'},
    {k:'team_name', t:'team', l:true},
    {k:'games_total', t:'games'},
    {k:'games_2p', t:'2p'},
    {k:'games_4p', t:'4p'},
    {k:'pct_4p', t:'4p %', f:fmtPct0},
    {k:'winrate_total', t:'WR all', f:fmtPct},
    {k:'winrate_2p', t:'WR 2p', f:fmtPct},
    {k:'winrate_4p', t:'WR 4p', f:fmtPct},
    {k:'wdl', t:'W-D-L', l:true, f:(_,r)=>`${r.wins_total}-${r.draws_total}-${r.losses_total}`},
  ], perTeam);

  el.innerHTML += '<h2>2p vs 4p winrate</h2><div id="volChart" class="chart" style="height:480px"></div>';

  el.innerHTML += '<h2>Per-team breakdown</h2><div id="volDetails"></div>';
  const dEl = el.querySelector('#volDetails');
  perTeam.forEach(r => {
    const d = document.createElement('details');
    d.innerHTML = `<summary>#${r.team_rank} ${esc(r.team_name)} · `+
      `${r.games_total} games (${r.games_2p} 2p / ${r.games_4p} 4p) · `+
      `WR ${fmtPct(r.winrate_total)}</summary>`;
    const subRows = (r.sub_ids||[]).map(sid => perSub[String(sid)]).filter(Boolean)
      .map(sr => ({
        ...sr,
        wdl2:`${sr.wins_2p}-${sr.draws_2p}-${sr.losses_2p}`,
        wdl4:`${sr.wins_4p}-${sr.draws_4p}-${sr.losses_4p}`,
      }));
    d.innerHTML += subRows.length ? table([
      {k:'submission_id', t:'sub_id'},
      {k:'sub_score', t:'lb_score', f:fmt1},
      {k:'games_total', t:'games'},
      {k:'games_2p', t:'2p'},
      {k:'games_4p', t:'4p'},
      {k:'pct_4p', t:'4p %', f:fmtPct0},
      {k:'winrate_total', t:'WR all', f:fmtPct},
      {k:'winrate_2p', t:'WR 2p', f:fmtPct},
      {k:'winrate_4p', t:'WR 4p', f:fmtPct},
      {k:'wdl2', t:'W-D-L 2p', l:true},
      {k:'wdl4', t:'W-D-L 4p', l:true},
    ], subRows) : '<p class="note">No per-sub rows.</p>';
    dEl.appendChild(d);
  });

  const scat = perTeam.filter(r=>r.winrate_2p!=null && r.winrate_4p!=null);
  Plotly.newPlot('volChart', [{
    x: scat.map(r=>r.winrate_2p), y: scat.map(r=>r.winrate_4p),
    text: scat.map(r=>r.team_name), mode:'markers',
    marker:{ size: scat.map(r=>Math.sqrt(r.games_total)), sizemode:'area',
             sizeref:0.6, color:'#58a6ff', opacity:0.7 },
    hovertemplate:'%{text}<br>WR2p=%{x:.1%} WR4p=%{y:.1%}<extra></extra>',
  }], {
    paper_bgcolor:TH.paper, plot_bgcolor:TH.paper, font:{color:TH.fg},
    margin:{t:10,r:10,b:45,l:55},
    xaxis:{gridcolor:TH.grid, title:'WR 2p', tickformat:'.0%'},
    yaxis:{gridcolor:TH.grid, title:'WR 4p', tickformat:'.0%'},
    shapes:[
      {type:'line', y0:0.25, y1:0.25, x0:0, x1:1, line:{color:'gray',dash:'dot'}},
      {type:'line', x0:0.5, x1:0.5, y0:0, y1:1, line:{color:'gray',dash:'dot'}},
    ],
  }, {responsive:true, displayModeBar:false});
}

// ---- tabs wiring --------------------------------------------------------
const TABS = [
  ['Final-rank tiers (team)', renderTiers],
  ['Predicted final rank (per sub)', renderPredicted],
  ['Team rating over time (actual LB)', renderTeamTrajectory],
  ['Per-sub rating over time (actual)', renderSubTrajectory],
  ['Match volume & winrates', renderVolume],
];
let activeIdx = 0;
function showTab(i) {
  activeIdx = i;
  document.querySelectorAll('nav button').forEach((b,j)=>b.classList.toggle('active', i===j));
  document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('active', i===j));
  const el = document.querySelectorAll('.tab')[i];
  if (!el.dataset.rendered) { TABS[i][1](el); el.dataset.rendered='1';
    window.dispatchEvent(new Event('resize')); }
}

// ---- theme toggle -------------------------------------------------------
const themeBtn = document.getElementById('themeBtn');
const THEME_LABEL = { auto:'◑ Auto', light:'☀ Light', dark:'☾ Dark' };
function applyTheme() {
  const t = storedTheme();
  if (t === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', t);
  TH = themeColors();
  themeBtn.textContent = THEME_LABEL[t];
  // Re-render any already-drawn tab so Plotly charts pick up the new colours;
  // tables re-theme for free via CSS vars.
  document.querySelectorAll('.tab').forEach(el => { delete el.dataset.rendered; });
  showTab(activeIdx);
}
themeBtn.onclick = () => {
  const next = THEMES[(THEMES.indexOf(storedTheme()) + 1) % THEMES.length];
  try { localStorage.setItem('ow-theme', next); } catch (e) {}
  applyTheme();
};
// In Auto mode, react live to OS theme changes.
if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (storedTheme() === 'auto') applyTheme();
  });
}

renderHeader();
const nav = document.getElementById('nav'), main = document.getElementById('main');
TABS.forEach(([name], i) => {
  const b = document.createElement('button'); b.textContent = name;
  b.onclick = ()=>showTab(i); nav.appendChild(b);
  const d = document.createElement('div'); d.className='tab'; main.appendChild(d);
});
applyTheme();
showTab(0);
</script>
</body>
</html>
"""


def build_html(pc: dict, traj_top_k: int = 30) -> str:
    pc = trim_payload(pc)
    thin_trajectory(pc, traj_top_k)
    payload = json.dumps(pc, separators=(",", ":"), default=str)
    # Guard against the JSON closing our <script> tag early.
    payload = payload.replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__PAYLOAD__", payload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="path to postcutoff_analysis.json")
    ap.add_argument("--output", required=True,
                    help="path to write index.html")
    ap.add_argument("--traj-top-k", type=int, default=30,
                    help="keep trajectory rows for the top-K subs by rating")
    args = ap.parse_args()

    pc = json.loads(Path(args.input).read_text())
    html = build_html(pc, traj_top_k=args.traj_top_k)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[build_static] wrote {out} ({len(html)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
