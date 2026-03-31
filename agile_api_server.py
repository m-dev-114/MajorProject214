"""
╔══════════════════════════════════════════════════════════════════╗
║  AGILE DATA API SERVER                                           ║
║  FastAPI · REST Endpoints · Live Agile Data                     ║
║  Run: uvicorn agile_api_server:app --reload --port 8000         ║
╚══════════════════════════════════════════════════════════════════╝
"""

from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agile Intelligence API",
    description="REST API serving live agile sprint, team, issue and ML prediction data.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data Generator ────────────────────────────────────────────────────────
ASSIGNEES   = ["Alice","Bob","Carol","David","Eve","Frank","Grace","Henry"]
ISSUE_TYPES = ["Bug","Story","Task","Epic","Sub-task"]
PRIORITIES  = ["High","Medium","Low"]
STATUSES    = ["To Do","In Progress","In Review","Done","Blocked"]
LABELS      = ["feature","bug","tech-debt","hotfix","regression","security","performance"]
SUMMARIES   = [
    "Fix login authentication bug","Add global search feature","Refactor API layer",
    "Update PostgreSQL schema","Deploy CI/CD pipeline","Fix null pointer exception",
    "Implement OAuth 2.0","Optimize database queries","Security audit and fixes",
    "Performance improvements","Migrate to cloud infrastructure","Fix regression in checkout",
    "Add unit test coverage","Code cleanup and linting","Improve error logging",
    "Fix memory leak in worker","Add dark mode support","Implement caching layer",
    "Fix CORS configuration","Add rate limiting","Update dependencies",
    "Implement webhook system","Fix data export bug","Add API versioning",
]

np.random.seed(42)

def gen_sprints(n_sprints=10):
    sprints = []
    for i in range(1, n_sprints+1):
        pln  = float(np.random.randint(40, 100))
        cmp  = float(round(np.clip(pln * np.random.uniform(0.4, 1.05), 0, pln), 1))
        pct  = round(cmp / pln * 100, 1)
        dr   = int(np.random.randint(0, 14))
        hv   = float(np.random.randint(30, 80))
        blk  = int(np.random.randint(0, 6))
        sc   = int(np.random.randint(-8, 12))
        wl   = round(float(np.random.uniform(70, 160)), 1)
        co   = int(np.random.randint(0, 5))
        start= (datetime.now() - timedelta(days=(n_sprints-i)*14)).strftime("%Y-%m-%d")
        end  = (datetime.now() - timedelta(days=(n_sprints-i)*14-14)).strftime("%Y-%m-%d")
        sprints.append({
            "sprint_id":              f"SP-{i:03d}",
            "sprint_number":          i,
            "sprint_name":            f"Sprint {i}",
            "start_date":             start,
            "end_date":               end,
            "state":                  "active" if i==n_sprints else "closed",
            "planned_story_points":   pln,
            "completed_story_points": cmp,
            "percent_done":           pct,
            "days_remaining":         dr if i==n_sprints else 0,
            "historical_velocity":    hv,
            "blocked_stories":        blk,
            "scope_change":           sc,
            "success_label":          int(pct > 60 and blk < 4 and dr > 0),
            "current_workload_pct":   wl,
            "expected_overload":      int(wl > 110),
            "consecutive_overloads":  co,
            "risk_flag":              int(co >= 2 or wl > 130),
            "assignees":              random.sample(ASSIGNEES, k=min(4, len(ASSIGNEES))),
        })
    return sprints

def gen_issues(n=200):
    issues = []
    for i in range(1, n+1):
        itype = np.random.choice(ISSUE_TYPES, p=[0.3,0.3,0.2,0.1,0.1])
        pri   = np.random.choice(PRIORITIES, p=[0.3,0.5,0.2])
        stat  = np.random.choice(STATUSES,   p=[0.15,0.35,0.2,0.25,0.05])
        asgn  = np.random.choice(ASSIGNEES)
        sp    = int(np.random.choice([1,2,3,5,8]))
        eh = round(float(np.clip(np.random.exponential(6), 1, 40)), 1)
        ts    = round(eh * float(np.random.uniform(0.5,1.8)), 1) if stat in ["Done","In Review"] else 0.0
        lbl   = np.random.choice(LABELS)
        smry  = np.random.choice(SUMMARIES)
        sprint= f"SP-{np.random.randint(1,11):03d}"
        cdate = (datetime.now()-timedelta(days=np.random.randint(1,60))).strftime("%Y-%m-%d")
        rdate = (datetime.now()-timedelta(days=np.random.randint(0,10))).strftime("%Y-%m-%d") if stat=="Done" else None
        issues.append({
            "issue_id":               f"AGI-{i:04d}",
            "sprint_id":              sprint,
            "summary":                smry,
            "issue_type":             itype,
            "priority":               pri,
            "status":                 stat,
            "assignee":               asgn,
            "story_points":           sp,
            "original_estimate_hours":eh,
            "time_spent_hours":       ts,
            "resolution_time_hours":  ts if ts > 0 else eh,
            "labels":                 lbl,
            "created_date":           cdate,
            "resolved_date":          rdate,
        })
    return issues

def gen_team(n=200):
    team = []
    for asgn in ASSIGNEES:
        issues_asgn = [i for i in range(n//len(ASSIGNEES))]
        wl  = round(float(np.random.uniform(70, 160)), 1)
        co  = int(np.random.randint(0, 5))
        sp  = int(np.random.randint(20, 80))
        hv  = float(np.random.randint(25, 65))
        hpt = int(np.random.randint(0, 6))
        team.append({
            "name":                     asgn,
            "role":                     np.random.choice(["Senior Dev","Dev","Lead","QA","DevOps"]),
            "current_workload_pct":     wl,
            "expected_overload":        int(wl > 110),
            "total_sp_this_sprint":     sp,
            "historical_avg_sp":        round(hv * 0.85, 1),
            "high_priority_tasks":      hpt,
            "consecutive_overloads":    co,
            "risk_flag":                int(co >= 2 or wl > 130),
            "planned_sp_resource":      float(np.random.randint(15, 60)),
            "current_assigned_sp":      float(sp),
            "remaining_days":           float(np.random.randint(3, 10)),
            "historical_velocity":      hv,
            "burnout_score":            round(min(100, co*15 + max(0, wl-100)*0.5), 1),
            "health_score":             round(max(0, 100 - co*15 - max(0,wl-100)*0.5), 1),
        })
    return team

# ── In-memory dataset ─────────────────────────────────────────────────────
_sprints = gen_sprints(10)
_issues  = gen_issues(200)
_team    = gen_team(200)

# ══════════════════════════════════════════════════════════════════════════
# API LANDING PAGE (HTML)
# ══════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse, tags=["Root"])
def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agile Intelligence API</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #050609; color: #e2e8f0; font-family: 'SF Mono','Fira Code',monospace; min-height: 100vh; }
.nav { background: rgba(5,6,9,0.95); border-bottom: 1px solid #0f172a; padding: 0 3rem; height: 56px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 99; }
.nav-logo { font-size: 0.9rem; font-weight: 800; letter-spacing: 0.12em; color: #f8fafc; display: flex; align-items: center; gap: 10px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: linear-gradient(135deg,#3b82f6,#06b6d4); box-shadow: 0 0 12px #3b82f6; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.nav-right { font-size: 0.68rem; color: #475569; letter-spacing: 0.08em; }
.hero { padding: 5rem 3rem 3rem; max-width: 900px; margin: 0 auto; }
.hero-badge { display: inline-flex; align-items: center; gap: 8px; background: #3b82f615; border: 1px solid #3b82f630; color: #3b82f6; border-radius: 20px; padding: 4px 14px; font-size: 0.68rem; letter-spacing: 0.1em; margin-bottom: 1.5rem; }
.hero-title { font-size: 3rem; font-weight: 900; letter-spacing: -0.04em; line-height: 1; color: #f8fafc; margin-bottom: 0.8rem; }
.hero-title span { background: linear-gradient(135deg,#3b82f6,#06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { font-size: 0.88rem; color: #475569; line-height: 1.7; margin-bottom: 2.5rem; }
.btn-row { display: flex; gap: 12px; flex-wrap: wrap; }
.btn { padding: 10px 24px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-decoration: none; cursor: pointer; }
.btn-primary { background: #3b82f6; color: #fff; }
.btn-outline { background: transparent; color: #3b82f6; border: 1px solid #3b82f6; }
.endpoints { padding: 3rem; max-width: 900px; margin: 0 auto; }
.sec-title { font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: #3b82f6; border-left: 2px solid #3b82f6; padding-left: 10px; margin-bottom: 1.5rem; }
.ep-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.ep-card { background: #0a0d14; border: 1px solid #161b27; border-radius: 10px; padding: 1.2rem 1.4rem; transition: border-color 0.2s; }
.ep-card:hover { border-color: #3b82f630; }
.ep-method { font-size: 0.62rem; font-weight: 800; letter-spacing: 0.1em; margin-bottom: 6px; }
.ep-method.get  { color: #10b981; }
.ep-method.post { color: #f59e0b; }
.ep-path { font-size: 0.82rem; color: #60a5fa; margin-bottom: 5px; font-weight: 700; }
.ep-desc { font-size: 0.74rem; color: #475569; line-height: 1.5; }
.ep-tag  { display: inline-block; background: #0f172a; color: #64748b; border-radius: 3px; padding: 1px 7px; font-size: 0.62rem; margin-top: 6px; }
.stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; padding: 0 3rem 3rem; max-width: 900px; margin: 0 auto; }
.stat-card { background: #0a0d14; border: 1px solid #161b27; border-radius: 8px; padding: 1rem; text-align: center; }
.stat-val { font-size: 1.8rem; font-weight: 900; color: #3b82f6; }
.stat-lbl { font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 3px; }
footer { text-align: center; padding: 2rem; font-size: 0.7rem; color: #1e293b; border-top: 1px solid #0f172a; }
</style>
</head>
<body>
<nav class="nav">
  <div class="nav-logo"><div class="dot"></div>AGILE INTELLIGENCE API</div>
  <div class="nav-right">v2.0.0 &nbsp;·&nbsp; <span style="color:#10b981;">● LIVE</span></div>
</nav>

<div class="hero">
  <div class="hero-badge"><div class="dot" style="width:6px;height:6px;"></div>REST API · FastAPI · JSON</div>
  <div class="hero-title">Agile Data<br><span>Intelligence</span> API</div>
  <div class="hero-sub">
    High-performance REST API serving live agile sprint data, team analytics,<br>
    issue tracking, and ML-ready datasets. Built for the Agile Intelligence Platform.
  </div>
  <div class="btn-row">
    <a class="btn btn-primary" href="/docs">◆ Swagger Docs</a>
    <a class="btn btn-outline" href="/redoc">◈ ReDoc</a>
    <a class="btn btn-outline" href="/api/health">● Health Check</a>
    <a class="btn btn-outline" href="/api/dataset/ml">⚡ ML Dataset</a>
  </div>
</div>

<div class="stats">
  <div class="stat-card"><div class="stat-val">16</div><div class="stat-lbl">Endpoints</div></div>
  <div class="stat-card"><div class="stat-val">200</div><div class="stat-lbl">Issues</div></div>
  <div class="stat-card"><div class="stat-val">10</div><div class="stat-lbl">Sprints</div></div>
  <div class="stat-card"><div class="stat-val">8</div><div class="stat-lbl">Team Members</div></div>
</div>

<div class="endpoints">
  <div class="sec-title">API ENDPOINTS</div>
  <div class="ep-grid">
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/sprints</div>
      <div class="ep-desc">All sprints with velocity, completion %, blocked stories, risk flags</div>
      <div class="ep-tag">Sprint Data</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/sprints/{id}</div>
      <div class="ep-desc">Single sprint detail with full metrics and assignees</div>
      <div class="ep-tag">Sprint Data</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/issues</div>
      <div class="ep-desc">All issues with type, priority, status, assignee, story points</div>
      <div class="ep-tag">Issue Data</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/issues/{id}</div>
      <div class="ep-desc">Single issue detail with full time tracking and resolution data</div>
      <div class="ep-tag">Issue Data</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/team</div>
      <div class="ep-desc">All team members with workload, burnout score, health score</div>
      <div class="ep-tag">Team Data</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/team/{name}</div>
      <div class="ep-desc">Single member detail with full workload and risk metrics</div>
      <div class="ep-tag">Team Data</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/dataset/ml</div>
      <div class="ep-desc">Full ML-ready dataset combining all objectives in one response</div>
      <div class="ep-tag">ML Dataset</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/analytics/summary</div>
      <div class="ep-desc">Project health summary with KPIs, risk counts, velocity trends</div>
      <div class="ep-tag">Analytics</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/analytics/velocity</div>
      <div class="ep-desc">Sprint velocity trend over time with completion rates</div>
      <div class="ep-tag">Analytics</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/analytics/burnout</div>
      <div class="ep-desc">Burnout risk report per team member with historical overload data</div>
      <div class="ep-tag">Analytics</div>
    </div>
    <div class="ep-card">
      <div class="ep-method post">POST</div>
      <div class="ep-path">/api/issues</div>
      <div class="ep-desc">Create new issue with summary, type, priority, story points</div>
      <div class="ep-tag">Write</div>
    </div>
    <div class="ep-card">
      <div class="ep-method post">POST</div>
      <div class="ep-path">/api/issues/{id}/comment</div>
      <div class="ep-desc">Add comment to an existing issue</div>
      <div class="ep-tag">Write</div>
    </div>
    <div class="ep-card">
      <div class="ep-method post">POST</div>
      <div class="ep-path">/api/issues/{id}/transition</div>
      <div class="ep-desc">Transition issue status (To Do → In Progress → Done etc.)</div>
      <div class="ep-tag">Write</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/refresh</div>
      <div class="ep-desc">Regenerate all data with fresh randomised values</div>
      <div class="ep-tag">Admin</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/health</div>
      <div class="ep-desc">API health check — status, version, record counts</div>
      <div class="ep-tag">Admin</div>
    </div>
    <div class="ep-card">
      <div class="ep-method get">GET</div>
      <div class="ep-path">/api/search</div>
      <div class="ep-desc">Search issues by assignee, priority, status, type or keyword</div>
      <div class="ep-tag">Search</div>
    </div>
  </div>
</div>
<footer>Agile Intelligence API v2.0.0 · FastAPI · Built for Agile Intelligence Platform</footer>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════
# READ ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════
@app.get("/api/health", tags=["Admin"])
def health():
    return {"status":"ok","version":"2.0.0","timestamp":datetime.now().isoformat(),
            "records":{"sprints":len(_sprints),"issues":len(_issues),"team":len(_team)}}

@app.get("/api/refresh", tags=["Admin"])
def refresh():
    global _sprints, _issues, _team
    _sprints = gen_sprints(10)
    _issues  = gen_issues(200)
    _team    = gen_team(200)
    return {"status":"refreshed","timestamp":datetime.now().isoformat()}

# ── Sprints ───────────────────────────────────────────────────────────────
@app.get("/api/sprints", tags=["Sprints"])
def get_sprints(state: Optional[str]=None, limit: int=Query(50,le=100)):
    data = _sprints
    if state: data = [s for s in data if s["state"]==state]
    return {"count":len(data[:limit]),"sprints":data[:limit]}

@app.get("/api/sprints/{sprint_id}", tags=["Sprints"])
def get_sprint(sprint_id: str):
    s = next((x for x in _sprints if x["sprint_id"]==sprint_id), None)
    if not s: raise HTTPException(404, f"Sprint {sprint_id} not found")
    issues = [i for i in _issues if i["sprint_id"]==sprint_id]
    return {**s, "issues_count":len(issues), "issues":issues[:10]}

# ── Issues ────────────────────────────────────────────────────────────────
@app.get("/api/issues", tags=["Issues"])
def get_issues(
    assignee:   Optional[str]=None,
    priority:   Optional[str]=None,
    status:     Optional[str]=None,
    issue_type: Optional[str]=None,
    sprint_id:  Optional[str]=None,
    limit: int=Query(100, le=500)
):
    data = _issues
    if assignee:   data = [i for i in data if i["assignee"]==assignee]
    if priority:   data = [i for i in data if i["priority"]==priority]
    if status:     data = [i for i in data if i["status"]==status]
    if issue_type: data = [i for i in data if i["issue_type"]==issue_type]
    if sprint_id:  data = [i for i in data if i["sprint_id"]==sprint_id]
    return {"count":len(data[:limit]),"issues":data[:limit]}

@app.get("/api/issues/{issue_id}", tags=["Issues"])
def get_issue(issue_id: str):
    i = next((x for x in _issues if x["issue_id"]==issue_id), None)
    if not i: raise HTTPException(404, f"Issue {issue_id} not found")
    return i

@app.get("/api/search", tags=["Search"])
def search(q: str="", assignee: Optional[str]=None,
           priority: Optional[str]=None, status: Optional[str]=None,
           limit: int=Query(50,le=200)):
    data = _issues
    if q:        data = [i for i in data if q.lower() in i["summary"].lower()
                                          or q.lower() in i["labels"]]
    if assignee: data = [i for i in data if i["assignee"]==assignee]
    if priority: data = [i for i in data if i["priority"]==priority]
    if status:   data = [i for i in data if i["status"]==status]
    return {"query":q,"count":len(data[:limit]),"results":data[:limit]}

# ── Team ──────────────────────────────────────────────────────────────────
@app.get("/api/team", tags=["Team"])
def get_team():
    return {"count":len(_team),"team":_team}

@app.get("/api/team/{name}", tags=["Team"])
def get_member(name: str):
    m = next((x for x in _team if x["name"].lower()==name.lower()), None)
    if not m: raise HTTPException(404, f"Member {name} not found")
    issues = [i for i in _issues if i["assignee"]==m["name"]]
    return {**m,"issues_assigned":len(issues),"recent_issues":issues[:5]}

# ── ML Dataset ───────────────────────────────────────────────────────────
@app.get("/api/dataset/ml", tags=["ML"])
def get_ml_dataset(limit: int=Query(500, le=1000)):
    """
    Generates a realistic ML dataset with per-row variation.
    Each row is an independent observation — no repeated sprint-level labels.
    Realistic accuracy targets: Sprint ~82-88%, Workload ~78-85%, Burnout ~80-87%
    """
    rng = np.random.default_rng(42)
    n   = min(limit, 1000)

    # ── Per-row numerical features with realistic noise ──────────────────
    pln  = rng.integers(20, 90, n).astype(float)
    hv   = rng.integers(20, 75, n).astype(float)
    dr   = rng.integers(0,  14, n).astype(float)
    blk  = rng.integers(0,   6, n).astype(float)
    sc   = rng.integers(-8, 15, n).astype(float)
    # Completion is correlated with features but noisy
    cmp_ratio = np.clip(
        0.55 + 0.25*(hv/75) - 0.15*(blk/5) - 0.10*(sc.clip(0,15)/15)
        + rng.normal(0, 0.18, n), 0.1, 1.05)
    cmp  = np.round(np.clip(pln * cmp_ratio, 0, pln), 1)
    pct  = np.round(cmp / pln * 100, 1)

    # Workload: correlated with assigned SP vs capacity
    wl_base   = 85 + rng.normal(0, 25, n)
    hist_sp   = rng.integers(18, 65, n).astype(float)
    asgn_sp   = hist_sp * rng.uniform(0.7, 1.5, n)
    rdr       = rng.integers(1, 12, n).astype(float)
    hpt       = rng.integers(0,  6, n).astype(float)
    wl        = np.round(np.clip(wl_base + (asgn_sp - hist_sp) * 0.8, 40, 200), 1)
    # Overload: probabilistic based on workload + high priority tasks
    ol_prob   = 1 / (1 + np.exp(-(wl - 115)/12 + (hpt - 2.5)*0.2))
    ol        = rng.binomial(1, ol_prob).astype(int)

    # Burnout: correlated with overload history + workload
    co        = rng.integers(0, 6, n).astype(int)
    burn_prob = 1 / (1 + np.exp(-(co - 2.0)*0.9 - (wl - 120)/20))
    burn_prob = np.clip(burn_prob + rng.normal(0, 0.08, n), 0.02, 0.98)
    risk      = rng.binomial(1, burn_prob).astype(int)

    # Sprint success: probabilistic (not deterministic)
    succ_prob = 1 / (1 + np.exp(
        -(pct - 65)/12 + blk*0.35 - dr*0.05 + sc.clip(0,15)*0.08))
    succ_prob = np.clip(succ_prob + rng.normal(0, 0.07, n), 0.03, 0.97)
    succ      = rng.binomial(1, succ_prob).astype(int)

    # Issue features
    eh   = np.round(rng.exponential(7, n).clip(1, 50), 1)
    sp   = rng.choice([1, 2, 3, 5, 8], n)
    # TTR correlated with estimate but noisy
    ttr_mult = np.where(
        rng.choice(PRIORITIES, n, p=[0.3,0.5,0.2]) == "High", 
        rng.uniform(0.7, 1.2, n),
        rng.uniform(0.8, 1.8, n))
    ttr  = np.round(eh * ttr_mult + rng.normal(0, 1.5, n), 1).clip(0.5, 80)

    itypes = rng.choice(ISSUE_TYPES, n, p=[0.3,0.3,0.2,0.1,0.1])
    pris   = rng.choice(PRIORITIES,  n, p=[0.3,0.5,0.2])
    asgns  = rng.choice(ASSIGNEES,   n)
    lbls   = rng.choice(LABELS,      n)
    sums   = rng.choice(SUMMARIES,   n)
    sprns  = rng.integers(1, 11, n)

    rows = []
    for i in range(n):
        rows.append({
            # Obj1 — Sprint
            "Planned_Story_Points_Sprint":  float(pln[i]),
            "Completed_Story_Points":       float(cmp[i]),
            "Percent_Done":                 float(pct[i]),
            "Days_Remaining_Sprint":        float(dr[i]),
            "Historical_Velocity":          float(hv[i]),
            "Blocked_Stories":              float(blk[i]),
            "Scope_Change":                 float(sc[i]),
            "Success_Label":                int(succ[i]),
            "Sprint_Number":                int(sprns[i]),
            # Obj2 — Workload
            "Planned_Story_Points_Resource":float(round(hist_sp[i]*0.9,1)),
            "Current_Assigned_SP":          float(round(asgn_sp[i],1)),
            "Historical_Avg_SP":            float(hist_sp[i]),
            "Remaining_Days_Resource":      float(rdr[i]),
            "High_Priority_Tasks_Resource": float(hpt[i]),
            "Current_Workload_Percent":     float(wl[i]),
            "Expected_Overload":            int(ol[i]),
            # Obj3 — TTR
            "Issue_Type":                   str(itypes[i]),
            "Priority":                     str(pris[i]),
            "Original_Estimate_Hours":      float(eh[i]),
            "Story_Points_Issue":           float(sp[i]),
            "Resolution_Time_Hours":        float(ttr[i]),
            # Obj4 — Burnout
            "Total_SP_This_Sprint":         float(pln[i]),
            "Historical_Avg_SP_Burnout":    float(hist_sp[i]*0.85),
            "High_Priority_Tasks_Burnout":  float(hpt[i]),
            "Consecutive_Overloads":        int(co[i]),
            "Risk_Flag":                    int(risk[i]),
            # Obj5 — Allocation
            "Summary":                      str(sums[i]),
            "Labels":                       str(lbls[i]),
            "Original_Estimate_Resource":   float(eh[i]),
            "Story_Points_Resource":        float(sp[i]),
            "Assignee_Resource":            str(asgns[i]),
            "Assignee":                     str(asgns[i]),
            # Meta
            "Issue_ID":                     f"AGI-{i+1:04d}",
            "Status":                       str(rng.choice(STATUSES, p=[0.15,0.35,0.2,0.25,0.05])),
        })

    return {
        "count":      len(rows),
        "columns":    list(rows[0].keys()) if rows else [],
        "source":     "Agile Intelligence API v2.0.0",
        "fetched_at": datetime.now().isoformat(),
        "label_dist": {
            "success_label_pct":  round(float(succ.mean())*100, 1),
            "overload_pct":       round(float(ol.mean())*100, 1),
            "burnout_pct":        round(float(risk.mean())*100, 1),
        },
        "records": rows
    }

# ── Analytics ─────────────────────────────────────────────────────────────
@app.get("/api/analytics/summary", tags=["Analytics"])
def analytics_summary():
    total_issues  = len(_issues)
    done          = sum(1 for i in _issues if i["status"]=="Done")
    blocked       = sum(1 for i in _issues if i["status"]=="Blocked")
    high_pri      = sum(1 for i in _issues if i["priority"]=="High")
    at_risk_sp    = sum(1 for s in _sprints if s["success_label"]==0)
    burnout_risk  = sum(1 for t in _team if t["risk_flag"]==1)
    overloaded    = sum(1 for t in _team if t["expected_overload"]==1)
    avg_vel       = round(sum(s["historical_velocity"] for s in _sprints)/len(_sprints),1)
    avg_cmp       = round(sum(s["percent_done"] for s in _sprints)/len(_sprints),1)
    health = max(0, 100 - at_risk_sp*8 - burnout_risk*6 - blocked*2 - overloaded*5)
    return {
        "health_score":       min(100, health),
        "total_issues":       total_issues,
        "issues_done":        done,
        "issues_blocked":     blocked,
        "high_priority":      high_pri,
        "sprints_at_risk":    at_risk_sp,
        "burnout_risk_count": burnout_risk,
        "overloaded_members": overloaded,
        "avg_velocity":       avg_vel,
        "avg_completion_pct": avg_cmp,
        "team_size":          len(_team),
        "active_sprint":      next((s for s in _sprints if s["state"]=="active"), None),
        "generated_at":       datetime.now().isoformat(),
    }

@app.get("/api/analytics/velocity", tags=["Analytics"])
def analytics_velocity():
    return {
        "sprints": [{
            "sprint_id":   s["sprint_id"],
            "sprint_name": s["sprint_name"],
            "velocity":    s["historical_velocity"],
            "completed":   s["completed_story_points"],
            "planned":     s["planned_story_points"],
            "pct_done":    s["percent_done"],
            "risk":        s["success_label"]==0,
        } for s in sorted(_sprints, key=lambda x: x["sprint_number"])]
    }

@app.get("/api/analytics/burnout", tags=["Analytics"])
def analytics_burnout():
    return {
        "team": sorted([{
            "name":               t["name"],
            "burnout_score":      t["burnout_score"],
            "health_score":       t["health_score"],
            "workload_pct":       t["current_workload_pct"],
            "consecutive_ol":     t["consecutive_overloads"],
            "risk_flag":          t["risk_flag"],
            "high_priority":      t["high_priority_tasks"],
        } for t in _team], key=lambda x: -x["burnout_score"])
    }

# ══════════════════════════════════════════════════════════════════════════
# WRITE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════
class IssueCreate(BaseModel):
    summary:     str
    issue_type:  str = "Task"
    priority:    str = "Medium"
    assignee:    str = "Unassigned"
    story_points:int = 3
    sprint_id:   str = "SP-010"
    labels:      str = "general"
    description: str = ""

class CommentCreate(BaseModel):
    text: str
    author: str = "API User"

class TransitionCreate(BaseModel):
    status: str

_comments: dict = {}

@app.post("/api/issues", tags=["Write"], status_code=201)
def create_issue(body: IssueCreate):
    new_id = f"AGI-{len(_issues)+1:04d}"
    eh     = float(body.story_points * np.random.uniform(1.5, 2.5))
    new_issue = {
        "issue_id":               new_id,
        "sprint_id":              body.sprint_id,
        "summary":                body.summary,
        "issue_type":             body.issue_type,
        "priority":               body.priority,
        "status":                 "To Do",
        "assignee":               body.assignee,
        "story_points":           body.story_points,
        "original_estimate_hours":round(eh,1),
        "time_spent_hours":       0.0,
        "resolution_time_hours":  round(eh,1),
        "labels":                 body.labels,
        "created_date":           datetime.now().strftime("%Y-%m-%d"),
        "resolved_date":          None,
    }
    _issues.append(new_issue)
    return {"created":True,"issue_id":new_id,"issue":new_issue}

@app.post("/api/issues/{issue_id}/comment", tags=["Write"])
def add_comment(issue_id: str, body: CommentCreate):
    iss = next((x for x in _issues if x["issue_id"]==issue_id), None)
    if not iss: raise HTTPException(404, f"Issue {issue_id} not found")
    if issue_id not in _comments: _comments[issue_id] = []
    c = {"id":len(_comments[issue_id])+1,"author":body.author,
         "text":body.text,"created_at":datetime.now().isoformat()}
    _comments[issue_id].append(c)
    return {"added":True,"comment":c,"total_comments":len(_comments[issue_id])}

@app.post("/api/issues/{issue_id}/transition", tags=["Write"])
def transition_issue(issue_id: str, body: TransitionCreate):
    valid = ["To Do","In Progress","In Review","Done","Blocked"]
    if body.status not in valid:
        raise HTTPException(400, f"Invalid status. Must be one of: {valid}")
    iss = next((x for x in _issues if x["issue_id"]==issue_id), None)
    if not iss: raise HTTPException(404, f"Issue {issue_id} not found")
    old_status = iss["status"]
    iss["status"] = body.status
    if body.status == "Done":
        iss["resolved_date"] = datetime.now().strftime("%Y-%m-%d")
        iss["time_spent_hours"] = iss["original_estimate_hours"] * round(float(np.random.uniform(0.7,1.3)),2)
    return {"transitioned":True,"issue_id":issue_id,
            "from":old_status,"to":body.status}
