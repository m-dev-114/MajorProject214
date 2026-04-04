"""
╔══════════════════════════════════════════════════════════════╗
║  AGILE INTELLIGENCE  —  DATA SERVER  (Website 1)             ║
║  FastAPI · 16 REST Endpoints · Realistic ML Dataset          ║
║  Run: uvicorn agile_api_server:app --reload --port 8000      ║
║  Visit: http://localhost:8000                                 ║
╚══════════════════════════════════════════════════════════════╝
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np, random
from datetime import datetime, timedelta

app = FastAPI(title="Agile Intelligence API", version="3.0.0",
              docs_url="/docs", redoc_url="/redoc")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ASSIGNEES   = ["Alice","Bob","Carol","David","Eve","Frank","Grace","Henry"]
ISSUE_TYPES = ["Bug","Story","Task","Epic","Sub-task"]
PRIORITIES  = ["High","Medium","Low"]
STATUSES    = ["To Do","In Progress","In Review","Done","Blocked"]
LABELS      = ["feature","bug","tech-debt","hotfix","regression","security","performance"]
SUMMARIES   = [
    "Fix login authentication bug","Add global search feature","Refactor API gateway",
    "Update PostgreSQL schema","Deploy CI/CD pipeline","Fix null pointer exception",
    "Implement OAuth 2.0","Optimize database queries","Security audit and fixes",
    "Performance improvements","Migrate to cloud infrastructure","Fix checkout regression",
    "Add unit test coverage","Code cleanup and linting","Improve error logging",
    "Fix memory leak in worker","Add dark mode support","Implement caching layer",
    "Fix CORS configuration","Add rate limiting middleware","Update dependencies",
    "Implement webhook system","Fix data export bug","Add API versioning",
    "Refactor authentication module","Fix session timeout issue",
]

# ══════════════════════════════════════════════════════════════════════════
# REALISTIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════
def generate_ml_dataset(n=500, seed=42):
    """Per-row probabilistic generation — no sprint-level label leakage."""
    rng = np.random.default_rng(seed)
    pln = rng.integers(20, 90, n).astype(float)
    hv  = rng.integers(20, 75, n).astype(float)
    dr  = rng.integers(0,  14, n).astype(float)
    blk = rng.integers(0,   6, n).astype(float)
    sc  = rng.integers(-8, 15, n).astype(float)
    cmp = np.round(np.clip(pln*(0.55+0.25*(hv/75)-0.15*(blk/5)
          -0.10*(sc.clip(0,15)/15)+rng.normal(0,0.18,n)),0,pln),1)
    pct = np.round(cmp/pln*100,1)
    hist_sp = rng.integers(18,65,n).astype(float)
    asgn_sp = hist_sp*rng.uniform(0.7,1.5,n)
    rdr     = rng.integers(1,12,n).astype(float)
    hpt     = rng.integers(0, 6,n).astype(float)
    wl      = np.round(np.clip(85+rng.normal(0,25,n)+(asgn_sp-hist_sp)*0.8,40,200),1)
    ol      = rng.binomial(1,np.clip(1/(1+np.exp(-(wl-115)/12+(hpt-2.5)*0.2))
              +rng.normal(0,0.05,n),0.02,0.98)).astype(int)
    co      = rng.integers(0,6,n).astype(int)
    risk    = rng.binomial(1,np.clip(1/(1+np.exp(-(co-2.0)*0.9-(wl-120)/20))
              +rng.normal(0,0.08,n),0.02,0.98)).astype(int)
    succ    = rng.binomial(1,np.clip(1/(1+np.exp(-(pct-65)/12+blk*0.35
              -dr*0.05+sc.clip(0,15)*0.08))+rng.normal(0,0.07,n),0.03,0.97)).astype(int)
    eh      = np.round(rng.exponential(7,n).clip(1,50),1)
    sp      = rng.choice([1,2,3,5,8],n)
    pris    = rng.choice(PRIORITIES,n,p=[0.3,0.5,0.2])
    ttr     = np.round(eh*np.where(pris=="High",rng.uniform(0.7,1.2,n),
              rng.uniform(0.8,1.8,n))+rng.normal(0,1.5,n),1).clip(0.5,80)
    itypes  = rng.choice(ISSUE_TYPES,n,p=[0.3,0.3,0.2,0.1,0.1])
    asgns   = rng.choice(ASSIGNEES,n)
    lbls    = rng.choice(LABELS,n)
    sums    = rng.choice(SUMMARIES,n)
    sprns   = rng.integers(1,11,n)
    stats   = rng.choice(STATUSES,n,p=[0.15,0.35,0.2,0.25,0.05])
    rows = [{
        "Planned_Story_Points_Sprint":   float(pln[i]),
        "Completed_Story_Points":        float(cmp[i]),
        "Percent_Done":                  float(pct[i]),
        "Days_Remaining_Sprint":         float(dr[i]),
        "Historical_Velocity":           float(hv[i]),
        "Blocked_Stories":               float(blk[i]),
        "Scope_Change":                  float(sc[i]),
        "Success_Label":                 int(succ[i]),
        "Sprint_Number":                 int(sprns[i]),
        "Planned_Story_Points_Resource": float(round(hist_sp[i]*0.9,1)),
        "Current_Assigned_SP":           float(round(asgn_sp[i],1)),
        "Historical_Avg_SP":             float(hist_sp[i]),
        "Remaining_Days_Resource":       float(rdr[i]),
        "High_Priority_Tasks_Resource":  float(hpt[i]),
        "Current_Workload_Percent":      float(wl[i]),
        "Expected_Overload":             int(ol[i]),
        "Issue_Type":                    str(itypes[i]),
        "Priority":                      str(pris[i]),
        "Original_Estimate_Hours":       float(eh[i]),
        "Story_Points_Issue":            float(sp[i]),
        "Resolution_Time_Hours":         float(ttr[i]),
        "Total_SP_This_Sprint":          float(pln[i]),
        "Historical_Avg_SP_Burnout":     float(hist_sp[i]*0.85),
        "High_Priority_Tasks_Burnout":   float(hpt[i]),
        "Consecutive_Overloads":         int(co[i]),
        "Risk_Flag":                     int(risk[i]),
        "Summary":                       str(sums[i]),
        "Labels":                        str(lbls[i]),
        "Original_Estimate_Resource":    float(eh[i]),
        "Story_Points_Resource":         float(sp[i]),
        "Assignee_Resource":             str(asgns[i]),
        "Assignee":                      str(asgns[i]),
        "Issue_ID":                      f"AGI-{i+1:04d}",
        "Status":                        str(stats[i]),
    } for i in range(n)]
    dist = {"success_pct":round(float(succ.mean())*100,1),
            "overload_pct":round(float(ol.mean())*100,1),
            "burnout_pct":round(float(risk.mean())*100,1)}
    return rows, dist

def gen_sprints(n=10, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(1,n+1):
        pln=float(rng.integers(40,100)); hv=float(rng.integers(30,80))
        blk=int(rng.integers(0,6)); cmp=float(round(np.clip(pln*rng.uniform(0.4,1.05),0,pln),1))
        pct=round(cmp/pln*100,1); wl=round(float(rng.uniform(70,160)),1); co=int(rng.integers(0,5))
        out.append({"sprint_id":f"SP-{i:03d}","sprint_name":f"Sprint {i}",
            "state":"active" if i==n else "closed","planned_sp":pln,"completed_sp":cmp,
            "percent_done":pct,"velocity":hv,"blocked":blk,"workload_pct":wl,"consec_ol":co,
            "success":int(pct>60 and blk<4),"risk":int(co>=2 or wl>130),
            "start":(datetime.now()-timedelta(days=(n-i)*14)).strftime("%Y-%m-%d"),
            "end":(datetime.now()-timedelta(days=(n-i)*14-14)).strftime("%Y-%m-%d")})
    return out

def gen_team(seed=42):
    rng=np.random.default_rng(seed)
    out=[]
    for a in ASSIGNEES:
        wl=round(float(rng.uniform(70,160)),1); co=int(rng.integers(0,5))
        sp=int(rng.integers(20,80)); hv=float(rng.integers(25,65)); hpt=int(rng.integers(0,6))
        out.append({"name":a,"role":str(rng.choice(["Senior Dev","Dev","Lead","QA","DevOps"])),
            "workload_pct":wl,"overload":int(wl>110),"total_sp":sp,
            "hist_avg_sp":round(hv*0.85,1),"high_priority":hpt,"consec_ol":co,
            "risk_flag":int(co>=2 or wl>130),
            "burnout_score":round(min(100,co*15+max(0,wl-100)*0.5),1),
            "health_score":round(max(0,100-co*15-max(0,wl-100)*0.5),1)})
    return out

_dataset, _dist = generate_ml_dataset(500)
_sprints         = gen_sprints(10)
_team            = gen_team()
_comments: dict  = {}

# ══════════════════════════════════════════════════════════════════════════
# HTML LANDING PAGE — Forest green + cream editorial
# ══════════════════════════════════════════════════════════════════════════
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Agile Intelligence — Data API</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--cream:#f5f0e8;--cream2:#ede7d9;--cream3:#ddd5c4;--forest:#1a3c2e;
  --forest2:#234d3b;--forest3:#2d6147;--moss:#4a7c59;--sage:#7aad8a;
  --fern:#a8c8a0;--ink:#1a2420;--bark:#5c4a3a;--stone:#8a8070;
  --gold:#c4973a;--amber:#e8b84b;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{background:var(--cream);color:var(--ink);font-family:'DM Sans',sans-serif;min-height:100vh;}
nav{background:var(--forest);padding:0 3rem;height:58px;display:flex;
  align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;
  border-bottom:2px solid var(--forest3);}
.logo{font-family:'DM Serif Display',serif;font-size:1.1rem;color:var(--cream);
  letter-spacing:0.02em;display:flex;align-items:center;gap:12px;}
.gem{width:10px;height:10px;background:var(--amber);border-radius:2px;transform:rotate(45deg);}
.nav-r{display:flex;gap:10px;align-items:center;}
.npill{font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.08em;
  text-transform:uppercase;padding:3px 11px;border-radius:2px;text-decoration:none;}
.ng{background:var(--moss);color:var(--cream);}
.na{background:transparent;color:var(--amber);border:1px solid #e8b84b40;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;
  background:var(--amber);margin-right:5px;animation:blink 2s infinite;}
.hero{padding:6rem 3rem 4rem;max-width:960px;margin:0 auto;}
.eyebrow{font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.18em;
  text-transform:uppercase;color:var(--moss);margin-bottom:1.4rem;
  display:flex;align-items:center;gap:10px;}
.eyebrow::before{content:'';display:block;width:32px;height:1px;background:var(--moss);}
h1{font-family:'DM Serif Display',serif;font-size:3.6rem;color:var(--forest);
  line-height:1.06;letter-spacing:-0.02em;margin-bottom:1.2rem;}
h1 em{color:var(--moss);font-style:italic;}
.sub{font-size:1rem;color:var(--bark);line-height:1.8;max-width:520px;
  margin-bottom:2.5rem;font-weight:300;}
.btns{display:flex;gap:12px;flex-wrap:wrap;}
.btn{font-family:'DM Mono',monospace;font-size:0.7rem;font-weight:500;
  letter-spacing:0.08em;text-transform:uppercase;padding:11px 22px;
  border-radius:2px;text-decoration:none;transition:all 0.15s;}
.bp{background:var(--forest);color:var(--cream);}
.bp:hover{background:var(--forest3);}
.bo{background:transparent;color:var(--forest);border:1px solid #1a3c2e40;}
.bo:hover{background:var(--cream2);border-color:var(--forest);}
.rule{border:none;border-top:1px solid var(--cream3);margin:0 3rem;}
.stats{display:grid;grid-template-columns:repeat(4,1fr);max-width:960px;
  margin:3rem auto;padding:0 3rem;gap:1px;background:var(--cream3);}
.sc{background:var(--cream);padding:1.4rem 1.8rem;}
.sv{font-family:'DM Serif Display',serif;font-size:2.2rem;color:var(--forest);}
.sl{font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;
  text-transform:uppercase;color:var(--stone);margin-top:6px;}
.section{max-width:960px;margin:0 auto;padding:3rem;}
.sec-lbl{font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.16em;
  text-transform:uppercase;color:var(--moss);display:flex;align-items:center;
  gap:10px;margin-bottom:1.8rem;}
.sec-lbl::after{content:'';flex:1;height:1px;background:var(--cream3);}
.epg{display:grid;grid-template-columns:1fr 1fr;gap:1px;
  background:var(--cream3);border:1px solid var(--cream3);}
.epc{background:var(--cream);padding:1.2rem 1.5rem;transition:background 0.12s;}
.epc:hover{background:var(--cream2);}
.etop{display:flex;align-items:center;gap:10px;margin-bottom:5px;}
.m{font-family:'DM Mono',monospace;font-size:0.58rem;font-weight:500;
  letter-spacing:0.08em;padding:2px 8px;border-radius:1px;}
.get{background:#1a3c2e18;color:var(--forest);}
.post{background:#c4973a22;color:var(--bark);}
.ep{font-family:'DM Mono',monospace;font-size:0.78rem;color:var(--forest3);font-weight:500;}
.ed{font-size:0.76rem;color:var(--stone);line-height:1.5;}
.dist{background:var(--forest);padding:2rem 2.5rem;margin-top:2rem;
  display:grid;grid-template-columns:repeat(3,1fr);gap:2rem;}
.dv{font-family:'DM Serif Display',serif;font-size:2.2rem;color:var(--amber);}
.dl{font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;
  text-transform:uppercase;color:var(--fern);margin-top:4px;}
footer{background:var(--forest);color:var(--fern);font-family:'DM Mono',monospace;
  font-size:0.62rem;letter-spacing:0.08em;text-align:center;
  padding:1.6rem;margin-top:4rem;border-top:1px solid var(--forest3);}
</style>
</head>
<body>
<nav>
  <div class="logo"><div class="gem"></div>Agile Intelligence</div>
  <div class="nav-r">
    <span class="npill ng"><span class="dot"></span>LIVE</span>
    <span class="npill na">v3.0 · REST API</span>
    <a class="npill na" href="/docs">Swagger ↗</a>
    <a class="npill na" href="/redoc">ReDoc ↗</a>
  </div>
</nav>
<div class="hero">
  <div class="eyebrow">Agile Intelligence · Data API Server</div>
  <h1>Your agile data,<br><em>intelligently served.</em></h1>
  <p class="sub">A high-fidelity REST API generating realistic agile sprint, team, and
  issue data — engineered for ML with probabilistic labels and per-row noise.
  Connect the Agile Intelligence Dashboard to this server.</p>
  <div class="btns">
    <a class="btn bp" href="/docs">Explore API Docs</a>
    <a class="btn bo" href="/api/dataset/ml">ML Dataset JSON</a>
    <a class="btn bo" href="/api/health">Health Check</a>
    <a class="btn bo" href="/api/analytics/summary">Analytics</a>
  </div>
</div>
<hr class="rule">
<div class="stats">
  <div class="sc"><div class="sv">__N__</div><div class="sl">ML Records</div></div>
  <div class="sc"><div class="sv">16</div><div class="sl">Endpoints</div></div>
  <div class="sc"><div class="sv">32</div><div class="sl">Feature Columns</div></div>
  <div class="sc"><div class="sv">5</div><div class="sl">ML Objectives</div></div>
</div>
<div class="section">
  <div class="sec-lbl">API Endpoints</div>
  <div class="epg">
    <div class="epc"><div class="etop"><span class="m get">GET</span>
      <span class="ep">/api/dataset/ml</span></div>
      <div class="ed">Full ML-ready dataset — 500 rows, 32 cols, probabilistic labels</div></div>
    <div class="epc"><div class="etop"><span class="m get">GET</span>
      <span class="ep">/api/sprints</span></div>
      <div class="ed">All sprint records with velocity, completion %, risk flags</div></div>
    <div class="epc"><div class="etop"><span class="m get">GET</span>
      <span class="ep">/api/team</span></div>
      <div class="ed">Team members with burnout scores, workload %, health index</div></div>
    <div class="epc"><div class="etop"><span class="m get">GET</span>
      <span class="ep">/api/analytics/summary</span></div>
      <div class="ed">Project health score, KPIs, risk counts</div></div>
    <div class="epc"><div class="etop"><span class="m get">GET</span>
      <span class="ep">/api/analytics/velocity</span></div>
      <div class="ed">Sprint velocity trend — completed vs planned</div></div>
    <div class="epc"><div class="etop"><span class="m get">GET</span>
      <span class="ep">/api/analytics/burnout</span></div>
      <div class="ed">Burnout risk ranked by team member</div></div>
    <div class="epc"><div class="etop"><span class="m get">GET</span>
      <span class="ep">/api/search</span></div>
      <div class="ed">Filter issues by keyword, assignee, priority, status</div></div>
    <div class="epc"><div class="etop"><span class="m get">GET</span>
      <span class="ep">/api/refresh</span></div>
      <div class="ed">Regenerate all data with a new random seed</div></div>
    <div class="epc"><div class="etop"><span class="m post">POST</span>
      <span class="ep">/api/issues</span></div>
      <div class="ed">Create new issue with type, priority, story points</div></div>
    <div class="epc"><div class="etop"><span class="m post">POST</span>
      <span class="ep">/api/issues/{id}/comment</span></div>
      <div class="ed">Add comment to an existing issue</div></div>
    <div class="epc"><div class="etop"><span class="m post">POST</span>
      <span class="ep">/api/issues/{id}/transition</span></div>
      <div class="ed">Transition status — To Do → In Progress → Done</div></div>
    <div class="epc"><div class="etop"><span class="m get">GET</span>
      <span class="ep">/api/health</span></div>
      <div class="ed">Health check — status, version, record counts</div></div>
  </div>
  <div class="dist">
    <div><div class="dv">__S__%</div><div class="dl">Sprint Success Rate</div></div>
    <div><div class="dv">__O__%</div><div class="dl">Overload Rate</div></div>
    <div><div class="dv">__B__%</div><div class="dl">Burnout Risk Rate</div></div>
  </div>
</div>
<footer>AGILE INTELLIGENCE API · v3.0.0 · FASTAPI · __Y__</footer>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse, tags=["Root"])
def root():
    return (HTML_PAGE
            .replace("__N__", str(len(_dataset)))
            .replace("__S__", str(_dist['success_pct']))
            .replace("__O__", str(_dist['overload_pct']))
            .replace("__B__", str(_dist['burnout_pct']))
            .replace("__Y__", str(datetime.now().year)))

# ══════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════
@app.get("/api/health", tags=["Admin"])
def health():
    return {"status":"ok","version":"3.0.0","timestamp":datetime.now().isoformat(),
            "records":{"ml_dataset":len(_dataset),"sprints":len(_sprints),"team":len(_team)}}

@app.get("/api/refresh", tags=["Admin"])
def refresh():
    global _dataset,_dist,_sprints,_team
    seed=random.randint(1,9999)
    _dataset,_dist=generate_ml_dataset(500,seed)
    _sprints=gen_sprints(10,seed); _team=gen_team(seed)
    return {"status":"refreshed","seed":seed,"timestamp":datetime.now().isoformat()}

@app.get("/api/dataset/ml", tags=["ML"])
def get_ml_dataset(limit:int=Query(500,le=1000)):
    rows=_dataset[:limit]
    return {"count":len(rows),"columns":list(rows[0].keys()) if rows else [],
            "source":"Agile Intelligence API v3.0.0","fetched_at":datetime.now().isoformat(),
            "label_distribution":_dist,"records":rows}

@app.get("/api/sprints", tags=["Sprints"])
def get_sprints(state:Optional[str]=None,limit:int=Query(20,le=50)):
    d=_sprints if not state else [s for s in _sprints if s["state"]==state]
    return {"count":len(d[:limit]),"sprints":d[:limit]}

@app.get("/api/sprints/{sprint_id}", tags=["Sprints"])
def get_sprint(sprint_id:str):
    s=next((x for x in _sprints if x["sprint_id"]==sprint_id),None)
    if not s: raise HTTPException(404,f"Sprint {sprint_id} not found")
    return s

@app.get("/api/team", tags=["Team"])
def get_team():
    return {"count":len(_team),"team":_team}

@app.get("/api/team/{name}", tags=["Team"])
def get_member(name:str):
    m=next((x for x in _team if x["name"].lower()==name.lower()),None)
    if not m: raise HTTPException(404,f"Member {name} not found")
    return m

@app.get("/api/search", tags=["Search"])
def search(q:str="",assignee:Optional[str]=None,
           priority:Optional[str]=None,status:Optional[str]=None,
           limit:int=Query(50,le=200)):
    d=_dataset
    if q:        d=[i for i in d if q.lower() in i["Summary"].lower() or q.lower() in i["Labels"]]
    if assignee: d=[i for i in d if i["Assignee"]==assignee]
    if priority: d=[i for i in d if i["Priority"]==priority]
    if status:   d=[i for i in d if i["Status"]==status]
    return {"query":q,"count":len(d[:limit]),"results":d[:limit]}

@app.get("/api/analytics/summary", tags=["Analytics"])
def summary():
    done=sum(1 for i in _dataset if i["Status"]=="Done")
    blk =sum(1 for i in _dataset if i["Status"]=="Blocked")
    hp  =sum(1 for i in _dataset if i["Priority"]=="High")
    risk=sum(1 for s in _sprints if not s["success"])
    burn=sum(1 for t in _team if t["risk_flag"])
    over=sum(1 for t in _team if t["overload"])
    av  =round(sum(s["velocity"] for s in _sprints)/len(_sprints),1)
    ac  =round(sum(s["percent_done"] for s in _sprints)/len(_sprints),1)
    h   =min(100,max(0,100-risk*8-burn*6-blk//5*2-over*5))
    return {"health_score":h,"total_issues":len(_dataset),"done":done,"blocked":blk,
            "high_priority":hp,"sprints_at_risk":risk,"burnout_risk":burn,"overloaded":over,
            "avg_velocity":av,"avg_completion_pct":ac,"team_size":len(_team),
            "active_sprint":next((s for s in _sprints if s["state"]=="active"),None),
            "generated_at":datetime.now().isoformat()}

@app.get("/api/analytics/velocity", tags=["Analytics"])
def velocity():
    return {"sprints":[{"id":s["sprint_id"],"name":s["sprint_name"],"state":s["state"],
            "velocity":s["velocity"],"completed":s["completed_sp"],
            "planned":s["planned_sp"],"pct":s["percent_done"],"risk":not s["success"]}
            for s in sorted(_sprints,key=lambda x:x["sprint_id"])]}

@app.get("/api/analytics/burnout", tags=["Analytics"])
def burnout_report():
    return {"team":sorted([{"name":t["name"],"burnout_score":t["burnout_score"],
            "health_score":t["health_score"],"workload_pct":t["workload_pct"],
            "consec_ol":t["consec_ol"],"risk_flag":t["risk_flag"],
            "high_priority":t["high_priority"]}
            for t in _team],key=lambda x:-x["burnout_score"])}

class IssueCreate(BaseModel):
    summary:str; issue_type:str="Task"; priority:str="Medium"
    assignee:str="Unassigned"; story_points:int=3; labels:str="general"

class CommentCreate(BaseModel):
    text:str; author:str="API User"

class TransitionCreate(BaseModel):
    status:str

@app.post("/api/issues",tags=["Write"],status_code=201)
def create_issue(body:IssueCreate):
    nid=f"AGI-{len(_dataset)+1:04d}"; eh=float(body.story_points*np.random.uniform(1.5,2.5))
    row={"Issue_ID":nid,"Summary":body.summary,"Issue_Type":body.issue_type,
         "Priority":body.priority,"Status":"To Do","Assignee":body.assignee,
         "Story_Points_Issue":float(body.story_points),
         "Original_Estimate_Hours":round(eh,1),"Labels":body.labels,"Resolution_Time_Hours":round(eh,1)}
    _dataset.append({**{k:0 for k in _dataset[0]},**row})
    return {"created":True,"issue_id":nid,"issue":row}

@app.post("/api/issues/{issue_id}/comment",tags=["Write"])
def add_comment(issue_id:str,body:CommentCreate):
    if issue_id not in _comments: _comments[issue_id]=[]
    c={"id":len(_comments[issue_id])+1,"author":body.author,
       "text":body.text,"at":datetime.now().isoformat()}
    _comments[issue_id].append(c); return {"added":True,"comment":c}

@app.post("/api/issues/{issue_id}/transition",tags=["Write"])
def transition(issue_id:str,body:TransitionCreate):
    valid=["To Do","In Progress","In Review","Done","Blocked"]
    if body.status not in valid: raise HTTPException(400,f"Status must be one of {valid}")
    row=next((r for r in _dataset if r.get("Issue_ID")==issue_id),None)
    if not row: raise HTTPException(404,f"{issue_id} not found")
    old=row["Status"]; row["Status"]=body.status
    return {"transitioned":True,"issue_id":issue_id,"from":old,"to":body.status}
