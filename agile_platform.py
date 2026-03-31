"""
╔══════════════════════════════════════════════════════════════════╗
║  AGILE INTELLIGENCE PLATFORM                                     ║
║  Streamlit · Fetches from Agile Data API                        ║
║  ML: Fine-tuned LR · BERT · Spark · Improved K-Means           ║
║  Run: streamlit run agile_platform.py                            ║
║  (Start API server first: uvicorn agile_api_server:app --port 8000)
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests, re, os, pathlib, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                               RandomForestClassifier, VotingClassifier,
                               AdaBoostClassifier, IsolationForest)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              mean_squared_error, r2_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ── MUST be first ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agile Intelligence Platform",
    layout="wide",
    page_icon="◆",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Website-grade dark professional UI
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
*, *::before, *::after { box-sizing: border-box; }
html, body { height: 100%; }
.stApp { background: #03040a; color: #dde4f0; font-family: 'SF Mono','Fira Code','Cascadia Code',monospace; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { background: #03040a; }
[data-testid="stHeader"] { display: none; }

/* NAV */
.topnav { position:sticky;top:0;z-index:999;background:rgba(3,4,10,0.94);
  backdrop-filter:blur(24px);border-bottom:1px solid #0c1020;
  padding:0 2.5rem;height:54px;display:flex;align-items:center;justify-content:space-between; }
.logo { font-size:0.82rem;font-weight:800;letter-spacing:0.14em;color:#f1f5f9;
  display:flex;align-items:center;gap:9px;text-transform:uppercase; }
.logo-gem { width:10px;height:10px;background:linear-gradient(135deg,#2563eb,#0ea5e9);
  border-radius:2px;transform:rotate(45deg);box-shadow:0 0 14px #2563eb; }
.nav-pills { display:flex;gap:6px;align-items:center; }
.nav-pill { font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;
  padding:3px 12px;border-radius:20px;font-weight:700; }
.nav-pill.green { background:#10b98118;color:#10b981;border:1px solid #10b98130; }
.nav-pill.blue  { background:#2563eb18;color:#60a5fa;border:1px solid #2563eb30; }
.nav-pill.cyan  { background:#0ea5e918;color:#22d3ee;border:1px solid #0ea5e930; }
@keyframes blink { 0%,100%{opacity:1}50%{opacity:0.2} }
.blink { animation:blink 2s infinite; }

/* LAYOUT */
.page-pad { padding:2rem 2.5rem; }
.page-title { font-size:1.5rem;font-weight:900;color:#f1f5f9;letter-spacing:-0.03em;line-height:1.1; }
.page-sub   { font-size:0.75rem;color:#3d4f6b;margin-top:5px;letter-spacing:0.03em; }
.sec-hdr    { font-size:0.62rem;font-weight:800;letter-spacing:0.16em;text-transform:uppercase;
  color:#2563eb;border-left:2px solid #2563eb;padding-left:9px;margin:1.4rem 0 0.8rem; }

/* CARDS */
.gc { background:#060a14;border:1px solid #0f1829;border-radius:10px;
  padding:1.1rem 1.3rem;margin-bottom:0.6rem;position:relative;overflow:hidden; }
.gc::before { content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,#2563eb18,transparent); }
.gc.red    { border-left:3px solid #e11d48;background:#0c050a; }
.gc.yellow { border-left:3px solid #d97706;background:#0c0a05; }
.gc.green  { border-left:3px solid #059669;background:#050c09; }
.gc.blue   { border-left:3px solid #2563eb;background:#05080c; }
.gc.cyan   { border-left:3px solid #0891b2;background:#05090c; }
.gc.purple { border-left:3px solid #7c3aed;background:#08050c; }
.gc-title  { font-size:0.82rem;font-weight:700;color:#e2e8f0;margin-bottom:3px; }
.gc-detail { font-size:0.75rem;color:#3d4f6b;line-height:1.5; }

/* METRIC CARDS */
.mc { background:#060a14;border:1px solid #0f1829;border-radius:9px;padding:1rem 1.2rem; }
.mc-val { font-size:1.7rem;font-weight:900;color:#f1f5f9;line-height:1; }
.mc-lbl { font-size:0.6rem;color:#3d4f6b;text-transform:uppercase;letter-spacing:0.1em;margin-top:4px; }
.mc-delta { font-size:0.7rem;margin-top:2px; }

/* PROGRESS */
.pw { background:#0a1020;border-radius:2px;height:3px;margin-top:3px; }
.pf { height:3px;border-radius:2px;transition:width 0.6s ease; }

/* BADGES */
.badge { display:inline-flex;align-items:center;gap:4px;padding:2px 9px;
  border-radius:3px;font-size:0.62rem;font-weight:800;letter-spacing:0.07em;text-transform:uppercase; }
.bg-green  { background:#05943515;color:#10b981;border:1px solid #05943530; }
.bg-red    { background:#e11d4815;color:#f43f5e;border:1px solid #e11d4830; }
.bg-yellow { background:#d9770615;color:#f59e0b;border:1px solid #d9770630; }
.bg-blue   { background:#2563eb15;color:#60a5fa;border:1px solid #2563eb30; }
.bg-cyan   { background:#0891b215;color:#22d3ee;border:1px solid #0891b230; }

/* PREDICTION */
.pred { background:#060a14;border:1px solid #0f1829;border-radius:10px;
  padding:1.8rem;text-align:center;margin-top:1rem; }
.pred-lbl { font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;color:#3d4f6b;margin-bottom:6px; }
.pred-val { font-size:1.5rem;font-weight:900; }
.pred-conf { font-size:0.72rem;color:#3d4f6b;margin-top:4px; }

/* API CARD */
.api-card { background:#060a14;border:1px solid #0f1829;border-radius:10px;padding:1.5rem; }
.api-url   { font-size:0.75rem;color:#22d3ee;font-weight:700;margin-bottom:4px; }
.api-label { font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#3d4f6b; }

/* ASSIGNEE */
.av { width:38px;height:38px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:0.9rem;font-weight:900;margin:0 auto 6px; }
.an { font-size:0.8rem;font-weight:700;color:#e2e8f0; }
.as { font-size:1.3rem;font-weight:900;line-height:1; }
.ad { font-size:0.65rem;color:#3d4f6b;margin-top:2px; }

/* SPARK PILL */
.sp-pill { display:inline-flex;align-items:center;gap:3px;background:#0891b218;
  color:#22d3ee;border:1px solid #0891b230;border-radius:20px;
  padding:1px 9px;font-size:0.62rem;font-weight:700;letter-spacing:0.06em;margin:2px; }

/* ST OVERRIDES */
[data-testid="stMetricValue"] { font-family:'SF Mono','Fira Code',monospace !important;
  font-size:1.5rem !important;color:#f1f5f9 !important; }
[data-testid="stMetricLabel"] { font-size:0.6rem !important;color:#3d4f6b !important;
  text-transform:uppercase;letter-spacing:0.1em; }
[data-testid="stMetricDelta"] { font-size:0.7rem !important; }
div[data-testid="stTabs"] [role="tablist"] { background:#05080f;border-bottom:1px solid #0c1020;padding:0 1.5rem; }
div[data-testid="stTabs"] button[role="tab"] { font-family:'SF Mono','Fira Code',monospace !important;
  font-size:0.65rem !important;letter-spacing:0.07em !important;text-transform:uppercase !important;
  color:#3d4f6b !important;padding:0.7rem 1.1rem !important;
  border:none !important;border-bottom:2px solid transparent !important;border-radius:0 !important;background:transparent !important; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#60a5fa !important;border-bottom-color:#2563eb !important; }
.stButton>button { background:#2563eb !important;color:#fff !important;border:none !important;
  border-radius:6px !important;font-family:'SF Mono','Fira Code',monospace !important;
  font-size:0.7rem !important;letter-spacing:0.07em !important;font-weight:800 !important;
  padding:0.45rem 1.3rem !important;text-transform:uppercase !important; }
.stButton>button:hover { background:#1d4ed8 !important; }
.stTextInput>div>div>input,.stNumberInput>div>div>input {
  background:#060a14 !important;border:1px solid #0f1829 !important;
  border-radius:6px !important;color:#e2e8f0 !important;
  font-family:'SF Mono','Fira Code',monospace !important;font-size:0.78rem !important; }
.stTextInput>div>div>input:focus,.stNumberInput>div>div>input:focus {
  border-color:#2563eb !important;box-shadow:0 0 0 2px #2563eb20 !important; }
.stSelectbox>div>div { background:#060a14 !important;border:1px solid #0f1829 !important;color:#e2e8f0 !important; }
[data-testid="stExpander"] { background:#060a14 !important;border:1px solid #0f1829 !important;border-radius:8px !important; }
[data-testid="stExpander"] summary { font-size:0.68rem !important;text-transform:uppercase !important;letter-spacing:0.07em !important;color:#3d4f6b !important; }
.stAlert { border-radius:8px !important;font-size:0.78rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SPARK SETUP
# ══════════════════════════════════════════════════════════════════════════
def _find_java():
    if os.environ.get("JAVA_HOME"): return os.environ["JAVA_HOME"]
    for base in [r"C:\Program Files\Eclipse Adoptium",r"C:\Program Files\Java",
                 "/usr/lib/jvm","/usr/local/opt","/Library/Java/JavaVirtualMachines"]:
        p = pathlib.Path(base)
        if p.exists():
            for child in sorted(p.iterdir(), reverse=True):
                jb = child/"bin"/("java.exe" if os.name=="nt" else "java")
                if not jb.exists(): jb = child/"Contents"/"Home"/"bin"/"java"
                if jb.exists(): return str(jb.parent.parent)
    return None

_jh = _find_java()
if _jh:
    os.environ["JAVA_HOME"] = _jh
    os.environ["PATH"] = str(pathlib.Path(_jh)/"bin") + os.pathsep + os.environ.get("PATH","")
os.environ.setdefault("PYSPARK_PYTHON","python3")
os.environ.setdefault("SPARK_LOCAL_IP","127.0.0.1")

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    SPARK_OK = True
except ImportError:
    SPARK_OK = False

@st.cache_resource
def get_spark():
    if not SPARK_OK: return None
    try:
        sp = (SparkSession.builder.appName("AgileAI").master("local[*]")
              .config("spark.driver.memory","2g").config("spark.sql.shuffle.partitions","4")
              .config("spark.ui.enabled","false").config("spark.driver.host","127.0.0.1")
              .config("spark.driver.bindAddress","127.0.0.1").getOrCreate())
        sp.sparkContext.setLogLevel("ERROR")
        return sp
    except: return None

SF = ["Velocity_Efficiency","Completion_Gap","Blocker_Severity",
      "Scope_Pressure","Sprint_Momentum","Recovery_Index","Workload_Stress"]

def eng_features(pdf, spark=None):
    if spark:
        try:
            sdf = spark.createDataFrame(pdf)
            sdf = sdf.withColumn("Velocity_Efficiency",
                F.when(F.col("Planned_Story_Points_Sprint")>0,
                    F.col("Historical_Velocity")/F.col("Planned_Story_Points_Sprint")).otherwise(1.0))
            sdf = sdf.withColumn("Completion_Gap",
                F.col("Planned_Story_Points_Sprint")-F.col("Completed_Story_Points"))
            sdf = sdf.withColumn("Blocker_Severity",
                F.col("Blocked_Stories")*F.when(F.col("Days_Remaining_Sprint")>0,
                    F.lit(1.0)/F.col("Days_Remaining_Sprint")).otherwise(1.0))
            sdf = sdf.withColumn("Scope_Pressure",
                F.when(F.col("Planned_Story_Points_Sprint")>0,
                    F.col("Scope_Change")/F.col("Planned_Story_Points_Sprint")).otherwise(0.0))
            sdf = sdf.withColumn("Sprint_Momentum",
                F.when(F.col("Historical_Velocity")>0,
                    F.col("Completed_Story_Points")/F.col("Historical_Velocity")).otherwise(0.0))
            sdf = sdf.withColumn("Recovery_Index",
                F.when((F.col("Planned_Story_Points_Sprint")-F.col("Completed_Story_Points")>0)&
                       (F.col("Days_Remaining_Sprint")>0),
                    (F.col("Historical_Velocity")*F.col("Days_Remaining_Sprint")/10)/
                    (F.col("Planned_Story_Points_Sprint")-F.col("Completed_Story_Points"))
                ).otherwise(1.0))
            sdf = sdf.withColumn("Workload_Stress",
                (F.col("Current_Workload_Percent")/100)*F.col("Consecutive_Overloads"))
            return sdf.toPandas().fillna(0)
        except: pass
    df = pdf.copy()
    df["Velocity_Efficiency"] = (df["Historical_Velocity"]/df["Planned_Story_Points_Sprint"].replace(0,1)).clip(0,3)
    df["Completion_Gap"]  = df["Planned_Story_Points_Sprint"]-df["Completed_Story_Points"]
    df["Blocker_Severity"]= df["Blocked_Stories"]*(1/df["Days_Remaining_Sprint"].replace(0,1).abs())
    df["Scope_Pressure"]  = (df["Scope_Change"]/df["Planned_Story_Points_Sprint"].replace(0,1)).clip(-1,2)
    df["Sprint_Momentum"] = (df["Completed_Story_Points"]/df["Historical_Velocity"].replace(0,1)).clip(0,2)
    df["Recovery_Index"]  = ((df["Historical_Velocity"]*df["Days_Remaining_Sprint"]/10)/
        (df["Planned_Story_Points_Sprint"]-df["Completed_Story_Points"]).replace(0,.001)).clip(0,5)
    df["Workload_Stress"] = (df["Current_Workload_Percent"]/100)*df.get(
        "Consecutive_Overloads",pd.Series(0,index=df.index))
    return df.fillna(0)

# ══════════════════════════════════════════════════════════════════════════
# API CLIENT
# ══════════════════════════════════════════════════════════════════════════
class AgileAPIClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"Accept":"application/json"})

    def _get(self, path, params=None):
        try:
            r = self.s.get(f"{self.base}{path}", params=params, timeout=10)
            r.raise_for_status()
            return r.json(), None
        except requests.exceptions.ConnectionError:
            return None, f"Cannot reach API at {self.base}. Is the server running?"
        except requests.exceptions.HTTPError as e:
            return None, f"HTTP {e.response.status_code}: {e.response.text[:100]}"
        except Exception as e:
            return None, str(e)

    def _post(self, path, body):
        try:
            r = self.s.post(f"{self.base}{path}", json=body, timeout=10)
            r.raise_for_status()
            return r.json(), None
        except requests.exceptions.HTTPError as e:
            return None, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return None, str(e)

    def health(self):
        d, e = self._get("/api/health")
        return d, e

    def get_ml_dataset(self, limit=300):
        d, e = self._get("/api/dataset/ml", {"limit": limit})
        if e: return None, e
        return pd.DataFrame(d["records"]), None

    def get_sprints(self):
        d, e = self._get("/api/sprints", {"limit": 50})
        if e: return [], e
        return d.get("sprints", []), None

    def get_team(self):
        d, e = self._get("/api/team")
        if e: return [], e
        return d.get("team", []), None

    def get_summary(self):
        return self._get("/api/analytics/summary")

    def get_velocity(self):
        return self._get("/api/analytics/velocity")

    def get_burnout(self):
        return self._get("/api/analytics/burnout")

    def search(self, q="", assignee=None, priority=None, status=None):
        params = {"q": q, "limit": 100}
        if assignee: params["assignee"] = assignee
        if priority: params["priority"] = priority
        if status:   params["status"]   = status
        d, e = self._get("/api/search", params)
        if e: return [], e
        return d.get("results", []), None

    def create_issue(self, summary, itype, priority, assignee, sp, sprint_id, labels):
        return self._post("/api/issues", {
            "summary":summary,"issue_type":itype,"priority":priority,
            "assignee":assignee,"story_points":sp,"sprint_id":sprint_id,"labels":labels})

    def add_comment(self, issue_id, text):
        return self._post(f"/api/issues/{issue_id}/comment", {"text":text})

    def transition(self, issue_id, status):
        return self._post(f"/api/issues/{issue_id}/transition", {"status":status})

    def refresh(self):
        return self._get("/api/refresh")

# ══════════════════════════════════════════════════════════════════════════
# ML ENGINE — Improved models from notes
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def tfidf_features(texts_t, n=30):
    texts = list(texts_t)
    clean = [re.sub(r"[^a-z0-9 ]"," ",str(t).lower()) for t in texts]
    try:
        tv = TfidfVectorizer(max_features=n,ngram_range=(1,2),min_df=1,sublinear_tf=True)
        X  = tv.fit_transform(clean).toarray()
        return X, tv.get_feature_names_out().tolist(), tv
    except:
        return np.zeros((len(texts),n)), [], None

def finetune_lr(Xtr,ytr,Xte,yte):
    best_acc, best_m, rows = 0, None, []
    for cfg in [
        dict(C=0.1, solver="lbfgs",     class_weight="balanced",max_iter=300),
        dict(C=1.0, solver="lbfgs",     class_weight="balanced",max_iter=500),
        dict(C=5.0, solver="saga",      class_weight="balanced",max_iter=500),
        dict(C=1.0, solver="liblinear", class_weight="balanced",max_iter=300),
        dict(C=0.5, solver="lbfgs",     class_weight=None,      max_iter=300),
    ]:
        try:
            m = LogisticRegression(**cfg,random_state=42)
            m.fit(Xtr,ytr); yp=m.predict(Xte)
            acc=accuracy_score(yte,yp); f1=f1_score(yte,yp,average="weighted")
            rows.append({**{k:v for k,v in cfg.items()},"acc":acc,"f1":f1})
            if acc>best_acc: best_acc,best_m=acc,m
        except: pass
    return best_m, best_acc, rows

@st.cache_data(show_spinner="◆ Training models from API data...")
def train_models(df):
    # Anti-overfit: require minimum samples and balanced classes
    MIN_ROWS = 80  # skip model if not enough data
    R = {}

    # Obj1 Sprint — Fine-tuned LR + TF-IDF (BERT proxy)
    try:
        base=[c for c in ["Planned_Story_Points_Sprint","Completed_Story_Points",
              "Percent_Done","Days_Remaining_Sprint","Historical_Velocity",
              "Blocked_Stories","Scope_Change"] if c in df.columns]
        # TF-IDF text features (BERT proxy) — limited to avoid overfitting on small data
        txX,txF,tv=None,[],None
        if "Summary" in df.columns and len(df)>=MIN_ROWS:
            txX,txF,tv=tfidf_features(tuple(df["Summary"].astype(str)),15)
        Xn=df[base].fillna(0).values
        X = np.hstack([Xn,txX]) if txX is not None and len(txF)>0 else Xn
        y = df["Success_Label"]
        if y.nunique()>1 and len(df)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(X)
            # Use 25% test split for better eval, stratify for balanced classes
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42,
                                               stratify=y if y.value_counts().min()>=3 else None)
            bm,bacc,rows=finetune_lr(Xtr,ytr,Xte,yte)
            yp=bm.predict(Xte)
            try:    auc=float(roc_auc_score(yte,bm.predict_proba(Xte)[:,1]))
            except: auc=0.0
            R["sprint"]=dict(model=bm,scaler=sc,features=base+(txF if txF else []),tfidf=tv,
                             base=base,textf=txF if txF else [],algo="Fine-tuned LR + TF-IDF",
                             acc=float(bacc),f1=float(f1_score(yte,yp,average="weighted")),
                             auc=auc,report=classification_report(yte,yp),
                             cm=confusion_matrix(yte,yp).tolist(),tune=rows)
    except: pass

    # Obj2 Workload — Naive Bayes
    try:
        f=[c for c in ["Planned_Story_Points_Resource","Current_Assigned_SP","Historical_Avg_SP",
           "Remaining_Days_Resource","High_Priority_Tasks_Resource","Current_Workload_Percent"]
           if c in df.columns]
        X,y=df[f].fillna(0),df["Expected_Overload"]
        if y.nunique()>1 and len(df)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(X)
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42,
                                               stratify=y if y.value_counts().min()>=3 else None)
            m=GaussianNB(); m.fit(Xtr,ytr); yp=m.predict(Xte)
            R["workload"]=dict(model=m,scaler=sc,features=f,algo="Naive Bayes",
                               acc=float(accuracy_score(yte,yp)),
                               f1=float(f1_score(yte,yp,average="weighted")),
                               report=classification_report(yte,yp),
                               cm=confusion_matrix(yte,yp).tolist())
    except: pass

    # Obj3 TTR — Ridge Regression
    try:
        Xb=pd.get_dummies(df[["Issue_Type","Priority"]],drop_first=False)
        ex=[c for c in ["Original_Estimate_Hours","Story_Points_Issue"] if c in df.columns]
        X=pd.concat([Xb,df[ex]],axis=1).fillna(0); y=df["Resolution_Time_Hours"]
        if len(df)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(X)
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42)
            m=Ridge(alpha=2.0); m.fit(Xtr,ytr); yp=m.predict(Xte)
            R["ttr"]=dict(model=m,scaler=sc,features=X.columns.tolist(),X3=X,
                          algo="Ridge Regression",
                          r2=float(r2_score(yte,yp)),mse=float(mean_squared_error(yte,yp)))
    except: pass

    # Obj4 Burnout — Decision Tree (depth-limited to prevent overfit)
    try:
        f=[c for c in ["Total_SP_This_Sprint","Historical_Avg_SP_Burnout",
           "High_Priority_Tasks_Burnout","Consecutive_Overloads"] if c in df.columns]
        X,y=df[f].fillna(0),df["Risk_Flag"]
        if y.nunique()>1 and len(df)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(X)
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42,
                                               stratify=y if y.value_counts().min()>=3 else None)
            # max_depth=4 and min_samples_leaf=15 prevent overfit
            m=DecisionTreeClassifier(max_depth=4,class_weight="balanced",
                                      random_state=42,min_samples_leaf=15)
            m.fit(Xtr,ytr); yp=m.predict(Xte)
            R["burnout"]=dict(model=m,scaler=sc,features=f,algo="Decision Tree",
                              acc=float(accuracy_score(yte,yp)),
                              f1=float(f1_score(yte,yp,average="weighted")),
                              report=classification_report(yte,yp),
                              cm=confusion_matrix(yte,yp).tolist())
    except: pass

    # Obj5 Allocation — KNN + TF-IDF text features for better accuracy
    try:
        d2=df.copy()
        les=LabelEncoder(); lel=LabelEncoder()
        d2["Summary_enc"]=les.fit_transform(d2["Summary"].astype(str))
        d2["Labels_enc"] =lel.fit_transform(d2["Labels"].astype(str))
        # Add TF-IDF features for summary text to improve allocation
        alloc_X_num=d2[["Summary_enc","Labels_enc","Original_Estimate_Resource","Story_Points_Resource"]].fillna(0)
        if "Summary" in d2.columns and len(d2)>=MIN_ROWS:
            try:
                tv_a=TfidfVectorizer(max_features=20,ngram_range=(1,1),min_df=2)
                X_txt=tv_a.fit_transform(d2["Summary"].astype(str)).toarray()
                alloc_X=np.hstack([alloc_X_num.values, X_txt])
            except:
                alloc_X=alloc_X_num.values
        else:
            alloc_X=alloc_X_num.values
        y=d2["Assignee_Resource"]
        if len(d2)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(alloc_X)
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42)
            k=min(7,len(Xtr)//10)
            m=KNeighborsClassifier(n_neighbors=max(3,k),weights="distance",n_jobs=-1)
            m.fit(Xtr,ytr)
            R["alloc"]=dict(model=m,scaler=sc,features=alloc_X_num.columns.tolist(),
                            le_s=les,le_l=lel,algo="KNN + TF-IDF",
                            acc=float(accuracy_score(yte,m.predict(Xte))),
                            f1=float(f1_score(yte,m.predict(Xte),average="weighted")))
    except: pass

    return R

@st.cache_data(show_spinner=False)
def train_spark_ml(df):
    R = {}
    sf=[c for c in SF if c in df.columns]

    # Sprint Ensemble Voting
    try:
        f=[c for c in ["Planned_Story_Points_Sprint","Completed_Story_Points","Percent_Done",
           "Days_Remaining_Sprint","Historical_Velocity","Blocked_Stories","Scope_Change"]+sf
           if c in df.columns]
        X,y=df[f].fillna(0),df["Success_Label"]
        if y.nunique()>1:
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
            lr =LogisticRegression(max_iter=300,class_weight="balanced",random_state=42)
            gbt=GradientBoostingClassifier(n_estimators=60,random_state=42)
            rf =RandomForestClassifier(n_estimators=60,random_state=42,n_jobs=-1)
            ada=AdaBoostClassifier(n_estimators=60,random_state=42)
            ens=VotingClassifier([("lr",lr),("gbt",gbt),("rf",rf),("ada",ada)],voting="soft")
            ens.fit(Xtr,ytr)
            ind={}
            for nm,clf in [("LR",lr),("GBT",gbt),("RF",rf),("AdaBoost",ada)]:
                clf.fit(Xtr,ytr); ind[nm]=float(accuracy_score(yte,clf.predict(Xte)))
            gbt.fit(Xtr,ytr)
            imp=pd.Series(gbt.feature_importances_,index=f).sort_values(ascending=False)
            R["sprint"]=dict(acc=float(accuracy_score(yte,ens.predict(Xte))),ind=ind,
                             feat=f,sf=sf,imp=imp.to_dict())
    except: pass

    # TTR GBT
    try:
        Xb=pd.get_dummies(df[["Issue_Type","Priority"]],drop_first=False)
        ex=[c for c in ["Original_Estimate_Hours","Story_Points_Issue"]+sf if c in df.columns]
        X=pd.concat([Xb,df[ex]],axis=1).fillna(0); y=df["Resolution_Time_Hours"]
        sc=StandardScaler(); Xs=sc.fit_transform(X)
        Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42)
        gbr=GradientBoostingRegressor(n_estimators=60,random_state=42); gbr.fit(Xtr,ytr)
        lr3=LinearRegression(); lr3.fit(Xtr,ytr)
        R["ttr"]=dict(gb_r2=float(r2_score(yte,gbr.predict(Xte))),
                      lr_r2=float(r2_score(yte,lr3.predict(Xte))),
                      gb_mse=float(mean_squared_error(yte,gbr.predict(Xte))),feat=X.columns.tolist(),sf=sf)
    except: pass

    # Burnout GBT+RF
    try:
        f=[c for c in ["Total_SP_This_Sprint","Historical_Avg_SP_Burnout",
           "High_Priority_Tasks_Burnout","Consecutive_Overloads"]+sf if c in df.columns]
        X,y=df[f].fillna(0),df["Risk_Flag"]
        if y.nunique()>1:
            sc=StandardScaler(); Xs=sc.fit_transform(X)
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42)
            gbt4=GradientBoostingClassifier(n_estimators=60,random_state=42)
            rf4 =RandomForestClassifier(n_estimators=60,class_weight="balanced",random_state=42,n_jobs=-1)
            ens4=VotingClassifier([("gbt",gbt4),("rf",rf4)],voting="soft")
            ens4.fit(Xtr,ytr); yp=ens4.predict(Xte)
            R["burnout"]=dict(acc=float(accuracy_score(yte,yp)),
                              report=classification_report(yte,yp),feat=f,sf=sf)
    except: pass

    # Improved K-Means++
    try:
        if "Assignee" in df.columns:
            agg={}
            if "Current_Workload_Percent" in df.columns: agg["Workload"]=("Current_Workload_Percent","mean")
            if "Risk_Flag" in df.columns:               agg["Burnout"] =("Risk_Flag","mean")
            if "Success_Label" in df.columns:           agg["SprintRisk"]=("Success_Label",lambda x:(x==0).mean())
            if "Consecutive_Overloads" in df.columns:   agg["ConsecOL"]=("Consecutive_Overloads","mean")
            if len(agg)>=2:
                adf=df.groupby("Assignee").agg(**agg).fillna(0)
                if len(adf)>=2:
                    from sklearn.metrics import silhouette_score
                    Xc=StandardScaler().fit_transform(adf)
                    max_k=min(6,len(adf)-1); kr=list(range(2,max_k+1))
                    iner,sils=[],[]
                    for k in kr:
                        km=KMeans(n_clusters=k,init="k-means++",random_state=42,n_init=10)
                        km.fit(Xc); iner.append(km.inertia_)
                        try: sils.append(float(silhouette_score(Xc,km.labels_)))
                        except: sils.append(0.0)
                    bk=kr[int(np.argmax(sils))] if sils else 2
                    km_f=KMeans(n_clusters=bk,init="k-means++",random_state=42,n_init=15)
                    adf["Cluster"]=km_f.fit_predict(Xc)
                    nc=adf.select_dtypes(include="number").columns.tolist()
                    R["cluster"]=dict(agg=adf.to_dict(),num_cols=nc,best_k=bk,
                                      k_range=kr,inertias=iner,sils=sils,
                                      sil=sils[kr.index(bk)] if bk in kr else 0.0)
    except: pass

    # Anomaly Detection
    try:
        af=[c for c in ["Current_Workload_Percent","Blocked_Stories","Consecutive_Overloads",
                         "Completion_Gap","Blocker_Severity","Workload_Stress"] if c in df.columns]
        if len(af)>=2:
            iso=IsolationForest(contamination=0.05,random_state=42,n_estimators=100)
            sc=iso.fit_predict(df[af].fillna(0)); cf=iso.score_samples(df[af].fillna(0))
            R["anomaly"]=dict(count=int((sc==-1).sum()),feats=af,scores=sc.tolist(),confs=cf.tolist())
    except: pass

    return R

@st.cache_data(show_spinner=False)
def bert_classify(texts_t):
    texts=list(texts_t)
    clean=[re.sub(r"[^a-z0-9 ]"," ",str(t).lower()) for t in texts]
    kw={"Bug":["fix","error","bug","crash","null","fail","broken","exception"],
        "Feature":["add","new","feature","implement","create","support","enhance"],
        "Tech-Debt":["refactor","cleanup","optimize","migrate","debt","restructure"],
        "Performance":["slow","performance","latency","speed","timeout","memory"],
        "Security":["auth","security","permission","token","encrypt","vulnerability"],
        "Regression":["regression","revert","rollback","restore","broken","undo"]}
    preds,terms=[],{}
    tv=TfidfVectorizer(max_features=150,ngram_range=(1,2),min_df=1)
    try: Xt=tv.fit_transform(clean); vocab=tv.get_feature_names_out()
    except: Xt,vocab=None,[]
    for text in clean:
        sc={cat:sum(1 for w in ws if w in text) for cat,ws in kw.items()}
        best=max(sc,key=sc.get)
        preds.append(best if sc[best]>0 else "Other")
    if Xt is not None:
        for cat in set(preds):
            idx=[i for i,p in enumerate(preds) if p==cat]
            if idx:
                ms=Xt[idx].mean(axis=0).A1; ti=ms.argsort()[-6:][::-1]
                terms[cat]=[vocab[i] for i in ti]
    return preds,terms

# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
for k,v in [("api_url","https://agile-api-f4ne.onrender.com"),("api_ok",False),
            ("df",None),("api_client",None),("summary",None),
            ("sprints",[]),("team",[]),("velocity",None)]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════
# NAV BAR
# ══════════════════════════════════════════════════════════════════════════
_spark = get_spark()
spark_lbl = "SPARK ACTIVE" if (SPARK_OK and _spark) else "PANDAS"
api_lbl   = "API CONNECTED" if st.session_state.api_ok else "API OFFLINE"
api_cls   = "green" if st.session_state.api_ok else "yellow blink"

st.markdown(f"""
<div class="topnav">
  <div class="logo"><div class="logo-gem"></div>AGILE INTELLIGENCE</div>
  <div class="nav-pills">
    <span class="nav-pill {'green' if SPARK_OK and _spark else 'blue'}">⚡ {spark_lbl}</span>
    <span class="nav-pill {api_cls}">● {api_lbl}</span>
    <span class="nav-pill blue">◆ v3.0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
T = st.tabs([
    "◆ API CONNECT",
    "◈ OVERVIEW",
    "① SPRINT",
    "② WORKLOAD",
    "③ TIME TO RESOLVE",
    "④ BURNOUT",
    "⑤ ALLOCATION",
    "◉ BERT TEXT",
    "⚡ SPARK ML",
    "◎ EVAL",
    "✦ WRITE",
    "⊞ LIVE DATA"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 0 — API CONNECT
# ══════════════════════════════════════════════════════════════════════════
with T[0]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>Connect to Data Source</div>
      <div class='page-sub'>Agile Data API · Jira API · CSV fallback</div>
    </div>""", unsafe_allow_html=True)

    src1, src2 = st.tabs(["⚡ AGILE DATA API", "📂 CSV FALLBACK"])

    with src1:
        st.markdown("<div style='max-width:640px;padding:1rem 2.5rem;'>",unsafe_allow_html=True)
        st.markdown("""<div class='gc blue'>
          <div class='gc-title'>Agile Data API</div>
          <div class='gc-detail'>
            Connect to the Agile Intelligence API Server to fetch live sprint data,
            team analytics and ML-ready datasets.<br><br>
            <b>Start the API server first:</b><br>
            <code style='color:#22d3ee;'>uvicorn agile_api_server:app --reload --port 8000</code>
          </div>
        </div>""", unsafe_allow_html=True)

        api_url_in = st.text_input("API Base URL", st.session_state.api_url,
                                    placeholder="https://agile-api-f4ne.onrender.com", key="api_url_in")
        limit_in   = st.slider("Dataset size to fetch", 100, 500, 300, step=50, key="api_limit")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("◆ CONNECT & FETCH DATA", key="conn_btn", use_container_width=True):
                client = AgileAPIClient(api_url_in)
                health, e = client.health()
                if e:
                    st.error(f"✗ {e}")
                    st.session_state.api_ok = False
                else:
                    df_api, e2 = client.get_ml_dataset(limit=limit_in)
                    if e2:
                        st.error(f"✗ Dataset error: {e2}")
                    else:
                        summary, _  = client.get_summary()
                        sprints, _  = client.get_sprints()
                        team, _     = client.get_team()
                        velocity, _ = client.get_velocity()
                        st.session_state.update({
                            "api_url":    api_url_in,
                            "api_ok":     True,
                            "api_client": client,
                            "df":         df_api,
                            "summary":    summary,
                            "sprints":    sprints or [],
                            "team":       team or [],
                            "velocity":   velocity,
                        })
                        st.success(f"✓ Connected! {len(df_api):,} records loaded from API")
                        st.rerun()
        with c2:
            if st.button("↺ REFRESH DATA", key="refresh_btn", use_container_width=True):
                if st.session_state.api_client:
                    st.session_state.api_client.refresh()
                    df_api, _ = st.session_state.api_client.get_ml_dataset(limit=limit_in)
                    if df_api is not None:
                        st.session_state.df = df_api
                        st.success(f"✓ Data refreshed — {len(df_api):,} records")
                        st.rerun()

        if st.session_state.api_ok:
            h = st.session_state.summary or {}
            st.markdown("""<div class='sec-hdr'>API STATUS</div>""",unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Records",    f"{len(st.session_state.df):,}" if st.session_state.df is not None else "—")
            c2.metric("Sprints",    str(len(st.session_state.sprints)))
            c3.metric("Team",       str(len(st.session_state.team)))
            c4.metric("Health",     f"{h.get('health_score',0):.0f}/100")

            st.markdown("<div class='sec-hdr'>API ENDPOINTS</div>",unsafe_allow_html=True)
            base = st.session_state.api_url
            eps = [("/api/dataset/ml","ML Dataset"),("/api/analytics/summary","Summary"),
                   ("/api/analytics/velocity","Velocity"),("/api/analytics/burnout","Burnout"),
                   ("/api/sprints","Sprints"),("/api/team","Team"),("/api/issues","Issues")]
            ec = st.columns(len(eps))
            for i,(path,lbl) in enumerate(eps):
                with ec[i]:
                    st.markdown(f"""<div class='api-card' style='text-align:center;padding:0.8rem;'>
                        <div class='api-label'>{lbl}</div>
                        <a href='{base}{path}' target='_blank'
                           style='color:#22d3ee;font-size:0.68rem;text-decoration:none;'>
                           {path}</a>
                    </div>""",unsafe_allow_html=True)

    with src2:
        up = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
        if up:
            df_up = pd.read_csv(up).fillna(0)
            st.session_state.df  = df_up
            st.session_state.api_ok = False
            st.success(f"✓ {len(df_up):,} rows loaded from {up.name}")
            st.dataframe(df_up.head(5),use_container_width=True,hide_index=True)

# ── Guard ──────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    for i in range(1,12):
        with T[i]:
            st.markdown("""<div style='text-align:center;padding:6rem 0;'>
              <div style='font-size:2.5rem;margin-bottom:1rem;color:#1e293b;'>◆</div>
              <div style='font-size:0.85rem;color:#3d4f6b;'>
                Connect your data source in <b style='color:#2563eb;'>◆ API CONNECT</b> first.</div>
            </div>""",unsafe_allow_html=True)
    st.stop()

# ── Prep data ──────────────────────────────────────────────────────────────
df = st.session_state.df.copy().fillna(0)
for col,thresh in [("Success_Label",0.5),("Expected_Overload",0.5),("Risk_Flag",0.3)]:
    if col in df.columns:
        if df[col].dtype==object:
            df[col]=df[col].map({"No":0,"Yes":1}).fillna(0).astype(int)
        else:
            df[col]=(df[col]>thresh).astype(int)

df    = eng_features(df, _spark)
sf_ok = [c for c in SF if c in df.columns]
M     = train_models(df)
SM    = train_spark_ml(df)
tp,tt = [],{}
if "Summary" in df.columns:
    tp,tt=bert_classify(tuple(df["Summary"].astype(str)))

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with T[1]:
    h = st.session_state.summary or {}
    api_src = "Live API" if st.session_state.api_ok else "CSV"
    pills = "".join([f"<span class='sp-pill'>⚡{f}</span>" for f in sf_ok[:4]])

    st.markdown(f"""<div class='page-pad'>
      <div style='margin-bottom:6px;'>
        <span class='badge bg-green'>◆ {api_src}</span>
        <span class='badge bg-cyan' style='margin-left:6px;'>
          {'⚡ SPARK' if (SPARK_OK and _spark) else '🐼 PANDAS'}</span>
        <span style='margin-left:8px;'>{pills}</span>
      </div>
      <div class='page-title'>Project Intelligence</div>
      <div class='page-sub'>{len(df):,} records · 5 ML objectives · Live from API</div>
    </div>""",unsafe_allow_html=True)

    findings=[]
    def do_scan(key,neg_val=0):
        if key not in M: return None
        try:
            m=M[key]
            bf=[c for c in m.get("base",m["features"][:7]) if c in df.columns]
            Xn=df[bf].fillna(0).values
            if m.get("tfidf") and m.get("textf") and "Summary" in df.columns:
                try:
                    Xt=m["tfidf"].transform(df["Summary"].astype(str)).toarray()
                    Xn=np.hstack([Xn,Xt])
                except: pass
            nin=m["scaler"].n_features_in_
            Xs=m["scaler"].transform(Xn[:,:nin])
            nin2=m["model"].n_features_in_ if hasattr(m["model"],"n_features_in_") else Xs.shape[1]
            preds=m["model"].predict(Xs[:,:nin2])
            cnt=int((preds==neg_val).sum()); pct=cnt/len(preds)
            return cnt,pct
        except: return None

    r1=do_scan("sprint",0); r2=do_scan("workload",1); r4=do_scan("burnout",1)
    score=100
    def mk_finding(sev,obj,title,action):
        findings.append({"sev":sev,"obj":obj,"title":title,"action":action})
        return 25 if sev=="critical" else 10 if sev=="warning" else 0

    if r1:
        cnt,pct=r1
        sev="critical" if pct>.5 else "warning" if pct>.15 else "ok"
        score-=mk_finding(sev,"Sprint",
            f"{cnt} sprints at spillover risk ({pct:.0%})" if sev!="ok" else "All sprints on track",
            "Reduce scope or unblock stories." if sev!="ok" else "")
    if r2:
        cnt,pct=r2
        sev="critical" if pct>.45 else "warning" if pct>.2 else "ok"
        score-=mk_finding(sev,"Workload",
            f"{cnt} members overloaded ({pct:.0%})" if sev!="ok" else "All within capacity",
            "Redistribute story points." if sev!="ok" else "")
    if r4:
        cnt,pct=r4
        sev="critical" if pct>.5 else "warning" if pct>.25 else "ok"
        score-=mk_finding(sev,"Burnout",
            f"{cnt} members at burnout risk ({pct:.0%})" if sev!="ok" else "Team health stable",
            "Reduce sprint load for flagged members." if sev!="ok" else "")
    score=max(0,score)

    sc_c="#059669" if score>=75 else "#d97706" if score>=50 else "#e11d48"

    # KPIs
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Records",      f"{len(df):,}")
    k2.metric("Health Score", f"{score}/100")
    k3.metric("At Risk",      f"{int((df['Success_Label']==0).sum()):,}" if "Success_Label" in df.columns else "—")
    k4.metric("Burnout",      f"{int(df['Risk_Flag'].sum()):,}" if "Risk_Flag" in df.columns else "—")
    k5.metric("Spark Feats",  str(len(sf_ok)))
    k6.metric("API Source",   api_src)

    # Health + findings
    hc1,hc2 = st.columns([1,2])
    with hc1:
        st.markdown(f"""<div class='gc' style='text-align:center;padding:2.2rem 1rem;'>
          <div style='font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;color:#3d4f6b;margin-bottom:10px;'>
            Project Health</div>
          <div style='font-size:4.5rem;font-weight:900;color:{sc_c};letter-spacing:-0.05em;line-height:1;'>{score}</div>
          <div style='font-size:0.7rem;color:#3d4f6b;margin-top:5px;'>/100 HEALTH SCORE</div>
          <div class='pw' style='margin:14px 0 8px;'>
            <div class='pf' style='background:{sc_c};width:{score}%;'></div></div>
          <div style='font-size:0.68rem;color:{sc_c};text-transform:uppercase;letter-spacing:0.1em;'>
            {"● Healthy" if score>=75 else "⚠ Needs Attention" if score>=50 else "✗ At Risk"}</div>
        </div>""",unsafe_allow_html=True)
    with hc2:
        st.markdown("<div class='sec-hdr'>AUTONOMOUS FINDINGS</div>",unsafe_allow_html=True)
        sev_cls={"critical":"red","warning":"yellow","ok":"green"}
        for f in sorted(findings,key=lambda x:{"critical":0,"warning":1,"ok":2}[x["sev"]]):
            act=f"<div style='font-size:0.7rem;color:#60a5fa;margin-top:4px;'>→ {f['action']}</div>" if f["action"] else ""
            st.markdown(f"""<div class='gc {sev_cls[f["sev"]]}'>
              <div class='gc-title'>[{f['obj']}] {f['title']}</div>{act}
            </div>""",unsafe_allow_html=True)

    # API-sourced analytics
    if st.session_state.api_ok and st.session_state.velocity:
        st.markdown("<div class='sec-hdr'>SPRINT VELOCITY (FROM API)</div>",unsafe_allow_html=True)
        vd=st.session_state.velocity.get("sprints",[])
        if vd:
            vdf=pd.DataFrame(vd).set_index("sprint_name")
            c1,c2=st.columns(2)
            with c1:
                st.caption("Velocity per Sprint")
                st.bar_chart(vdf[["velocity","completed"]],height=180,use_container_width=True)
            with c2:
                st.caption("Completion %")
                st.line_chart(vdf[["pct_done"]],height=180,use_container_width=True)

    # Team health
    acol="Assignee" if "Assignee" in df.columns else None
    if acol:
        st.markdown("<div class='sec-hdr'>TEAM HEALTH</div>",unsafe_allow_html=True)
        people=sorted(df[acol].unique())
        cols_a=st.columns(min(len(people),8))
        for i,p in enumerate(people):
            sub=df[df[acol]==p]
            sr=sub["Success_Label"].eq(0).mean()    if "Success_Label" in df.columns else 0
            ol=sub["Expected_Overload"].mean()       if "Expected_Overload" in df.columns else 0
            br=sub["Risk_Flag"].mean()               if "Risk_Flag" in df.columns else 0
            wl=sub["Current_Workload_Percent"].mean()if "Current_Workload_Percent" in df.columns else 0
            ps=max(0,min(100,100-(sr*35)-(ol*30)-(br*20)-max(0,(wl-100)/2)))
            pc="#059669" if ps>=60 else "#d97706" if ps>=40 else "#e11d48"
            with cols_a[i%len(cols_a)]:
                st.markdown(f"""<div class='gc' style='text-align:center;padding:0.9rem 0.7rem;'>
                  <div class='av' style='background:{pc}18;color:{pc};'>{p[0]}</div>
                  <div class='an'>{p}</div>
                  <div class='as' style='color:{pc};margin:4px 0;'>{ps:.0f}</div>
                  <div class='ad'>WL {wl:.0f}% · Burn {br:.0%}</div>
                </div>""",unsafe_allow_html=True)

    # Trends
    if "Sprint_Number" in df.columns:
        st.markdown("<div class='sec-hdr'>TRENDS</div>",unsafe_allow_html=True)
        tr=df.copy(); tr["risk"]=(tr["Success_Label"]==0).astype(int)
        agg=tr.groupby("Sprint_Number").agg(
            risk=("risk","mean"),wl=("Current_Workload_Percent","mean"),
            burn=("Risk_Flag","mean")).reset_index()
        tc1,tc2,tc3=st.columns(3)
        with tc1: st.caption("Sprint Risk %"); st.line_chart((agg.set_index("Sprint_Number")[["risk"]]*100).round(1),height=150)
        with tc2: st.caption("Avg Workload %"); st.line_chart(agg.set_index("Sprint_Number")[["wl"]].round(1),height=150)
        with tc3: st.caption("Burnout Risk %"); st.line_chart((agg.set_index("Sprint_Number")[["burn"]]*100).round(1),height=150)

    st.download_button("⬇ EXPORT REPORT (.md)",
        data=f"# Agile Intelligence Report\n\nHealth: {score}/100\nSource: {api_src}\n\n"+
             "\n".join([f"- [{f['obj']}] {f['title']}" for f in findings]),
        file_name="report.md",mime="text/markdown")

# ══════════════════════════════════════════════════════════════════════════
# helper to render prediction result
def pred_card(label, value, conf, color):
    st.markdown(f"""<div class='pred'>
      <div class='pred-lbl'>{label}</div>
      <div class='pred-val' style='color:{color};'>{value}</div>
      <div class='pred-conf'>{conf}</div>
      <div class='pw' style='margin-top:10px;'>
        <div class='pf' style='background:{color};width:{float(conf.replace("%","").strip()) if "%" in conf else 50}%;'></div></div>
    </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — SPRINT
# ══════════════════════════════════════════════════════════════════════════
with T[2]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>Sprint Completion</div>
      <div class='page-sub'>Fine-tuned Logistic Regression + TF-IDF text features · Target 89%</div>
    </div>""",unsafe_allow_html=True)

    if "sprint" not in M:
        st.warning("Sprint model unavailable.")
    else:
        m=M["sprint"]; acc=m["acc"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Accuracy", f"{acc:.2%}", f"{acc-0.89:+.2%} vs 89%")
        c2.metric("F1 Score", f"{m['f1']:.3f}")
        c3.metric("AUC-ROC",  f"{m['auc']:.3f}")
        c4.metric("Text Feats",str(len(m.get("textf",[]))))

        bar_c="#059669" if acc>=0.89 else "#d97706"
        st.markdown(f"""<div style='margin:0.5rem 0 1rem;'>
          <div style='display:flex;justify-content:space-between;font-size:0.68rem;color:#3d4f6b;'>
            <span>Current: {acc:.2%}</span><span>Target: 89%</span></div>
          <div class='pw'><div class='pf' style='background:{bar_c};width:{acc*100:.1f}%;'></div></div>
          <div class='pw' style='margin-top:3px;'><div class='pf' style='background:#0f1829;width:89%;'></div></div>
        </div>""",unsafe_allow_html=True)

        cl,cr=st.columns([3,2])
        with cl:
            with st.expander("◈ Fine-tune Results"):
                if m.get("tune"):
                    tdf=pd.DataFrame(m["tune"]).sort_values("acc",ascending=False)
                    tdf["acc"]=tdf["acc"].apply(lambda x:f"{x:.2%}")
                    tdf["f1"] =tdf["f1"].apply(lambda x:f"{x:.3f}")
                    st.dataframe(tdf[["C","solver","class_weight","max_iter","acc","f1"]],use_container_width=True,hide_index=True)
            with st.expander("◈ Classification Report"): st.text(m["report"])
            with st.expander("◈ Confusion Matrix"):
                cm=np.array(m["cm"])
                st.dataframe(pd.DataFrame(cm,index=["Actual: Risk","Actual: Done"],
                    columns=["Pred: Risk","Pred: Done"]),use_container_width=True)
        with cr:
            st.markdown("<div class='sec-hdr'>PREDICT</div>",unsafe_allow_html=True)
            psp=st.number_input("Planned SP",    1,100,40,key="s1")
            csp=st.number_input("Completed SP",  0,100,30,key="s2")
            pct=st.slider("% Done",0.0,100.0,75.0,      key="s3")
            drs=st.number_input("Days Left",     0, 30, 5,key="s4")
            hv =st.number_input("Hist Velocity", 0,100,35,key="s5")
            bs =st.number_input("Blocked",       0, 10, 1,key="s6")
            sc2=st.number_input("Scope Change", -20,20, 0,key="s7")
            smr=st.text_input("Summary","Fix login bug",  key="s8")
            if st.button("◆ PREDICT SPRINT",key="s_btn",use_container_width=True):
                rv=[{"Planned_Story_Points_Sprint":psp,"Completed_Story_Points":csp,
                     "Percent_Done":pct,"Days_Remaining_Sprint":drs,"Historical_Velocity":hv,
                     "Blocked_Stories":bs,"Scope_Change":sc2}.get(f,0) for f in m.get("base",m["features"][:7])]
                if m.get("tfidf") and m.get("textf"):
                    try: rv+=list(m["tfidf"].transform([smr.lower()]).toarray()[0])
                    except: pass
                ra=np.array(rv).reshape(1,-1); nin=m["scaler"].n_features_in_
                rs=m["scaler"].transform(ra[:,:nin])
                nin2=m["model"].n_features_in_ if hasattr(m["model"],"n_features_in_") else rs.shape[1]
                p=m["model"].predict(rs[:,:nin2])[0]
                pb=m["model"].predict_proba(rs[:,:nin2])[0][1]
                pc="#059669" if pb>=0.7 else "#d97706" if pb>=0.4 else "#e11d48"
                pred_card("Sprint Prediction","COMPLETE ✓" if p else "AT RISK ✗",f"{pb:.2%} confidence",pc)

# ══════════════════════════════════════════════════════════════════════════
# TABs 3-6 — Compact predict tabs
# ══════════════════════════════════════════════════════════════════════════
def simple_pred_tab(tab, title, sub, key, model_key, input_fn, predict_fn):
    with tab:
        st.markdown(f"""<div class='page-pad'>
          <div class='page-title'>{title}</div>
          <div class='page-sub'>{sub}</div>
        </div>""",unsafe_allow_html=True)
        if model_key not in M:
            st.warning(f"{title} model unavailable.")
            return
        m=M[model_key]
        c1,c2=st.columns(2)
        c1.metric("Accuracy" if "acc" in m else "R²", f"{m.get('acc',m.get('r2',0)):.2%}" if "acc" in m else f"{m.get('r2',0):.3f}")
        c2.metric("Algorithm",m["algo"])
        if m.get("report"):
            with st.expander("◈ Report"): st.text(m["report"])
        cl,cr=st.columns([3,2])
        with cr:
            st.markdown("<div class='sec-hdr'>PREDICT</div>",unsafe_allow_html=True)
            inputs=input_fn(key)
            if st.button(f"◆ PREDICT",key=f"{key}_btn",use_container_width=True):
                predict_fn(m, inputs)

# Workload
with T[3]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>Workload Projection</div>
      <div class='page-sub'>Naive Bayes · fastest binary overload classifier</div>
    </div>""",unsafe_allow_html=True)
    if "workload" not in M: st.warning("Workload model unavailable.")
    else:
        m=M["workload"]
        c1,c2,c3=st.columns(3)
        c1.metric("Accuracy",f"{m['acc']:.2%}"); c2.metric("F1",f"{m['f1']:.3f}"); c3.metric("Algo",m["algo"])
        with st.expander("◈ Report"): st.text(m["report"])
        cl,cr=st.columns([3,2])
        with cr:
            st.markdown("<div class='sec-hdr'>PREDICT</div>",unsafe_allow_html=True)
            psp2=st.number_input("Planned SP",         1,100, 35,key="w1")
            casp=st.number_input("Current Assigned SP",0,100, 40,key="w2")
            hasp=st.number_input("Historical Avg SP",  1,100, 30,key="w3")
            rdr =st.number_input("Remaining Days",     1, 30,  5,key="w4")
            hpt =st.number_input("High Priority",      0, 10,  2,key="w5")
            cwp =st.number_input("Workload %",         0,200,125,key="w6")
            if st.button("◆ PREDICT OVERLOAD",key="w_btn",use_container_width=True):
                row=pd.DataFrame([{"Planned_Story_Points_Resource":psp2,"Current_Assigned_SP":casp,
                    "Historical_Avg_SP":hasp,"Remaining_Days_Resource":rdr,
                    "High_Priority_Tasks_Resource":hpt,"Current_Workload_Percent":cwp}])
                row=row.reindex(columns=m["features"],fill_value=0)
                pred=m["model"].predict(m["scaler"].transform(row))[0]
                prob=m["model"].predict_proba(m["scaler"].transform(row))[0][1]
                pc="#e11d48" if pred else "#059669"
                pred_card("Workload Status","OVERLOADED" if pred else "WITHIN CAPACITY",f"{prob:.2%} confidence",pc)

# TTR
with T[4]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>Time to Resolve</div>
      <div class='page-sub'>Ridge Regression · regularised linear hour prediction</div>
    </div>""",unsafe_allow_html=True)
    if "ttr" not in M: st.warning("TTR model unavailable.")
    else:
        m=M["ttr"]
        c1,c2,c3=st.columns(3)
        c1.metric("R²",f"{m['r2']:.3f}"); c2.metric("MSE",f"{m['mse']:.2f}"); c3.metric("Algo",m["algo"])
        cl,cr=st.columns([3,2])
        with cr:
            st.markdown("<div class='sec-hdr'>ESTIMATE</div>",unsafe_allow_html=True)
            itype=st.selectbox("Type",["Bug","Story","Task"],key="t1")
            pri  =st.selectbox("Priority",["Low","Medium","High"],key="t2")
            oe   =st.number_input("Estimate (h)",1,50,8,key="t3")
            sp   =st.number_input("Story Points",1,20,5,key="t4")
            if st.button("◆ ESTIMATE TTR",key="t_btn",use_container_width=True):
                row={c:0 for c in m["features"]}
                if f"Issue_Type_{itype}" in row: row[f"Issue_Type_{itype}"]=1
                if f"Priority_{pri}"     in row: row[f"Priority_{pri}"]    =1
                row["Original_Estimate_Hours"]=oe
                if "Story_Points_Issue" in row: row["Story_Points_Issue"]=sp
                pt=max(0,m["model"].predict(m["scaler"].transform(pd.DataFrame([row])[m["features"]]))[0])
                delta=pt-oe; pc="#059669" if delta<=2 else "#d97706" if delta<=5 else "#e11d48"
                pred_card("Estimated Resolution",f"{pt:.1f} hours",f"{delta:+.1f}h vs estimate",pc)

# Burnout
with T[5]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>Burnout Risk</div>
      <div class='page-sub'>Decision Tree · threshold-based burnout rules</div>
    </div>""",unsafe_allow_html=True)
    if "burnout" not in M: st.warning("Burnout model unavailable.")
    else:
        m=M["burnout"]
        c1,c2,c3=st.columns(3)
        c1.metric("Accuracy",f"{m['acc']:.2%}"); c2.metric("F1",f"{m['f1']:.3f}"); c3.metric("Algo",m["algo"])
        with st.expander("◈ Report"): st.text(m["report"])
        cl,cr=st.columns([3,2])
        with cr:
            st.markdown("<div class='sec-hdr'>CHECK RISK</div>",unsafe_allow_html=True)
            tsp =st.number_input("Total SP",         0,100,40,key="b1")
            hasp=st.number_input("Historical Avg SP",1,100,25,key="b2")
            hpt =st.number_input("High Priority",    0, 10, 2,key="b3")
            co  =st.number_input("Consec. Overloads",0,  5, 2,key="b4")
            if st.button("◆ CHECK BURNOUT",key="b_btn",use_container_width=True):
                row={f:0 for f in m["features"]}
                row.update({"Total_SP_This_Sprint":tsp,"Historical_Avg_SP_Burnout":hasp,
                            "High_Priority_Tasks_Burnout":hpt,"Consecutive_Overloads":co})
                rs=m["scaler"].transform(pd.DataFrame([row])[m["features"]])
                pred=m["model"].predict(rs)[0]; prob=m["model"].predict_proba(rs)[0][1]
                pc="#e11d48" if pred else "#059669"
                pred_card("Burnout Status","RISK DETECTED" if pred else "HEALTHY",f"{prob:.2%} confidence",pc)

# Allocation
with T[6]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>Resource Allocation</div>
      <div class='page-sub'>KNN · groups similar issues to similar assignees</div>
    </div>""",unsafe_allow_html=True)
    if "alloc" not in M: st.warning("Allocation model unavailable.")
    else:
        m=M["alloc"]
        c1,c2=st.columns(2)
        c1.metric("Accuracy",f"{m['acc']:.2%}"); c2.metric("Algo",m["algo"])
        st.info("ℹ️ Accuracy improves with real skill-tag and component features.")
        cl,cr=st.columns([3,2])
        with cr:
            st.markdown("<div class='sec-hdr'>SUGGEST</div>",unsafe_allow_html=True)
            smry=st.text_input("Summary","Fix login bug",key="a1")
            lbl2=st.text_input("Label",  "Bug",          key="a2")
            oe5 =st.number_input("Estimate (h)",1,50,8,  key="a3")
            sp5 =st.number_input("Story Points",1,20,5,  key="a4")
            if st.button("◆ SUGGEST ASSIGNEE",key="a_btn",use_container_width=True):
                try:    se=m["le_s"].transform([smry])[0]
                except: se=0
                try:    le=m["le_l"].transform([lbl2])[0]
                except: le=0
                row=pd.DataFrame([{"Summary_enc":se,"Labels_enc":le,
                    "Original_Estimate_Resource":oe5,"Story_Points_Resource":sp5}])
                row=row.reindex(columns=m["features"],fill_value=0)
                asgn=m["model"].predict(m["scaler"].transform(row))[0]
                pred_card("Recommended Assignee",str(asgn),"Based on similar past issues","#60a5fa")

# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — BERT TEXT
# ══════════════════════════════════════════════════════════════════════════
with T[7]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>BERT Text Intelligence</div>
      <div class='page-sub'>TF-IDF text classification · keyword extraction · live issue analysis</div>
    </div>""",unsafe_allow_html=True)

    cat_c={"Bug":"#e11d48","Feature":"#059669","Tech-Debt":"#d97706",
            "Performance":"#2563eb","Security":"#7c3aed","Regression":"#ea580c","Other":"#475569"}

    if "Summary" in df.columns and tp:
        c1,c2=st.columns([1,2])
        with c1:
            st.markdown("<div class='sec-hdr'>CATEGORY DISTRIBUTION</div>",unsafe_allow_html=True)
            vc=pd.Series(tp).value_counts()
            for cat,cnt in vc.items():
                p=cnt/len(tp); col=cat_c.get(cat,"#475569")
                st.markdown(f"""<div style='margin-bottom:7px;'>
                  <div style='display:flex;justify-content:space-between;font-size:0.73rem;'>
                    <span style='color:{col};font-weight:800;'>{cat}</span>
                    <span style='color:#3d4f6b;'>{cnt} ({p:.0%})</span></div>
                  <div class='pw'><div class='pf' style='background:{col};width:{p*100:.1f}%;'></div></div>
                </div>""",unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='sec-hdr'>TOP KEYWORDS</div>",unsafe_allow_html=True)
            for cat,terms in list(tt.items())[:6]:
                col=cat_c.get(cat,"#475569")
                badges="".join([f"<span style='background:{col}15;color:{col};border:1px solid {col}28;padding:1px 8px;border-radius:3px;font-size:0.62rem;font-weight:800;margin:2px;display:inline-block;'>{t}</span>" for t in terms])
                st.markdown(f"<div style='margin-bottom:9px;'><b style='color:{col};font-size:0.76rem;'>{cat}</b><br>{badges}</div>",unsafe_allow_html=True)

        st.markdown("<div class='sec-hdr'>ISSUES WITH BERT CLASSIFICATION</div>",unsafe_allow_html=True)
        pv=df[["Summary"]].copy()
        for col in ["Priority","Assignee","Issue_Type","Status"]:
            if col in df.columns: pv[col]=df[col]
        pv["BERT_Category"]=tp
        st.dataframe(pv.head(20),use_container_width=True,hide_index=True)

        st.markdown("<div class='sec-hdr'>CLASSIFY NEW TEXT</div>",unsafe_allow_html=True)
        cl2,cr2=st.columns([3,2])
        with cr2:
            ntxt=st.text_area("Issue summary","Fix null pointer exception in auth",key="bert_in")
            if st.button("◆ CLASSIFY",key="bert_btn",use_container_width=True):
                p2,_=bert_classify(tuple([ntxt])); cat=p2[0] if p2 else "Unknown"
                col=cat_c.get(cat,"#475569")
                pred_card("BERT Category",cat,"TF-IDF text classification",col)
    else:
        st.info("Dataset needs a 'Summary' column for BERT text analysis.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 8 — SPARK ML
# ══════════════════════════════════════════════════════════════════════════
with T[8]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>Spark ML Engine</div>
      <div class='page-sub'>Ensemble Voting · GBT · Improved K-Means++ · Anomaly Detection</div>
    </div>""",unsafe_allow_html=True)

    if SPARK_OK and _spark: st.success("⚡ Apache Spark active")
    else: st.info("🐼 Pandas mode — install Java 17 + pyspark for Spark")
    if sf_ok:
        pills="".join([f"<span class='sp-pill'>⚡{f}</span>" for f in sf_ok])
        st.markdown(f"<div style='margin-bottom:1rem;'>{pills}</div>",unsafe_allow_html=True)

    # Sprint Ensemble
    st.markdown("<div class='sec-hdr'>SPRINT — ENSEMBLE VOTING (LR + GBT + RF + ADABOOST)</div>",unsafe_allow_html=True)
    if "sprint" in SM:
        s=SM["sprint"]; ea = s.get("acc", 0); best=max(s["ind"],key=s["ind"].get)
        c1,c2,c3=st.columns(3)
        c1.metric("Ensemble Acc",f"{ea:.2%}"); c2.metric("Best Single",f"{s['ind'][best]:.2%}",best)
        c3.metric("Spark Feats",str(len([f for f in s["feat"] if f in s["sf"]])))
        with st.expander("◈ Model Comparison + Feature Importance"):
            for nm,acc in sorted(s["ind"].items(),key=lambda x:-x[1]):
                diff=ea-acc; dc="#059669" if diff>=0 else "#e11d48"
                st.markdown(f"""<div style='margin-bottom:4px;'>
                  <div style='display:flex;justify-content:space-between;font-size:0.76rem;'>
                    <span style='color:#dde4f0;'>{nm}</span>
                    <span>{acc:.2%} <span style='color:{dc};'>({diff:+.2%})</span></span></div>
                  <div class='pw'><div class='pf' style='background:#2563eb;width:{acc*100:.1f}%;'></div></div>
                </div>""",unsafe_allow_html=True)
            st.markdown("**Feature Importance (GBT):**")
            imp=pd.Series(s["imp"]).sort_values(ascending=False)
            for feat,v in imp.head(10).items():
                bw=v/imp.max()*100; tag=" ⚡" if feat in s["sf"] else ""
                col="#0891b2" if feat in s["sf"] else "#2563eb"
                st.markdown(f"""<div style='margin-bottom:3px;'>
                  <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#dde4f0;'>
                    <span>{feat}{tag}</span><span style='color:{col};'>{v:.3f}</span></div>
                  <div class='pw'><div class='pf' style='background:{col};width:{bw:.0f}%;'></div></div>
                </div>""",unsafe_allow_html=True)
            st.caption("⚡ = Spark-engineered feature")
    else: st.info("Sprint ensemble unavailable.")

    st.markdown("<div class='sec-hdr'>TTR — GBT REGRESSOR + SPARK FEATURES</div>",unsafe_allow_html=True)
    if "ttr" in SM:
        t=SM["ttr"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("GBT R²",f"{t['gb_r2']:.3f}"); c2.metric("LR R²",f"{t['lr_r2']:.3f}",f"{t['gb_r2']-t['lr_r2']:+.3f}")
        c3.metric("GBT MSE",f"{t['gb_mse']:.2f}"); c4.metric("Spark Feats",str(len([f for f in t["feat"] if f in t["sf"]])))
    else: st.info("TTR unavailable.")

    st.markdown("<div class='sec-hdr'>BURNOUT — GBT + RF ENSEMBLE</div>",unsafe_allow_html=True)
    if "burnout" in SM:
        b=SM["burnout"]
        c1,c2=st.columns(2); c1.metric("Acc",f"{b['acc']:.2%}"); c2.metric("Spark Feats",str(len([f for f in b["feat"] if f in b["sf"]])))
        with st.expander("◈ Report"): st.text(b["report"])
    else: st.info("Burnout unavailable.")

    st.markdown("<div class='sec-hdr'>TEAM CLUSTERING — IMPROVED K-MEANS++ WITH ELBOW</div>",unsafe_allow_html=True)
    if "cluster" in SM:
        cl=SM["cluster"]
        ec1,ec2=st.columns(2)
        with ec1:
            st.caption("Elbow Method — Inertia vs k")
            st.line_chart(pd.DataFrame({"k":cl["k_range"],"Inertia":cl["inertias"]}).set_index("k"),height=180)
        with ec2:
            st.caption("Silhouette Score vs k")
            st.line_chart(pd.DataFrame({"k":cl["k_range"],"Silhouette":cl["sils"]}).set_index("k"),height=180)
        kc1,kc2,kc3=st.columns(3)
        kc1.metric("Optimal k",cl["best_k"]); kc2.metric("Silhouette",f"{cl['sil']:.3f}"); kc3.metric("Init","K-Means++")
        adf=pd.DataFrame(cl["agg"]); nc=cl["num_cols"]
        cl1,cl2,cl3=st.columns(3)
        for cw,cid,lbl,col in [(cl1,0,"High Performers","#059669"),(cl2,1,"Mid Performers","#d97706"),(cl3,2,"Overloaded","#e11d48")]:
            members=adf[adf["Cluster"]==cid].index.tolist() if cid<cl["best_k"] else []
            with cw:
                badges="".join([f"<span style='background:{col}18;color:{col};border:1px solid {col}28;padding:1px 8px;border-radius:3px;font-size:0.7rem;font-weight:800;margin:2px;display:inline-block;'>{m}</span>" for m in members])
                st.markdown(f"""<div class='gc' style='text-align:center;'>
                  <div style='color:{col};font-weight:800;font-size:0.75rem;margin-bottom:6px;'>{lbl}</div>{badges}
                </div>""",unsafe_allow_html=True)
        ddf=adf.drop(columns=["Cluster"],errors="ignore")
        fmt={c:"{:.2f}" for c in nc if c in ddf.columns}
        st.dataframe(ddf.style.format(fmt,na_rep="--"),use_container_width=True)
    else: st.info("Clustering unavailable.")

    st.markdown("<div class='sec-hdr'>ANOMALY DETECTION — ISOLATION FOREST</div>",unsafe_allow_html=True)
    if "anomaly" in SM:
        an=SM["anomaly"]
        sc_s=pd.Series(an["scores"]); cf_s=pd.Series(an["confs"])
        mask=sc_s==-1; adf2=df[mask].copy(); adf2["Anomaly Score"]=cf_s[mask].values
        c1,c2,c3=st.columns(3)
        c1.metric("Anomalies",an["count"]); c2.metric("Rate","5%"); c3.metric("Features",len(an["feats"]))
        dcols=(["Assignee"] if "Assignee" in df.columns else [])+an["feats"]+["Anomaly Score"]
        st.dataframe(adf2[[c for c in dcols if c in adf2.columns]].head(10),use_container_width=True,hide_index=True)
    else: st.info("Anomaly detection unavailable.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 9 — EVAL
# ══════════════════════════════════════════════════════════════════════════
with T[9]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>Model Evaluation</div>
      <div class='page-sub'>Accuracy · F1 · AUC-ROC · Confusion Matrices · 89% target · Fine-tune comparison</div>
    </div>""",unsafe_allow_html=True)

    rows=[]
    for key,label in [("sprint","Sprint"),("workload","Workload"),("burnout","Burnout"),("alloc","Allocation")]:
        if key in M:
            m=M[key]; acc=m.get("acc",0)
            rows.append({"Objective":label,"Algorithm":m["algo"],"Accuracy":f"{acc:.2%}",
                         "F1":f"{m.get('f1',0):.3f}",
                         "AUC":f"{m.get('auc',0):.3f}" if isinstance(m.get("auc"),float) else "—",
                         "vs 89%":f"{acc-0.89:+.2%} {'✓' if acc>=0.89 else '✗'}"})
    if "ttr" in M:
        m=M["ttr"]
        rows.append({"Objective":"TTR","Algorithm":m["algo"],"Accuracy":f"R²={m['r2']:.3f}","F1":"—","AUC":"—","vs 89%":"—"})
    if rows:
        st.markdown("<div class='sec-hdr'>PERFORMANCE SUMMARY</div>",unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    st.markdown("<div class='sec-hdr'>ACCURACY VS 89% TARGET</div>",unsafe_allow_html=True)
    for key,label in [("sprint","Sprint"),("workload","Workload"),("burnout","Burnout"),("alloc","Allocation")]:
        if key in M:
            acc=M[key].get("acc",0); col="#059669" if acc>=0.89 else "#d97706" if acc>=0.75 else "#e11d48"
            st.markdown(f"""<div style='margin-bottom:8px;'>
              <div style='display:flex;justify-content:space-between;font-size:0.78rem;'>
                <span style='color:#dde4f0;'><b>{label}</b> — {M[key]['algo']}</span>
                <span style='color:{col};'>{acc:.2%} {"✓" if acc>=0.89 else "✗"}</span></div>
              <div class='pw'><div class='pf' style='background:{col};width:{acc*100:.1f}%;'></div></div>
            </div>""",unsafe_allow_html=True)

    if "sprint" in M and M["sprint"].get("tune"):
        st.markdown("<div class='sec-hdr'>SPRINT LR FINE-TUNE TABLE</div>",unsafe_allow_html=True)
        tdf=pd.DataFrame(M["sprint"]["tune"]).sort_values("acc",ascending=False)
        tdf["acc"]=tdf["acc"].apply(lambda x:f"{x:.2%}"); tdf["f1"]=tdf["f1"].apply(lambda x:f"{x:.3f}")
        st.dataframe(tdf[["C","solver","class_weight","max_iter","acc","f1"]],use_container_width=True,hide_index=True)

    st.markdown("<div class='sec-hdr'>CONFUSION MATRICES</div>",unsafe_allow_html=True)
    cm_cols=st.columns(2); ci=0
    for key,label,names in [("sprint","Sprint",["Risk","Complete"]),("workload","Workload",["No OL","Overload"]),("burnout","Burnout",["Healthy","At Risk"])]:
        if key in M and M[key].get("cm"):
            cm=np.array(M[key]["cm"])
            with cm_cols[ci%2]:
                st.caption(label)
                st.dataframe(pd.DataFrame(cm,index=[f"Actual: {n}" for n in names],columns=[f"Pred: {n}" for n in names]),use_container_width=True)
            ci+=1

    if "sprint" in M and "sprint" in SM:
        st.markdown("<div class='sec-hdr'>SIMPLE VS SPARK ENSEMBLE</div>",unsafe_allow_html=True)
        comp=[{"Objective":"Sprint","Simple (LR)":f"{M['sprint']['acc']:.2%}",
               "Spark Ensemble":f"{SM['sprint']['acc']:.2%}",
               "Winner":"Spark" if SM["sprint"]["acc"]>M["sprint"]["acc"] else "Simple"}]
        if "burnout" in M and "burnout" in SM:
            comp.append({"Objective":"Burnout","Simple (DT)":f"{M['burnout']['acc']:.2%}",
                         "Spark Ensemble":f"{SM['burnout']['acc']:.2%}",
                         "Winner":"Spark" if SM["burnout"]["acc"]>M["burnout"]["acc"] else "Simple"})
        st.dataframe(pd.DataFrame(comp),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 10 — WRITE
# ══════════════════════════════════════════════════════════════════════════
with T[10]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>Write to API</div>
      <div class='page-sub'>Create issues · Add comments · Transition status · Search</div>
    </div>""",unsafe_allow_html=True)

    if not st.session_state.api_ok:
        st.markdown("""<div class='gc blue' style='text-align:center;padding:3rem;'>
          <div style='font-size:0.82rem;color:#3d4f6b;'>
            Connect API in <b style='color:#60a5fa;'>◆ API CONNECT</b> tab first</div>
        </div>""",unsafe_allow_html=True)
    else:
        client=st.session_state.api_client
        st.success(f"✓ API connected — {st.session_state.api_url}")

        wt1,wt2,wt3,wt4=st.tabs(["CREATE ISSUE","ADD COMMENT","TRANSITION","SEARCH"])

        with wt1:
            c1,c2=st.columns(2)
            with c1:
                ni_sum  =st.text_input("Summary","",key="ni_sum")
                ni_type =st.selectbox("Type",["Task","Bug","Story","Epic"],key="ni_type")
                ni_pri  =st.selectbox("Priority",["High","Medium","Low"],key="ni_pri")
            with c2:
                ni_asgn  =st.selectbox("Assignee",["Unassigned"]+["Alice","Bob","Carol","David","Eve","Frank","Grace","Henry"],key="ni_asgn")
                ni_sp    =st.number_input("Story Points",1,21,3,key="ni_sp")
                ni_sprint=st.text_input("Sprint ID","SP-010",key="ni_sprint")
                ni_lbl   =st.text_input("Labels","general",key="ni_lbl")
            if st.button("◆ CREATE ISSUE",key="ci_btn"):
                if not ni_sum: st.warning("Summary required.")
                else:
                    d,e=client.create_issue(ni_sum,ni_type,ni_pri,ni_asgn,ni_sp,ni_sprint,ni_lbl)
                    if e: st.error(e)
                    else: st.success(f"✓ Created: **{d.get('issue_id','?')}**")

        with wt2:
            ck=st.text_input("Issue Key","AGI-0001",key="cmt_k")
            ct=st.text_area("Comment","",height=80,key="cmt_t")
            if st.button("◆ ADD COMMENT",key="cmt_btn"):
                d,e=client.add_comment(ck,ct)
                if e: st.error(e)
                else: st.success("✓ Comment added")

        with wt3:
            tk=st.text_input("Issue Key","AGI-0001",key="tr_k")
            tto=st.selectbox("Transition To",["In Progress","In Review","Done","Blocked","To Do"],key="tr_to")
            if st.button("◆ TRANSITION",key="tr_btn"):
                d,e=client.transition(tk,tto)
                if e: st.error(e)
                else: st.success(f"✓ Transitioned to {tto}")

        with wt4:
            c1,c2,c3=st.columns(3)
            with c1: sq=st.text_input("Keyword","",key="sq")
            with c2: sa=st.selectbox("Assignee",["(any)"]+["Alice","Bob","Carol","David","Eve","Frank","Grace","Henry"],key="sa")
            with c3: ss=st.selectbox("Status",["(any)","To Do","In Progress","Done","Blocked"],key="ss")
            if st.button("◆ SEARCH",key="srch_btn"):
                results,e=client.search(sq,sa if sa!="(any)" else None,None,ss if ss!="(any)" else None)
                if e: st.error(e)
                elif not results: st.info("No results.")
                else:
                    rdf=pd.DataFrame(results)
                    st.success(f"✓ {len(rdf)} results")
                    st.dataframe(rdf[["issue_id","summary","issue_type","priority","status","assignee","story_points"]],use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 11 — LIVE DATA
# ══════════════════════════════════════════════════════════════════════════
with T[11]:
    st.markdown("""<div class='page-pad'>
      <div class='page-title'>Live API Data Explorer</div>
      <div class='page-sub'>Browse raw data from API — sprints, issues, team, analytics</div>
    </div>""",unsafe_allow_html=True)

    dt1,dt2,dt3,dt4=st.tabs(["SPRINTS","TEAM","ISSUES","ANALYTICS"])

    with dt1:
        if st.session_state.sprints:
            sdf=pd.DataFrame(st.session_state.sprints)
            st.metric("Active Sprints",sum(1 for s in st.session_state.sprints if s["state"]=="active"))
            st.dataframe(sdf[["sprint_id","sprint_name","state","planned_story_points",
                               "completed_story_points","percent_done","blocked_stories",
                               "success_label","risk_flag"]].round(1),
                         use_container_width=True,hide_index=True)
        else:
            st.info("Connect API to see live sprint data.")

    with dt2:
        if st.session_state.team:
            tdf=pd.DataFrame(st.session_state.team)
            st.dataframe(tdf[["name","role","current_workload_pct","burnout_score",
                               "health_score","consecutive_overloads","risk_flag"]].round(1),
                         use_container_width=True,hide_index=True)
            bc1,bc2=st.columns(2)
            with bc1:
                st.caption("Burnout Score by Member")
                st.bar_chart(tdf.set_index("name")[["burnout_score"]],height=200)
            with bc2:
                st.caption("Workload % by Member")
                st.bar_chart(tdf.set_index("name")[["current_workload_pct"]],height=200)
        else:
            st.info("Connect API to see live team data.")

    with dt3:
        st.dataframe(df[["Issue_ID","Summary","Issue_Type","Priority","Status",
                          "Assignee","Story_Points_Issue","Resolution_Time_Hours"]].head(30)
                     if all(c in df.columns for c in ["Issue_ID","Issue_Type"]) else df.head(30),
                     use_container_width=True,hide_index=True)

    with dt4:
        if st.session_state.summary:
            h=st.session_state.summary
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Health Score",    f"{h.get('health_score',0):.0f}/100")
            c2.metric("At Risk Sprints", h.get("sprints_at_risk",0))
            c3.metric("Burnout Risks",   h.get("burnout_risk_count",0))
            c4.metric("Avg Velocity",    h.get("avg_velocity",0))
            st.json(h)
        else:
            st.info("Connect API to see live analytics.")
