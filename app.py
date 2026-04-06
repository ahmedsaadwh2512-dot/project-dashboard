import io
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Primavera XER Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.image("profile.jpg", width=130)
st.sidebar.markdown("### 👷 Project Controls Dashboard")
st.sidebar.caption("by Ahmed Saad")
# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
COLORS = {
    "bg":           "#f6f8fb",
    "panel":        "#ffffff",
    "panel_soft":   "#fbfdff",
    "line":         "#e7edf5",
    "text":         "#0f172a",
    "muted":        "#64748b",
    "primary":      "#18b7a0",
    "primary_dark": "#0f766e",
    "good":         "#10b981",
    "warn":         "#f59e0b",
    "bad":          "#ef4444",
    "navy":         "#10233f",
    "gray":         "#cbd5e1",
}

st.markdown(
    f"""
    <style>
        [data-testid="stHeader"], [data-testid="stToolbar"], .stDeployButton {{display:none !important;}}
        .stApp {{background:{COLORS['bg']}; color:{COLORS['text']};}}
        .block-container {{max-width: 1600px; padding-top: 0.8rem; padding-bottom: 1.5rem;}}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border-right: 1px solid {COLORS['line']};
        }}
        .hero {{
            background: linear-gradient(135deg, #ffffff 0%, #f6fffd 50%, #eefaf7 100%);
            border: 1px solid {COLORS['line']};
            border-top: 4px solid {COLORS['primary']};
            border-radius: 24px;
            padding: 22px 24px 18px 24px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            margin-bottom: 14px;
        }}
        .hero-title   {{font-size: 16px; color:{COLORS['muted']}; font-weight:600;}}
        .hero-project {{font-size: 26px; font-weight:900; margin: 4px 0 2px 0; color:{COLORS['text']};}}
        .hero-sub     {{font-size: 13px; color:{COLORS['muted']};}}
        .kpi {{
            background:{COLORS['panel']};
            border:1px solid {COLORS['line']};
            border-top:4px solid {COLORS['primary']};
            border-radius:20px;
            padding:14px 16px;
            min-height: 118px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
        }}
        .kpi-label {{font-size:11px; text-transform:uppercase; letter-spacing:.14em; color:{COLORS['muted']}; font-weight:800;}}
        .kpi-value {{font-size:18px; font-weight:900; color:{COLORS['text']}; margin-top:8px; line-height:1.15;}}
        .kpi-sub   {{font-size:12px; color:{COLORS['muted']}; margin-top:4px;}}
        .tag       {{display:inline-block; padding: 4px 10px; border-radius:999px; font-size:11px; font-weight:800; margin-top:8px;}}
        .chart-card {{
            background:{COLORS['panel']};
            border:1px solid {COLORS['line']};
            border-radius: 22px;
            padding: 6px 12px 0 12px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
            margin-bottom: 12px;
        }}
        .section-head {{font-size: 15px; font-weight: 900; letter-spacing:.1em; text-transform:uppercase; color:{COLORS['muted']};}}
        .banner {{padding: 12px 16px; border-radius: 16px; border:1px solid; font-weight:700; margin-bottom:10px;}}
        .small-note {{font-size:12px; color:{COLORS['muted']};}}
        .stTabs [data-baseweb="tab-list"] {{gap: 10px;}}
        .stTabs [data-baseweb="tab"] {{
            background:#f8fafc; border-radius: 14px;
            padding: 10px 18px; border:1px solid {COLORS['line']}; height:46px;
        }}
        .stTabs [aria-selected="true"] {{
            background:#ecfdf5 !important;
            border-color:#99f6e4 !important;
            color:{COLORS['primary_dark']} !important;
        }}
        .stDataFrame, div[data-testid="stDataFrame"] {{border-radius: 18px; overflow: hidden;}}
        .status-good {{color:{COLORS['good']}; font-weight:800;}}
        .status-warn {{color:{COLORS['warn']}; font-weight:800;}}
        .status-bad  {{color:{COLORS['bad']};  font-weight:800;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# ✅ IMPROVEMENT #9 — chart_card context manager
#    Replaces 25+ duplicate open/close div blocks
# ─────────────────────────────────────────────
@contextmanager
def chart_card():
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def pill(text: str, tone: str) -> str:
    mapping = {
        "good":    ("#ecfdf5", COLORS["good"]),
        "warn":    ("#fff7ed", COLORS["warn"]),
        "bad":     ("#fef2f2", COLORS["bad"]),
        "neutral": ("#f1f5f9", COLORS["muted"]),
    }
    bg, fg = mapping.get(tone, mapping["neutral"])
    return f'<span class="tag" style="background:{bg}; color:{fg};">{text}</span>'


def kpi_card(label: str, value: str, sub: str = "", tag: str = "", tone: str = "neutral"):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
            {pill(tag, tone) if tag else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# XER PARSER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def parse_xer(upload_bytes: bytes):
    text    = upload_bytes.decode("utf-8", errors="ignore").splitlines()
    tables  = {}
    current = None
    cols    = []
    rows    = []
    for raw in text:
        line = raw.rstrip("\n\r")
        if line.startswith("%T\t"):
            if current is not None:
                tables[current] = pd.DataFrame(rows, columns=cols)
            current = line.split("\t", 1)[1]
            cols, rows = [], []
        elif current is not None and line.startswith("%F\t"):
            cols = line.split("\t")[1:]
        elif current is not None and line.startswith("%R\t"):
            vals = line.split("\t")[1:]
            if len(vals) < len(cols):
                vals += [""] * (len(cols) - len(vals))
            rows.append(vals[: len(cols)])
    if current is not None:
        tables[current] = pd.DataFrame(rows, columns=cols)
    return tables


# ─────────────────────────────────────────────
# DATA TYPE HELPERS
# ─────────────────────────────────────────────
def to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(df), index=df.index)


def to_date(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_datetime(df[col], errors="coerce")
    return pd.Series([pd.NaT] * len(df), index=df.index)


def safe_ratio(a, b) -> float:
    try:
        if pd.isna(b) or float(b) == 0:
            return 0.0
        return float(a) / float(b)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# ✅ IMPROVEMENT #7 — Currency-aware money formatter
# ─────────────────────────────────────────────
def fmt_money(v: float, currency: str = "SAR") -> str:
    if abs(v) >= 1_000_000:
        return f"{currency} {v / 1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"{currency} {v / 1_000:,.1f}K"
    return f"{currency} {v:,.0f}"


def fmt_days(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}{abs(int(round(v)))}d"


def performance_tone(v, good=1.0, warn=0.95):
    if v >= good:
        return "good"
    if v >= warn:
        return "warn"
    return "bad"


def style_plot(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=30, r=20, t=36, b=36),
        font=dict(family="Inter, Segoe UI, sans-serif", color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    return fig


# ─────────────────────────────────────────────
# ✅ IMPROVEMENT #10 — Conditional float bar colors
# ─────────────────────────────────────────────
def float_bar_colors(values) -> list:
    """Red for negative float (delayed), green for positive (available)."""
    return [COLORS["bad"] if v < 0 else COLORS["primary"] for v in values]


# ─────────────────────────────────────────────
# EV ENGINE
# ─────────────────────────────────────────────
def earned_pct_from_task(row: pd.Series) -> float:
    status   = row.get("status_code", "")
    pct_type = row.get("complete_pct_type", "")

    raw_phys = pd.to_numeric(row.get("phys_complete_pct", 0), errors="coerce") or 0.0
    # ✅ IMPROVEMENT #2 — Auto-normalise: some P6 versions store 0–1 fraction, others 0–100
    phys = raw_phys if raw_phys > 1.0 else raw_phys * 100

    target_work = pd.to_numeric(row.get("target_work_qty",    0), errors="coerce") or 0.0
    act_work    = pd.to_numeric(row.get("act_work_qty",       0), errors="coerce") or 0.0
    target_drtn = pd.to_numeric(row.get("target_drtn_hr_cnt", 0), errors="coerce") or 0.0
    remain_drtn = pd.to_numeric(row.get("remain_drtn_hr_cnt", 0), errors="coerce") or 0.0

    if status == "TK_Complete":
        return 1.0
    if status == "TK_NotStart":
        return 0.0
    if pct_type == "CP_Phys":
        return max(0.0, min(1.0, safe_ratio(phys, 100)))
    if pct_type == "CP_Units" and target_work > 0:
        return max(0.0, min(1.0, safe_ratio(act_work, target_work)))
    if pct_type == "CP_Drtn" and target_drtn > 0:
        return max(0.0, min(1.0, safe_ratio(target_drtn - remain_drtn, target_drtn)))
    return max(0.0, min(1.0, safe_ratio(phys, 100)))


# ─────────────────────────────────────────────
# TIME ALLOCATION — MONTHLY
# ─────────────────────────────────────────────
def allocate_linear(start, finish, value, cutoff=None):
    """Distribute value linearly across calendar months."""
    if pd.isna(start) or pd.isna(finish) or float(value) == 0:
        return {}
    start  = pd.Timestamp(start)
    finish = pd.Timestamp(finish)
    if finish < start:
        finish = start
    if cutoff is not None and pd.notna(cutoff):
        cutoff = pd.Timestamp(cutoff)
        if cutoff <= start:
            return {}
        finish_for_alloc = min(finish, cutoff)
    else:
        finish_for_alloc = finish
    if finish_for_alloc <= start:
        return {}
    total_seconds = max((finish - start).total_seconds(), 1)
    out = {}
    period_start = start.to_period("M").to_timestamp()
    period_end   = finish_for_alloc.to_period("M").to_timestamp()
    for ms in pd.date_range(period_start, period_end, freq="MS"):
        me      = ms + pd.offsets.MonthBegin(1)
        seg_s   = max(start, ms)
        seg_e   = min(finish_for_alloc, me)
        seconds = max((seg_e - seg_s).total_seconds(), 0)
        if seconds > 0:
            out[ms] = float(value) * seconds / total_seconds
    return out


# ─────────────────────────────────────────────
# ✅ IMPROVEMENT #5 — Weekly allocation helper
#    Distributes hours across the actual work period by week
# ─────────────────────────────────────────────
def allocate_weekly(start, finish, value, cutoff=None):
    """Distribute value linearly across calendar weeks (Sun–Sat anchor)."""
    if pd.isna(start) or pd.isna(finish) or float(value) == 0:
        return {}
    start  = pd.Timestamp(start)
    finish = pd.Timestamp(finish)
    if finish < start:
        finish = start
    if cutoff is not None and pd.notna(cutoff):
        cutoff = pd.Timestamp(cutoff)
        if cutoff <= start:
            return {}
        finish = min(finish, cutoff)
    total_seconds = max((finish - start).total_seconds(), 1)
    # Align to start-of-week (Sunday)
    week_anchor = start - pd.Timedelta(days=start.weekday() + 1 if start.weekday() < 6 else 0)
    out = {}
    for week_start in pd.date_range(week_anchor, finish, freq="7D"):
        seg_s   = max(start, week_start)
        seg_e   = min(finish, week_start + pd.Timedelta(days=7))
        seconds = max((seg_e - seg_s).total_seconds(), 0)
        if seconds > 0:
            out[week_start] = float(value) * seconds / total_seconds
    return out


# ─────────────────────────────────────────────
# S-CURVES
# ─────────────────────────────────────────────
def build_time_curves(task_ev_df: pd.DataFrame, start, end, cutoff):
    planned_map = defaultdict(float)
    earned_map  = defaultdict(float)
    actual_map  = defaultdict(float)
    for _, row in task_ev_df.iterrows():
        ts     = row.get("target_start_date")
        te     = row.get("target_end_date")
        budget = float(row.get("task_budget",  0) or 0)
        ev     = float(row.get("earned_value", 0) or 0)
        ac     = float(row.get("actual_cost",  0) or 0)
        for k, v in allocate_linear(ts, te, budget, cutoff=cutoff).items():
            planned_map[k] += v
        for k, v in allocate_linear(ts, te, ev, cutoff=cutoff).items():
            earned_map[k] += v
        astart = row.get("act_start_date") if pd.notna(row.get("act_start_date")) else ts
        aend   = (
            row.get("act_end_date") if pd.notna(row.get("act_end_date"))
            else (min(cutoff, te) if pd.notna(te) and pd.notna(cutoff) else te)
        )
        for k, v in allocate_linear(astart, aend, ac, cutoff=cutoff).items():
            actual_map[k] += v
    periods = list(pd.date_range(
        pd.Timestamp(start).to_period("M").to_timestamp(),
        pd.Timestamp(end).to_period("M").to_timestamp(),
        freq="MS",
    ))
    out = pd.DataFrame({"period": periods})
    out["planned"]     = out["period"].map(planned_map).fillna(0.0)
    out["earned"]      = out["period"].map(earned_map).fillna(0.0)
    out["actual"]      = out["period"].map(actual_map).fillna(0.0)
    out["cum_planned"] = out["planned"].cumsum()
    out["cum_earned"]  = out["earned"].cumsum()
    out["cum_actual"]  = out["actual"].cumsum()
    return out


# ─────────────────────────────────────────────
# ✅ IMPROVEMENT #11 — Proper PV per WBS group
# ─────────────────────────────────────────────
def compute_group_pv(group_df: pd.DataFrame, cutoff) -> float:
    """Compute Planned Value at cutoff for a subset of tasks."""
    total = 0.0
    for _, row in group_df.iterrows():
        total += sum(allocate_linear(
            row.get("target_start_date"),
            row.get("target_end_date"),
            row.get("task_budget", 0),
            cutoff=cutoff,
        ).values())
    return total


# ─────────────────────────────────────────────
# AREA / PARCEL INFERENCE
# ─────────────────────────────────────────────
def infer_area_mapping(tables, task_df):
    taskactv = tables.get("TASKACTV", pd.DataFrame())
    actvcode = tables.get("ACTVCODE", pd.DataFrame())
    actvtype = tables.get("ACTVTYPE", pd.DataFrame())
    if taskactv.empty or actvcode.empty or actvtype.empty:
        return pd.DataFrame(columns=["task_id", "parcel_id", "parcel_name"])
    merged = taskactv.merge(actvcode, on="actv_code_id", how="left", suffixes=("", "_code"))
    merged = merged.merge(actvtype[["actv_code_type_id", "actv_code_type"]], on="actv_code_type_id", how="left")
    area = merged[merged["actv_code_type"].astype(str).str.lower().eq("area")].copy()
    if area.empty:
        return pd.DataFrame(columns=["task_id", "parcel_id", "parcel_name"])
    area["parcel_id"]   = area["short_name"].replace({"": np.nan}).fillna(area["actv_code_name"])
    area["parcel_name"] = area["actv_code_name"].replace({"": np.nan}).fillna(area["short_name"])
    area = area[["task_id", "parcel_id", "parcel_name"]].drop_duplicates(subset=["task_id"])
    return area


# ─────────────────────────────────────────────
# ✅ IMPROVEMENT #12 — @st.cache_data on build_parcel_df
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_parcel_df(task_ev_df: pd.DataFrame, cutoff):
    if task_ev_df.empty:
        return pd.DataFrame()
    parcel_group = task_ev_df.groupby(
        ["parcel_id", "parcel_name"], dropna=False, as_index=False
    ).agg(
        bac=           ("task_budget",   "sum"),
        ev=            ("earned_value",  "sum"),
        ac=            ("actual_cost",   "sum"),
        min_float=     ("float_days",    "min"),
        active_tasks=  ("status_code",   lambda s: int((s == "TK_Active").sum())),
        completed_tasks=("status_code",  lambda s: int((s == "TK_Complete").sum())),
        total_tasks=   ("task_id",       "count"),
        forecast_finish=("early_end_date",  "max"),
        baseline_finish=("target_end_date", "max"),
    )
    # PV per parcel
    pv_list = []
    for pid, grp in task_ev_df.groupby("parcel_id", dropna=False):
        pv_list.append((pid, compute_group_pv(grp, cutoff)))
    pv_df        = pd.DataFrame(pv_list, columns=["parcel_id", "pv"])
    parcel_group = parcel_group.merge(pv_df, on="parcel_id", how="left")
    parcel_group["pv"]           = parcel_group["pv"].fillna(0.0)
    parcel_group["plan_pct"]     = np.where(parcel_group["bac"] > 0, parcel_group["pv"]  / parcel_group["bac"] * 100, 0.0)
    parcel_group["actual_pct"]   = np.where(parcel_group["bac"] > 0, parcel_group["ev"]  / parcel_group["bac"] * 100, 0.0)
    parcel_group["variance_pct"] = parcel_group["actual_pct"] - parcel_group["plan_pct"]
    parcel_group["delay_days"]   = np.where(
        parcel_group["baseline_finish"].notna() & parcel_group["forecast_finish"].notna(),
        (parcel_group["forecast_finish"] - parcel_group["baseline_finish"]).dt.days,
        0,
    )
    parcel_group["status"] = np.where(
        (parcel_group["variance_pct"] <= -10) | (parcel_group["min_float"] < 0),
        "DELAYED",
        np.where(parcel_group["variance_pct"] < -3, "MINOR DELAY", "ON PLAN"),
    )
    parcel_group = parcel_group.sort_values(
        ["status", "actual_pct", "parcel_id"], ascending=[True, True, True]
    )
    return parcel_group


# ─────────────────────────────────────────────
# SIDEBAR — FILE INPUT
# ─────────────────────────────────────────────
st.sidebar.markdown("## Primavera XER Input")
uploaded_file = st.sidebar.file_uploader("Upload XER file", type=["xer", "txt"])

if uploaded_file is None:
    st.markdown(
        """
        <div class="hero" style="text-align:center; max-width:980px; margin:40px auto; padding:70px 20px;">
            <div style="font-size:62px;">📋</div>
            <div class="hero-project" style="font-size:48px;">Weekly Progress Dashboard</div>
            <div class="hero-sub" style="font-size:20px; margin-top:10px;">
                Drop a Primavera P6 XER file to generate an executive project controls dashboard.
            </div>
            <div class="hero-sub" style="margin-top:14px;">
                Overview · Schedule · Parcels · Cost &amp; EV · Manpower · Risk
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ─────────────────────────────────────────────
# PARSE & LOAD
# ─────────────────────────────────────────────
# ✅ IMPROVEMENT #15 — Spinner while parsing
with st.spinner("Parsing XER file…"):
    tables = parse_xer(uploaded_file.getvalue())

project_df = tables.get("PROJECT", pd.DataFrame())
task_df    = tables.get("TASK",     pd.DataFrame())
tr_df      = tables.get("TASKRSRC", pd.DataFrame())
wbs_df     = tables.get("PROJWBS",  pd.DataFrame())
rsrc_df    = tables.get("RSRC",     pd.DataFrame())

if project_df.empty or task_df.empty:
    st.error("PROJECT / TASK tables are missing in this XER file.")
    st.stop()

# ─────────────────────────────────────────────
# CLEAN DATA
# ─────────────────────────────────────────────
# Dates — task
for col in ["act_start_date", "act_end_date", "target_start_date",
            "target_end_date", "early_start_date", "early_end_date"]:
    task_df[col] = to_date(task_df, col)

# ✅ IMPROVEMENT #14 — Guard tr_df before iterating
if not tr_df.empty:
    for col in ["target_start_date", "target_end_date", "act_start_date", "act_end_date"]:
        tr_df[col] = to_date(tr_df, col)
    for col in ["target_cost", "act_reg_cost", "act_ot_cost", "remain_cost",
                "target_qty", "act_reg_qty", "act_ot_qty"]:
        tr_df[col] = to_num(tr_df, col)

# Numerics — task
for col in ["phys_complete_pct", "target_work_qty", "act_work_qty",
            "target_drtn_hr_cnt", "remain_drtn_hr_cnt", "total_float_hr_cnt"]:
    task_df[col] = to_num(task_df, col)

# ─────────────────────────────────────────────
# PROJECT METADATA
# ─────────────────────────────────────────────
project      = project_df.iloc[0]
project_name = project.get("proj_short_name", "Unknown Project")
project_id   = str(project.get("proj_id", ""))
plan_start   = pd.to_datetime(project.get("plan_start_date", ""), errors="coerce")
plan_finish  = pd.to_datetime(project.get("plan_end_date",   ""), errors="coerce")
forecast_finish = pd.to_datetime(project.get("scd_end_date", ""), errors="coerce")
data_date    = pd.to_datetime(project.get("last_recalc_date", ""), errors="coerce")

# ✅ IMPROVEMENT #4 — End-of-day cutoff for complete daily accounting
cutoff = (
    data_date.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    if pd.notna(data_date) else data_date
)

# ─────────────────────────────────────────────
# SIDEBAR — SETTINGS
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
# ✅ IMPROVEMENT #7 — Currency selector
CURRENCY = st.sidebar.selectbox("Currency", ["SAR", "USD", "AED", "EGP"], index=0)
# ✅ IMPROVEMENT #3 — Configurable working hours per day
HOURS_PER_DAY = st.sidebar.number_input(
    "Hours per workday", min_value=1.0, max_value=24.0, value=8.0, step=0.5
)

# ─────────────────────────────────────────────
# RESOURCE ROLLUP
# ─────────────────────────────────────────────
if not tr_df.empty:
    res_rollup = tr_df.groupby("task_id", as_index=False).agg(
        task_budget=("target_cost",  "sum"),
        actual_reg= ("act_reg_cost", "sum"),
        actual_ot=  ("act_ot_cost",  "sum"),
        remain_cost=("remain_cost",  "sum"),
        target_qty= ("target_qty",   "sum"),
        act_reg_qty=("act_reg_qty",  "sum"),
        act_ot_qty= ("act_ot_qty",   "sum"),
    )
else:
    res_rollup = pd.DataFrame(columns=[
        "task_id", "task_budget", "actual_reg", "actual_ot",
        "remain_cost", "target_qty", "act_reg_qty", "act_ot_qty",
    ])

model = task_df.merge(res_rollup, on="task_id", how="left")
for c in ["task_budget", "actual_reg", "actual_ot", "remain_cost",
          "target_qty",  "act_reg_qty", "act_ot_qty"]:
    model[c] = pd.to_numeric(model[c], errors="coerce").fillna(0.0)

# WBS names
if not wbs_df.empty and "wbs_id" in model.columns:
    keep_cols = [c for c in ["wbs_id", "wbs_short_name", "wbs_name"] if c in wbs_df.columns]
    model = model.merge(
        wbs_df[keep_cols].drop_duplicates(subset=["wbs_id"]),
        on="wbs_id", how="left",
    )

# Parcel (Area activity code → WBS fallback → UNASSIGNED)
parcel_map = infer_area_mapping(tables, task_df)
model = model.merge(parcel_map, on="task_id", how="left")
model["parcel_id"] = (
    model["parcel_id"]
    .fillna(model.get("wbs_short_name", pd.Series(index=model.index, dtype=object)))
    .fillna("UNASSIGNED")
)
model["parcel_name"] = (
    model["parcel_name"]
    .fillna(model.get("wbs_name", pd.Series(index=model.index, dtype=object)))
    .fillna(model["parcel_id"])
)

# Derived metrics
model["actual_cost"]  = model["actual_reg"] + model["actual_ot"]
model["earned_pct"]   = model.apply(earned_pct_from_task, axis=1)
model["earned_value"] = model["task_budget"] * model["earned_pct"]
# ✅ IMPROVEMENT #3 — Use configurable hours per day
model["float_days"]   = model["total_float_hr_cnt"] / HOURS_PER_DAY

# ─────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
parcel_options  = sorted([x for x in model["parcel_id"].dropna().astype(str).unique() if x])
wbs_options     = sorted([x for x in model.get("wbs_name", pd.Series(dtype=str)).dropna().astype(str).unique() if x])
status_options  = ["TK_NotStart", "TK_Active", "TK_Complete"]
selected_parcels = st.sidebar.multiselect("Filter Parcel / Area", parcel_options)
selected_wbs     = st.sidebar.multiselect("Filter WBS",           wbs_options)
selected_status  = st.sidebar.multiselect("Filter Status",        status_options)
critical_only    = st.sidebar.toggle("Critical path only", value=False)

view = model.copy()
if selected_parcels:
    view = view[view["parcel_id"].isin(selected_parcels)]
if selected_wbs and "wbs_name" in view.columns:
    view = view[view["wbs_name"].isin(selected_wbs)]
if selected_status:
    view = view[view["status_code"].isin(selected_status)]
if critical_only:
    view = view[view["float_days"] <= 0]

# ─────────────────────────────────────────────
# MAIN METRICS
# ─────────────────────────────────────────────
BAC = float(view["task_budget"].sum())
EV  = float(view["earned_value"].sum())
AC  = float(view["actual_cost"].sum())

# Build S-Curves first — PV is read from curves to avoid a second iterrows pass
curve_start_candidates = [d for d in [plan_start, view["target_start_date"].min(), view["act_start_date"].min()] if pd.notna(d)]
curve_end_candidates   = [d for d in [forecast_finish, data_date, view["target_end_date"].max(), view["act_end_date"].max()] if pd.notna(d)]
curve_start = min(curve_start_candidates) if curve_start_candidates else pd.Timestamp.today().normalize()
curve_end   = max(curve_end_candidates)   if curve_end_candidates   else pd.Timestamp.today().normalize()
curves      = build_time_curves(view, curve_start, curve_end, cutoff)

# ✅ IMPROVEMENT #1 — PV from curves (no duplicate iterrows loop)
PV = float(curves["cum_planned"].iloc[-1]) if not curves.empty else 0.0

SPI           = safe_ratio(EV, PV)
CPI           = safe_ratio(EV, AC)
plan_pct      = safe_ratio(PV, BAC) * 100
actual_pct    = safe_ratio(EV, BAC) * 100
forecast_delay = int((forecast_finish - plan_finish).days) if pd.notna(forecast_finish) and pd.notna(plan_finish) else 0
min_float      = float(view["float_days"].min()) if not view.empty else 0.0
EAC            = safe_ratio(BAC, CPI) if CPI else BAC
ETC            = EAC - AC
VAC            = BAC - EAC
completed      = int((view["status_code"] == "TK_Complete").sum())
active         = int((view["status_code"] == "TK_Active").sum())
not_started    = int((view["status_code"] == "TK_NotStart").sum())
health         = max(0, min(100, round(
    min(SPI, 1.0) * 40 + min(CPI, 1.0) * 30 + min(actual_pct / max(plan_pct, 1e-9), 1.0) * 30
)))
status_tone = (
    "bad"  if (SPI < 0.95 or min_float < 0 or forecast_delay > 30)
    else "warn" if (SPI < 1 or forecast_delay > 0)
    else "good"
)
status_text = {"bad": "Critical", "warn": "Watch", "good": "Controlled"}[status_tone]

# Parcel analytics
parcel_df = build_parcel_df(view, cutoff)

# ─────────────────────────────────────────────
# MANPOWER
# ─────────────────────────────────────────────
weekly = pd.DataFrame()
if not tr_df.empty and "act_start_date" in tr_df.columns:
    temp = tr_df[tr_df["task_id"].isin(view["task_id"])].copy()
    temp = temp[temp["act_start_date"].notna() & (temp["act_reg_qty"] > 0)]
    if not temp.empty:
        # ✅ IMPROVEMENT #5 — Distribute hours across work period by week
        week_map = defaultdict(float)
        for _, row in temp.iterrows():
            astart = row["act_start_date"]
            aend   = (
                row.get("act_end_date") if pd.notna(row.get("act_end_date"))
                else (cutoff if pd.notna(cutoff) else astart)
            )
            for wk, hrs in allocate_weekly(astart, aend, row["act_reg_qty"], cutoff=cutoff).items():
                week_map[wk] += hrs
        if week_map:
            weekly = (
                pd.DataFrame(list(week_map.items()), columns=["week", "manhours"])
                .sort_values("week")
                .tail(13)
                .reset_index(drop=True)
            )

manhour_budget = float(view["target_qty"].sum())
manhour_actual = float((view["act_reg_qty"] + view["act_ot_qty"]).sum())
manhour_util   = safe_ratio(manhour_actual, manhour_budget) * 100

# ─────────────────────────────────────────────
# RISK REGISTER
# ─────────────────────────────────────────────
risk_rows = []
if forecast_delay > 0:
    # ✅ IMPROVEMENT #6 — Dynamic probability based on delay magnitude
    prob = min(5, max(1, int(forecast_delay / 30)))
    risk_rows.append([
        "Schedule",
        f"Forecast finish is {forecast_delay} days behind baseline.",
        prob, min(5, max(3, 1 + forecast_delay // 60)), "Planning", "Open",
    ])
if SPI < 1:
    risk_rows.append([
        "Schedule", f"SPI is {SPI:.2f}; progress is trailing planned value.",
        5, 4 if SPI < 0.95 else 3, "PMC", "Open",
    ])
if min_float < 0:
    risk_rows.append([
        "Schedule", f"Negative total float detected at {fmt_days(min_float)}.",
        5, 5, "Planning", "Open",
    ])
if CPI < 1:
    risk_rows.append([
        "Financial", f"CPI is {CPI:.2f}; monitor cost efficiency.",
        4, 3, "Contracts", "Active",
    ])
if manhour_budget > 0 and manhour_util < 95:
    risk_rows.append([
        "Resources", f"Manhour utilisation is {manhour_util:.1f}% of budgeted profile.",
        4, 3, "Construction", "Active",
    ])
for _, r in parcel_df.head(5).iterrows():
    if r["status"] != "ON PLAN":
        risk_rows.append([
            "Parcel",
            f"{r['parcel_id']} — {r['parcel_name']} at {r['actual_pct']:.1f}% vs {r['plan_pct']:.1f}% plan.",
            4, 4, "Area Lead", "Watch",
        ])

risk_df = pd.DataFrame(
    risk_rows,
    columns=["category", "risk_description", "probability", "impact", "owner", "status"],
).drop_duplicates()

if not risk_df.empty:
    risk_df["score"]    = risk_df["probability"] * risk_df["impact"]
    risk_df["severity"] = pd.cut(
        risk_df["score"], bins=[-1, 5, 9, 14, 99],
        labels=["Low", "Medium", "High", "Critical"],
    )

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
hero_left, hero_stats = st.columns([3.8, 3])
with hero_left:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">Primavera P6 XER Intelligence Dashboard</div>
            <div class="hero-project">{project_name}</div>
            <div class="hero-sub">
                Data date: {data_date.strftime('%d %b %Y') if pd.notna(data_date) else '-'} ·
                Reporting cutoff: {cutoff.strftime('%d %b %Y') if pd.notna(cutoff) else '-'} ·
                Project ID: {project_id}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_stats:
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Dashboard health", f"{health}/100", "Composite health score", status_text, status_tone)
    with c2: kpi_card("Data date", data_date.strftime("%d %b %Y") if pd.notna(data_date) else "-", "Last recalculation")
    with c3: kpi_card("Overall actual", f"{actual_pct:,.1f}%", "Earned vs budget", "Actual", status_tone)

if forecast_delay > 0:
    st.markdown(
        f'<div class="banner" style="background:#fff5f5; border-color:#fca5a5; color:{COLORS["bad"]};">'
        f'● Schedule Delay Detected — forecast finish is {forecast_delay} days behind baseline.</div>',
        unsafe_allow_html=True,
    )
if SPI < 1:
    st.markdown(
        f'<div class="banner" style="background:#fff8eb; border-color:#fdba74; color:{COLORS["warn"]};">'
        f'● SPI is {SPI:.2f} — progress is trailing the planned curve.</div>',
        unsafe_allow_html=True,
    )

row = st.columns(5)
with row[0]: kpi_card("SPI",            f"{SPI:.2f}",  "Earned / Planned", "Behind" if SPI < 1 else "On plan",      performance_tone(SPI))
with row[1]: kpi_card("CPI",            f"{CPI:.2f}",  "Earned / Actual",  "Controlled" if CPI >= 1 else "Pressure", performance_tone(CPI))
with row[2]: kpi_card("Forecast delay", f"+{forecast_delay}d" if forecast_delay > 0 else "0d",
                       "vs baseline finish", "Overrun" if forecast_delay > 0 else "On time",
                       "bad" if forecast_delay > 0 else "good")
with row[3]: kpi_card("Activities",     f"{len(view):,}", f"{completed} complete · {active} active", f"{not_started} not started", "neutral")
with row[4]: kpi_card("Forecast finish", forecast_finish.strftime("%d %b %Y") if pd.notna(forecast_finish) else "-", "Current finish date")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_overview, tab_schedule, tab_parcels, tab_cost, tab_manpower, tab_risk = st.tabs([
    "Overview", "Schedule", "Parcels", "Cost & EV", "Manpower", "Risk",
])


# ── OVERVIEW ──────────────────────────────────
with tab_overview:
    left, middle, right = st.columns([1.05, 2.6, 1.15])

    with left:
        with chart_card():
            st.markdown("### Project Info")
            info_df = pd.DataFrame({
                "Metric": ["BL Start", "BL Finish", "Forecast Finish", "Delay",
                           "Total Float", "BAC", "PV", "EV", "AC", "EAC", "VAC"],
                "Value": [
                    plan_start.strftime("%d %b %Y")      if pd.notna(plan_start)      else "-",
                    plan_finish.strftime("%d %b %Y")     if pd.notna(plan_finish)     else "-",
                    forecast_finish.strftime("%d %b %Y") if pd.notna(forecast_finish) else "-",
                    f"+{forecast_delay}d" if forecast_delay > 0 else "0d",
                    fmt_days(min_float),
                    fmt_money(BAC, CURRENCY),
                    fmt_money(PV,  CURRENCY),
                    fmt_money(EV,  CURRENCY),
                    fmt_money(AC,  CURRENCY),
                    fmt_money(EAC, CURRENCY),
                    fmt_money(VAC, CURRENCY),
                ],
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)

    with middle:
        with chart_card():
            st.markdown("### Discipline Progress — Plan vs. Actual")

            wbs_series = view.get("wbs_name", pd.Series(index=view.index, dtype=object)).fillna("Unassigned").astype(str)
            grp = (
                pd.DataFrame({
                    "discipline": wbs_series,
                    "bac": pd.to_numeric(view["task_budget"], errors="coerce").fillna(0.0),
                    "ev": pd.to_numeric(view["earned_value"], errors="coerce").fillna(0.0),
                })
                .groupby("discipline", as_index=False)
                .sum()
            )
            grp = grp[grp["bac"] > 0].sort_values("bac", ascending=False).head(8).copy()

            grp_pv = []
            for discipline in grp["discipline"]:
                grp_pv.append(compute_group_pv(view[wbs_series == discipline], cutoff))

            grp["pv"] = grp_pv
            grp["plan_pct"] = np.where(grp["bac"] > 0, grp["pv"] / grp["bac"] * 100, 0.0)
            grp["actual_pct"] = np.where(grp["bac"] > 0, grp["ev"] / grp["bac"] * 100, 0.0)
            grp["variance_pct"] = grp["actual_pct"] - grp["plan_pct"]
            grp[["plan_pct", "actual_pct"]] = grp[["plan_pct", "actual_pct"]].clip(lower=0, upper=100)

            def discipline_color(v):
                if v < -10:
                    return COLORS["bad"]
                if v < -3:
                    return COLORS["warn"]
                return COLORS["primary"]

            def short_label(v, limit=30):
                v = str(v).strip() or "Unassigned"
                return v if len(v) <= limit else v[: limit - 1].rstrip() + "…"

            plot_df = grp.copy()
            plot_df["discipline_label"] = plot_df["discipline"].apply(short_label)
            plot_df["bar_color"] = plot_df["variance_pct"].apply(discipline_color)
            plot_df = plot_df.sort_values(["actual_pct", "plan_pct"], ascending=[True, True]).reset_index(drop=True)
            plot_df["y_pos"] = np.arange(len(plot_df))
            x_max = max(100.0, float(plot_df[["plan_pct", "actual_pct"]].max().max()) + 12.0)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=plot_df["plan_pct"].tolist(),
                y=plot_df["y_pos"].tolist(),
                orientation="h",
                name="Plan",
                marker=dict(color="#e5e7eb"),
                width=0.56,
                hovertemplate="<b>%{customdata}</b><br>Plan: %{x:.1f}%<extra></extra>",
                customdata=plot_df["discipline"],
            ))
            fig.add_trace(go.Bar(
                x=plot_df["actual_pct"].tolist(),
                y=plot_df["y_pos"].tolist(),
                orientation="h",
                name="Actual",
                marker=dict(color=plot_df["bar_color"].tolist()),
                width=0.34,
                hovertemplate="<b>%{customdata}</b><br>Actual: %{x:.1f}%<extra></extra>",
                customdata=plot_df["discipline"],
            ))
            fig.add_trace(go.Scatter(
                x=(np.maximum(plot_df["plan_pct"], plot_df["actual_pct"]) + 3).tolist(),
                y=plot_df["y_pos"].tolist(),
                mode="text",
                text=[f"{v:+.1f}%" for v in plot_df["variance_pct"]],
                textposition="middle right",
                textfont=dict(
                    size=12,
                    color=[discipline_color(v) if v < 0 else COLORS["good"] for v in plot_df["variance_pct"]],
                ),
                hoverinfo="skip",
                showlegend=False,
            ))

            fig.update_layout(
                barmode="overlay",
                height=max(360, 72 + len(plot_df) * 46),
                xaxis_title="Progress %",
                yaxis_title=None,
                margin=dict(l=10, r=34, t=18, b=26),
                legend=dict(orientation="h", y=1.08, x=0),
                xaxis=dict(range=[0, x_max]),
                yaxis=dict(
                    tickmode="array",
                    tickvals=plot_df["y_pos"].tolist(),
                    ticktext=plot_df["discipline_label"].tolist(),
                    autorange="reversed",
                ),
            )
            fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor="#eef2f7", zeroline=False)
            fig.update_yaxes(showgrid=False, automargin=True)
            st.plotly_chart(style_plot(fig, max(360, 72 + len(plot_df) * 46)), use_container_width=True)

    with right:
        with chart_card():
            st.markdown("### Progress Indicators")
            indicator = pd.DataFrame({
                "Label":  ["Plan", "Overall", "SPI", "CPI"],
                "Value":  [plan_pct, actual_pct, SPI * 100, CPI * 100],
                "Suffix": ["%",     "%",        "%",        "%"],
            })
            gcols = st.columns(2)
            for i, (_, r) in enumerate(indicator.iterrows()):
                with gcols[i % 2]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=float(r["Value"]),
                        number={"suffix": r["Suffix"], "font": {"size": 18}},
                        title={"text": r["Label"], "font": {"size": 16}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": COLORS["primary"] if r["Label"] in ["Plan", "Overall"]
                                     else (COLORS["good"] if r["Value"] >= 100
                                           else COLORS["warn"] if r["Value"] >= 95
                                           else COLORS["bad"])},
                            "steps": [
                                {"range": [0,   70],  "color": "#fef2f2"},
                                {"range": [70,  90],  "color": "#fff7ed"},
                                {"range": [90,  100], "color": "#ecfdf5"},
                            ],
                        },
                    ))
                    fig.update_layout(height=230, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="#ffffff")
                    st.plotly_chart(fig, use_container_width=True)

    lower_left, lower_mid, lower_right = st.columns([2.1, 2.0, 1.8])

    with lower_left:
        with chart_card():
            st.markdown("### S-Curve — Cumulative Progress %")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curves["period"],
                y=np.where(BAC > 0, curves["cum_planned"] / BAC * 100, 0),
                name="Baseline Plan", line=dict(color="#cbd5e1", dash="dot", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=curves["period"],
                y=np.where(BAC > 0, curves["cum_earned"] / BAC * 100, 0),
                name="Actual Progress", line=dict(color=COLORS["primary"], width=3),
                fill="tozeroy", fillcolor="rgba(24,183,160,0.10)",
            ))
            fig.update_yaxes(title="%", range=[0, 110])
            st.plotly_chart(style_plot(fig, 360), use_container_width=True)

    with lower_mid:
        with chart_card():
            st.markdown("### Parcel Actual vs. Plan %")
            parcel_plot = parcel_df.sort_values("actual_pct", ascending=False).head(10)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=parcel_plot["parcel_id"], y=parcel_plot["plan_pct"],
                name="Plan%", marker_color="#e5e7eb",
            ))
            fig.add_trace(go.Bar(
                x=parcel_plot["parcel_id"], y=parcel_plot["actual_pct"],
                name="Actual%",
                marker_color=np.where(
                    parcel_plot["status"].eq("DELAYED"),    COLORS["bad"],
                    np.where(parcel_plot["status"].eq("MINOR DELAY"), COLORS["warn"], COLORS["primary"]),
                ),
            ))
            fig.update_layout(barmode="group", xaxis_title=None, yaxis_title="Percent")
            st.plotly_chart(style_plot(fig, 360), use_container_width=True)

    with lower_right:
        with chart_card():
            st.markdown("### Manhour Trend (13 Weeks)")
            if not weekly.empty:
                fig = px.bar(weekly, x="week", y="manhours")
                fig.update_traces(marker_color=COLORS["primary"])
            else:
                fig = go.Figure()
                fig.add_annotation(text="No weekly manhour data available.",
                                    x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            st.plotly_chart(style_plot(fig, 360), use_container_width=True)


# ── SCHEDULE ──────────────────────────────────
with tab_schedule:
    top = st.columns(4)
    with top[0]: kpi_card("BL Duration",          fmt_days((plan_finish - plan_start).days)    if pd.notna(plan_finish)    and pd.notna(plan_start) else "-", "Calendar days")
    with top[1]: kpi_card("At Completion Duration", fmt_days((forecast_finish - plan_start).days) if pd.notna(forecast_finish) and pd.notna(plan_start) else "-", "Forecast total")
    with top[2]: kpi_card("Total Float",           fmt_days(min_float), "Negative = delay", "Critical" if min_float < 0 else "Available", "bad" if min_float < 0 else "good")
    with top[3]: kpi_card("SPI",                   f"{SPI:.2f}", "Earned / Planned", "Behind" if SPI < 1 else "On plan", performance_tone(SPI))

    a, b = st.columns([1.15, 1.25])
    with a:
        with chart_card():
            st.markdown("### Key Schedule Dates")
            key_df = pd.DataFrame({
                "Field": ["BL Start Date", "BL Finish Date", "Data Date",
                          "Forecast Finish", "Delay vs Baseline", "Elapsed Duration"],
                "Value": [
                    plan_start.strftime("%d %b %Y")      if pd.notna(plan_start)      else "-",
                    plan_finish.strftime("%d %b %Y")     if pd.notna(plan_finish)     else "-",
                    data_date.strftime("%d %b %Y")       if pd.notna(data_date)       else "-",
                    forecast_finish.strftime("%d %b %Y") if pd.notna(forecast_finish) else "-",
                    f"+{forecast_delay}d" if forecast_delay > 0 else "0d",
                    fmt_days((cutoff - plan_start).days) if pd.notna(cutoff) and pd.notna(plan_start) else "-",
                ],
            })
            st.dataframe(key_df, use_container_width=True, hide_index=True)

    with b:
        with chart_card():
            st.markdown("### Duration Comparison (Days)")
            dd = pd.DataFrame({
                "Metric": ["BL Duration", "At Completion", "Elapsed"],
                "Days": [
                    (plan_finish    - plan_start).days if pd.notna(plan_finish)    and pd.notna(plan_start) else 0,
                    (forecast_finish - plan_start).days if pd.notna(forecast_finish) and pd.notna(plan_start) else 0,
                    (cutoff         - plan_start).days if pd.notna(cutoff)         and pd.notna(plan_start) else 0,
                ],
            })
            fig = px.bar(dd, x="Metric", y="Days")
            fig.update_traces(marker_color=[COLORS["primary"], COLORS["warn"], COLORS["bad"]])
            st.plotly_chart(style_plot(fig, 340), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        with chart_card():
            st.markdown("### Baseline vs Actual Progress Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curves["period"], y=curves["cum_planned"],
                                     name="PV", line=dict(color="#cbd5e1", dash="dot", width=3)))
            fig.add_trace(go.Scatter(x=curves["period"], y=curves["cum_earned"],
                                     name="EV", line=dict(color=COLORS["primary"], width=3),
                                     fill="tozeroy", fillcolor="rgba(24,183,160,0.10)"))
            st.plotly_chart(style_plot(fig, 350), use_container_width=True)

    with c2:
        with chart_card():
            st.markdown("### Float Distribution by Parcel (Days)")
            plot_df = parcel_df.sort_values("min_float").head(10)
            # ✅ IMPROVEMENT #10 — Conditional colors: red negative, green positive
            bar_colors = float_bar_colors(plot_df["min_float"].tolist())
            fig = px.bar(plot_df, x="min_float", y="parcel_id", orientation="h")
            fig.update_traces(marker_color=bar_colors)
            st.plotly_chart(style_plot(fig, 350), use_container_width=True)


# ── PARCELS ───────────────────────────────────
with tab_parcels:
    with chart_card():
        st.markdown("### Parcel Performance Table")
        table_df = parcel_df[[
            "parcel_id", "parcel_name", "plan_pct", "actual_pct",
            "variance_pct", "min_float", "status",
            "active_tasks", "completed_tasks", "total_tasks",
        ]].copy()
        table_df.columns = [
            "Parcel ID", "Parcel Name", "Plan %", "Actual %",
            "Variance %", "Float (D)", "Status", "Active", "Complete", "Total",
        ]

        # ✅ IMPROVEMENT #8 — Styled dataframe with conditional colors
        def _style_parcel_row(row):
            styles = [""] * len(row)
            cols_list = list(row.index)
            # Status column
            if "Status" in cols_list:
                idx = cols_list.index("Status")
                if row["Status"] == "DELAYED":
                    styles[idx] = "background-color:#fef2f2; color:#ef4444; font-weight:800"
                elif row["Status"] == "MINOR DELAY":
                    styles[idx] = "background-color:#fff7ed; color:#f59e0b; font-weight:800"
                else:
                    styles[idx] = "background-color:#ecfdf5; color:#10b981; font-weight:800"
            # Variance column
            if "Variance %" in cols_list:
                idx = cols_list.index("Variance %")
                if row["Variance %"] < -10:
                    styles[idx] = "color:#ef4444; font-weight:700"
                elif row["Variance %"] < -3:
                    styles[idx] = "color:#f59e0b; font-weight:700"
                else:
                    styles[idx] = "color:#10b981; font-weight:700"
            return styles

        styled_table = (
            table_df.style
            .apply(_style_parcel_row, axis=1)
            .format({
                "Plan %":     "{:.1f}",
                "Actual %":   "{:.1f}",
                "Variance %": "{:+.1f}",
                "Float (D)":  "{:.1f}",
            })
        )
        st.dataframe(styled_table, use_container_width=True, hide_index=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        with chart_card():
            st.markdown("### Actual % — All Parcels")
            fig = px.bar(
                parcel_df, x="parcel_id", y="actual_pct", color="status",
                color_discrete_map={
                    "ON PLAN":    COLORS["good"],
                    "MINOR DELAY": COLORS["warn"],
                    "DELAYED":    COLORS["bad"],
                },
            )
            st.plotly_chart(style_plot(fig, 330), use_container_width=True)

    with p2:
        with chart_card():
            st.markdown("### Plan vs Actual Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=parcel_df["parcel_id"], y=parcel_df["plan_pct"],
                                  name="Plan%",   marker_color="#e5e7eb"))
            fig.add_trace(go.Bar(
                x=parcel_df["parcel_id"], y=parcel_df["actual_pct"], name="Actual%",
                marker_color=np.where(
                    parcel_df["status"].eq("DELAYED"),     COLORS["bad"],
                    np.where(parcel_df["status"].eq("MINOR DELAY"), COLORS["warn"], COLORS["primary"]),
                ),
            ))
            fig.update_layout(barmode="group")
            st.plotly_chart(style_plot(fig, 330), use_container_width=True)

    with p3:
        with chart_card():
            st.markdown("### Float by Parcel (Days)")
            sorted_pf  = parcel_df.sort_values("min_float")
            # ✅ IMPROVEMENT #10 — Conditional colors
            bar_colors = float_bar_colors(sorted_pf["min_float"].tolist())
            fig = px.bar(sorted_pf, x="min_float", y="parcel_id", orientation="h")
            fig.update_traces(marker_color=bar_colors)
            st.plotly_chart(style_plot(fig, 330), use_container_width=True)


# ── COST & EV ─────────────────────────────────
with tab_cost:
    cost_row = st.columns(5)
    with cost_row[0]: kpi_card("BAC", fmt_money(BAC, CURRENCY), "Budget at completion")
    with cost_row[1]: kpi_card("EV",  fmt_money(EV,  CURRENCY), "Earned value")
    with cost_row[2]: kpi_card("AC",  fmt_money(AC,  CURRENCY), "Actual cost")
    with cost_row[3]: kpi_card("PV",  fmt_money(PV,  CURRENCY), "Planned value")
    with cost_row[4]: kpi_card(
        "EAC", fmt_money(EAC, CURRENCY), "Estimate at completion",
        "Overrun" if EAC > BAC else "Under budget",
        "bad" if EAC > BAC else "good",
    )

    c1, c2, c3 = st.columns([1.2, 1.25, 1.2])
    with c1:
        with chart_card():
            st.markdown("### Budget Distribution by Parcel")
            fig = px.pie(parcel_df, names="parcel_id", values="bac", hole=0.45)
            st.plotly_chart(style_plot(fig, 340), use_container_width=True)

    with c2:
        with chart_card():
            st.markdown("### EV vs Planned Value — Parcel")
            plot_df = parcel_df.sort_values("bac", ascending=False).head(8)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=plot_df["parcel_id"], y=plot_df["pv"],
                                  name="Planned Value", marker_color="#e5e7eb"))
            fig.add_trace(go.Bar(x=plot_df["parcel_id"], y=plot_df["ev"],
                                  name="Earned Value",  marker_color=COLORS["primary"]))
            fig.update_layout(barmode="group")
            st.plotly_chart(style_plot(fig, 340), use_container_width=True)

    with c3:
        with chart_card():
            st.markdown("### Cumulative Cost S-Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curves["period"], y=curves["cum_planned"],
                                     name="Planned Value", line=dict(color="#cbd5e1", dash="dot", width=3)))
            fig.add_trace(go.Scatter(x=curves["period"], y=curves["cum_earned"],
                                     name="Earned Value",  line=dict(color=COLORS["primary"], width=3),
                                     fill="tozeroy", fillcolor="rgba(24,183,160,0.10)"))
            fig.add_trace(go.Scatter(x=curves["period"], y=curves["cum_actual"],
                                     name="Actual Cost",   line=dict(color=COLORS["warn"], width=3)))
            st.plotly_chart(style_plot(fig, 340), use_container_width=True)

    with chart_card():
        st.markdown("### Financial Breakdown")
        fin_df = parcel_df[["parcel_id", "parcel_name", "bac", "pv", "ev", "ac", "actual_pct"]].copy()
        fin_df.columns = ["Parcel ID", "Parcel Name", "Budget", "PV", "EV", "AC", "% Complete"]
        for col in ["Budget", "PV", "EV", "AC"]:
            fin_df[col] = fin_df[col].apply(lambda v: fmt_money(v, CURRENCY))
        fin_df["% Complete"] = fin_df["% Complete"].apply(lambda v: f"{v:.1f}%")
        st.dataframe(fin_df, use_container_width=True, hide_index=True)

    # ✅ IMPROVEMENT #13 — Export to Excel
    st.markdown("---")
    if st.button("⬇ Export Financial Breakdown to Excel"):
        export_df = parcel_df[[
            "parcel_id", "parcel_name", "bac", "pv", "ev", "ac",
            "actual_pct", "variance_pct", "delay_days", "status",
        ]].copy()
        export_df.columns = [
            "Parcel ID", "Parcel Name", "BAC", "PV", "EV", "AC",
            "% Complete", "Variance %", "Delay (Days)", "Status",
        ]
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Financial Breakdown")
            parcel_df.to_excel(writer, index=False, sheet_name="Full Parcel Data")
        st.download_button(
            label="📥 Download Excel",
            data=buf.getvalue(),
            file_name=f"{project_name}_financial_breakdown.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ── MANPOWER ──────────────────────────────────
with tab_manpower:
    top = st.columns(4)
    with top[0]: kpi_card("Cumulative manhours",
                           f"{manhour_actual:,.0f} hrs", "Total hours expended")
    with top[1]: kpi_card("This period manhours",
                           f"{weekly['manhours'].iloc[-1]:,.0f} hrs" if not weekly.empty else "0 hrs",
                           "Current reporting week")
    with top[2]: kpi_card("Active tasks",
                           f"{len(view[view['status_code'] == 'TK_Active']):,}", "In progress")
    with top[3]: kpi_card("Manhour utilisation",
                           f"{manhour_util:,.1f}%", "Actual vs budgeted",
                           "High" if manhour_util > 95 else "Low",
                           "good" if manhour_util > 95 else "warn")

    m1, m2, m3 = st.columns([1.45, 1.45, 1.25])
    with m1:
        with chart_card():
            st.markdown("### Weekly Manhour Histogram")
            if not weekly.empty:
                fig = px.bar(weekly, x="week", y="manhours")
                fig.update_traces(marker_color=COLORS["primary"])
            else:
                fig = go.Figure()
                fig.add_annotation(text="No manhour data available.",
                                    x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            st.plotly_chart(style_plot(fig, 330), use_container_width=True)

    with m2:
        with chart_card():
            st.markdown("### Cumulative Manhours Trend")
            if not weekly.empty:
                week2       = weekly.copy()
                week2["cum"] = week2["manhours"].cumsum()
                fig = go.Figure(go.Scatter(
                    x=week2["week"], y=week2["cum"],
                    mode="lines+markers", fill="tozeroy",
                    line=dict(color=COLORS["primary"], width=3),
                ))
            else:
                fig = go.Figure()
            st.plotly_chart(style_plot(fig, 330), use_container_width=True)

    with m3:
        with chart_card():
            st.markdown("### Manhours by Parcel")
            mh = (
                view.groupby("parcel_id", as_index=False)
                .agg(manhours=("target_qty", "sum"))
                .sort_values("manhours", ascending=False)
            )
            fig = px.pie(mh, names="parcel_id", values="manhours", hole=0.48)
            st.plotly_chart(style_plot(fig, 330), use_container_width=True)


# ── RISK ──────────────────────────────────────
with tab_risk:
    if risk_df.empty:
        risk_df = pd.DataFrame(columns=[
            "category", "risk_description", "probability", "impact",
            "owner", "status", "score", "severity",
        ])

    crit = int((risk_df.get("severity") == "Critical").sum()) if not risk_df.empty else 0
    high = int((risk_df.get("severity") == "High").sum())     if not risk_df.empty else 0
    med  = int((risk_df.get("severity") == "Medium").sum())   if not risk_df.empty else 0
    low  = int((risk_df.get("severity") == "Low").sum())      if not risk_df.empty else 0

    rtop = st.columns(4)
    with rtop[0]: kpi_card("Critical", str(crit), "Score ≥ 15",  tone="bad")
    with rtop[1]: kpi_card("High",     str(high), "Score 10–14", tone="warn")
    with rtop[2]: kpi_card("Medium",   str(med),  "Score 6–9")
    with rtop[3]: kpi_card("Low",      str(low),  "Score 1–5",   tone="good")

    rr1, rr2 = st.columns([1.55, 1.2])
    with rr1:
        with chart_card():
            st.markdown("### Risk Register")
            if not risk_df.empty:
                # ✅ IMPROVEMENT #8 — Colored severity column
                def _color_severity(val):
                    if val == "Critical": return "background-color:#fef2f2; color:#ef4444; font-weight:800"
                    if val == "High":     return "background-color:#fff7ed; color:#f59e0b; font-weight:800"
                    if val == "Medium":   return "background-color:#fefce8; color:#a16207; font-weight:800"
                    return "background-color:#ecfdf5; color:#10b981; font-weight:800"

                risk_display = risk_df[[
                    "category", "risk_description", "probability",
                    "impact", "score", "owner", "status", "severity",
                ]].copy()
                styled_risk = risk_display.style.applymap(_color_severity, subset=["severity"])
                st.dataframe(styled_risk, use_container_width=True, hide_index=True)
            else:
                st.info("No risks detected for this project.")

    with rr2:
        with chart_card():
            st.markdown("### Probability × Impact Matrix")
            matrix = [[p * i for i in range(1, 6)] for p in range(5, 0, -1)]
            heat   = pd.DataFrame(matrix, index=[5, 4, 3, 2, 1], columns=[1, 2, 3, 4, 5])
            fig    = px.imshow(
                heat, text_auto=True,
                color_continuous_scale=[[0, "#dcfce7"], [0.5, "#fef3c7"], [1, "#fecaca"]],
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_plot(fig, 330), use_container_width=True)

    with chart_card():
        st.markdown("### Risk by Score")
        if not risk_df.empty:
            plot       = risk_df.copy().sort_values("score", ascending=True)
            plot["label"] = plot["category"] + " — " + plot["risk_description"].str[:45]
            # ✅ Conditional colors by severity
            sev_color = np.where(
                plot["severity"].astype(str) == "Critical", COLORS["bad"],
                np.where(plot["severity"].astype(str) == "High", COLORS["warn"], COLORS["primary"]),
            )
            fig = px.bar(plot, x="score", y="label", orientation="h")
            fig.update_traces(marker_color=sev_color)
        else:
            fig = go.Figure()
        st.plotly_chart(style_plot(fig, 320), use_container_width=True)


# ─────────────────────────────────────────────
# SIDEBAR — DEBUG + NOTES
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
with st.sidebar.expander("🔍 Debug EV / SPI"):
    st.write({
        "BAC":        round(BAC,  2),
        "PV":         round(PV,   2),
        "EV":         round(EV,   2),
        "AC":         round(AC,   2),
        "EAC":        round(EAC,  2),
        "VAC":        round(VAC,  2),
        "Plan %":     round(plan_pct,   4),
        "Actual %":   round(actual_pct, 4),
        "SPI":        round(SPI,  6),
        "CPI":        round(CPI,  6),
        "Cutoff":     str(cutoff),
        "Currency":   CURRENCY,
        "Hours/Day":  HOURS_PER_DAY,
    })

st.sidebar.markdown(
    '<div class="small-note">'
    "Parcel IDs come from the Primavera <b>Area</b> activity code when present; "
    "otherwise the WBS short name is used as fallback."
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Dashboard matches Primavera closely on BAC, EV, AC, SPI, and progress %. "
    "Exact parity is not always possible from XER alone — P6 applies internal calendar, "
    "spread, and summarisation rules not fully exported in XER format."
)
