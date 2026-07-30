"""
app.py — Project Controls Dashboard | XER Analytics

Progress is earned value over budget. Actual cost never enters a progress
figure. Percentages are cost-weighted; groups with no cost basis report as
unmeasured rather than as zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import charts
import evm
import theme
import xer_parser as xp

st.set_page_config(
    page_title="Project Controls Dashboard | XER Analytics",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    try:
        st.image("pcd_logo.png", use_container_width=True)
    except Exception:
        pass
    st.markdown("### Project Controls Dashboard")
    st.caption(f"XER Analytics · Ahmed Saad · build {theme.BUILD}")
    st.divider()

    mode = st.radio("Appearance", ["Dark", "Light"], horizontal=True, index=0).lower()
    current_file = st.file_uploader("Current update (.xer)", type=["xer"], key="curr")
    baseline_file = st.file_uploader("Baseline for comparison (.xer)", type=["xer"], key="base")

    st.divider()
    st.markdown("**Scope**")
    physical_only = st.toggle(
        "Physical scope only", value=True,
        help="Excludes time-related WBS whose value accrues with elapsed time rather "
             "than physical progress. Keeps the earned percentage honest.",
    )
    period_label = st.select_slider("Curve interval", ["Weekly", "Monthly", "Quarterly"], value="Monthly")
    freq = {"Weekly": "W", "Monthly": "M", "Quarterly": "Q"}[period_label]

    with st.expander("Resource calibration"):
        labour_divisor = st.number_input(
            "Labour units divisor", min_value=1.0, max_value=24.0, value=1.0, step=1.0,
            help="Set to the calendar hours per day if Units/Time was entered per day "
                 "while P6 stored it per hour. Affects manhours only, never cost.",
        )

theme.inject_css(st, mode)
TPL = theme.register_plotly(mode)
C = theme.palette(mode)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Reading XER…")
def load(raw: bytes, divisor: float):
    import io
    tables = xp.read_xer(io.BytesIO(raw))
    cals = xp.build_calendars(tables)
    fb = xp.default_calendar(cals, tables)
    header = xp.project_header(tables)
    wbs = xp.wbs_tree(tables)
    acts = evm.build_activities(tables, wbs, cals, fb, header.data_date, divisor)
    return tables, cals, fb, header, wbs, acts


if current_file is None:
    st.markdown('<div class="masthead"><span class="title">Project Controls Dashboard</span>'
                '<span class="meta">XER Analytics</span></div>', unsafe_allow_html=True)
    st.markdown(theme.note(
        "Upload a Primavera <b>.xer</b> export in the sidebar to begin. "
        "Add a second file to unlock update comparison and true baseline variance."),
        unsafe_allow_html=True)
    st.stop()

tables, calendars, fallback, header, wbs, activities = load(current_file.getvalue(), labour_divisor)
pred = tables.get("TASKPRED", pd.DataFrame())
data_date = header.data_date

# Time-related WBS: value accrues with the clock, not with physical work
TIME_RELATED = ["GENERAL REQUIREMENTS", "MILESTONES"]
scope = activities.copy()
if physical_only:
    scope = scope[~scope["wbs_main"].isin(TIME_RELATED)]

daily = evm.time_phase(scope, calendars, fallback, data_date)
periods = evm.to_periods(daily, freq)
m = evm.summarise(scope, periods, data_date, tables.get('PROJWBS'))


def money(v, unit="M"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v / 1e6:,.2f}M" if unit == "M" else f"{v:,.0f}"


def pct(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:,.1f}%"


def idx(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.3f}"


def tone_for(v, good=1.0, warn=0.95):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "none"
    return "good" if v >= good else ("warn" if v >= warn else "bad")


# --------------------------------------------------------------------------
# Masthead
# --------------------------------------------------------------------------

scope_tag = "physical scope" if physical_only else "full scope"
st.markdown(
    f'<div class="masthead">'
    f'<span class="title">{header.short_name}</span>'
    f'<span class="meta">data date <b>{data_date:%d %b %Y}</b></span>'
    f'<span class="meta">activities <b>{len(scope):,}</b></span>'
    f'<span class="meta">BAC <b>SAR {money(m["bac"])}</b></span>'
    f'<span class="meta">basis <b>cost-weighted, {scope_tag}</b></span>'
    f'</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "Overview", "WBS Decomposition", "S-Curve & EVM", "Slippage & Variance",
    "Critical Drivers", "DCMA 14-Point", "Update Comparison", "Data Health", "Activities",
])

# --------------------------------------------------------------------------
# 1. Overview
# --------------------------------------------------------------------------

with tabs[0]:
    cols = st.columns(4)
    cols[0].markdown(theme.kpi(
        "Earned progress", pct(m["performance_pct"]),
        f"EV SAR {money(m['ev'])} of {money(m['bac'])}",
        theme.gauge_strip(m["schedule_pct"], m["performance_pct"]), accent=True),
        unsafe_allow_html=True)
    cols[1].markdown(theme.kpi(
        "Schedule variance", f"SAR {money(m['sv'])}",
        f"earned minus planned value"), unsafe_allow_html=True)
    cols[2].markdown(theme.kpi(
        "SPI · cost based", idx(m["spi"]),
        theme.flag("above 1.00 does not mean on time", "warn")
        if (m["spi"] or 0) > 1 and (m.get("es_spi_t") or 1) < 1 else ""),
        unsafe_allow_html=True)
    cols[3].markdown(theme.kpi(
        "SPI(t) · earned schedule", idx(m.get("es_spi_t")),
        f"{m.get('es_sv_t') or 0:+.2f} months against plan", accent=True),
        unsafe_allow_html=True)

    if (m["spi"] or 0) > 1.0 and (m.get("es_spi_t") or 1) < 1.0:
        st.markdown(theme.note(
            "<b>The two schedule indices disagree.</b> Cost-based SPI is flattered by "
            "value that accrues with elapsed time. Earned Schedule measures progress in "
            "time and is the figure to report.", caution=True), unsafe_allow_html=True)

    st.markdown("#### Progress by main WBS")
    roll = evm.rollup(scope, "wbs_main")

    for _, r in roll.iterrows():
        measurable = r["bac"] > 0
        left, right = st.columns([3, 1])
        with left:
            st.markdown(
                f'<div class="kpi-label">{r["wbs_main"]}</div>'
                + theme.gauge_strip(r["planned_pct"] if measurable else None,
                                    r["earned_pct"] if measurable else None),
                unsafe_allow_html=True)
        with right:
            if measurable:
                st.markdown(
                    f'<div class="kpi-sub">SAR {money(r["bac"])} · {int(r["activities"])} act · '
                    f'{theme.flag("SPI " + idx(r["spi"]), tone_for(r["spi"]))}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="kpi-sub">{int(r["activities"])} act · '
                    f'{theme.flag("no cost basis", "none")}</div>', unsafe_allow_html=True)

    zero = roll[roll["bac"] <= 0]
    if not zero.empty:
        st.markdown(theme.note(
            f"<b>{int(zero['activities'].sum())} activities</b> in "
            f"{', '.join(zero['wbs_main'].astype(str))} carry no budgeted cost, so they cannot be "
            "measured on a cost-weighted basis. They are reported as unmeasured, not as zero progress."),
            unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 2. WBS decomposition
# --------------------------------------------------------------------------

with tabs[1]:
    st.markdown("#### Decomposition")
    levels = sorted(wbs["level"].dropna().unique())
    lvl = st.select_slider("WBS level", options=[int(x) for x in levels if x >= 2],
                           value=int(min([x for x in levels if x >= 2], default=2)))

    name_of = dict(zip(wbs["wbs_id"], wbs["wbs_name"].astype(str)))
    at_level = {}
    for wid, chain in zip(wbs["wbs_id"], wbs["ancestors"]):
        at_level[wid] = name_of.get(chain[lvl - 1]) if len(chain) >= lvl else name_of.get(chain[-1], "—")
    node = scope.copy()
    node["node"] = node["wbs_id"].map(at_level).fillna("—")

    tbl = evm.rollup(node, "node")
    shown = tbl[tbl["bac"] > 0].head(25).iloc[::-1]

    if shown.empty:
        st.markdown(theme.note("No cost-bearing activities at this level."), unsafe_allow_html=True)
    else:
        fig = charts.wbs_bars(shown, "node", C, TPL)
        st.plotly_chart(fig, use_container_width=True)

    disp = tbl.copy()
    disp["Earned %"] = np.where(disp["bac"] > 0, (disp["earned_pct"] * 100).round(1), np.nan)
    disp["Planned %"] = np.where(disp["bac"] > 0, (disp["planned_pct"] * 100).round(1), np.nan)
    disp["Variance %"] = np.where(disp["bac"] > 0, (disp["variance_pct"] * 100).round(1), np.nan)
    disp["BAC (SAR M)"] = (disp["bac"] / 1e6).round(2)
    st.dataframe(
        disp[["node", "BAC (SAR M)", "activities", "Planned %", "Earned %", "Variance %", "spi"]]
        .rename(columns={"node": "WBS", "activities": "Activities", "spi": "SPI"}),
        use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# 3. S-curve
# --------------------------------------------------------------------------

with tabs[2]:
    st.markdown("#### Value curve")
    if periods.empty:
        st.markdown(theme.note("No time-phased value available."), unsafe_allow_html=True)
    else:
        show_ac = st.toggle("Show actual cost", value=False,
                            help="P6 derives actuals from percent complete on this project, "
                                 "so the actual line sits on top of the earned line.")
        fig = charts.scurve(periods, data_date, C, TPL, period_label, show_ac)
        st.plotly_chart(fig, use_container_width=True)

    k = st.columns(5)
    k[0].markdown(theme.kpi("Planned value", f"SAR {money(m['pv'])}"), unsafe_allow_html=True)
    k[1].markdown(theme.kpi("Earned value", f"SAR {money(m['ev'])}", accent=True), unsafe_allow_html=True)
    k[2].markdown(theme.kpi("Estimate at completion", f"SAR {money(m['eac'])}",
                            f"technique {m['etc_technique']}"), unsafe_allow_html=True)
    k[3].markdown(theme.kpi("Variance at completion", f"SAR {money(m['vac'])}"), unsafe_allow_html=True)
    k[4].markdown(theme.kpi("TCPI", idx(m["tcpi"])), unsafe_allow_html=True)

    with st.expander("Calculation method — read from this file, not assumed"):
        tech = scope["ev_technique"].value_counts()
        st.markdown(f"""
| Figure | P6 formula | Setting in this file |
|---|---|---|
| Activity % Complete | per `complete_pct_type` | {', '.join(f'{k} x{v}' for k, v in scope['complete_pct_type'].value_counts().items())} |
| Performance % / EV | BAC x Performance % | `ev_compute_type` = {', '.join(f'{k} x{v}' for k, v in tech.items())} |
| Planned Value | BAC x Schedule % Complete | baseline dates on the activity calendar |
| Schedule % Complete | elapsed baseline working hours / total | `{fallback.name}`, {fallback.days_per_week}-day week |
| ETC | per `ev_etc_compute_type` | {m['etc_technique']} |
| EAC | AC + ETC | SAR {money(m['eac'])} |
| SPI / CPI | EV/PV, EV/AC | {idx(m['spi'])} / {idx(m['cpi'])} |
""")
        st.caption("Cumulative planned value at the data date reconciles exactly to the sum of "
                   "BAC x Schedule % Complete across every activity.")

    st.markdown(theme.note(
        "The earned curve <b>before</b> the data date is reconstructed by spreading each activity's "
        "earned value across its actual dates. A single XER holds no progress history, so that part "
        "of the curve is an assumption. Every figure <b>at</b> the data date is calculated, not "
        "assumed."), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 4. Slippage & variance
# --------------------------------------------------------------------------

with tabs[3]:
    st.markdown("#### Slippage & variance")
    st.markdown(theme.note(
        "Variance analysis for internal schedule control. <b>Not a forensic delay analysis</b> "
        "and not a statement of entitlement."), unsafe_allow_html=True)

    slip = an.activity_slippage(scope, calendars, fallback)
    impact = an.delay_impact(slip, pred)
    by_wbs = an.wbs_slippage(slip)

    counts = slip["impact_class"].value_counts()
    k = st.columns(4)
    k[0].markdown(theme.kpi("Critical", f"{counts.get('Critical', 0):,}",
                            "slipped with no float left"), unsafe_allow_html=True)
    k[1].markdown(theme.kpi("Float-eroding", f"{counts.get('Float-eroding', 0):,}",
                            "slipped, float still positive"), unsafe_allow_html=True)
    k[2].markdown(theme.kpi("Originating", f"{(slip['origin'] == 'Originating').sum():,}",
                            "self-generated, not inherited", accent=True), unsafe_allow_html=True)
    k[3].markdown(theme.kpi("Worst slip", f"{np.nanmax(slip['finish_var_d']):,.0f} d"
                            if slip["finish_var_d"].notna().any() else "—",
                            "working days"), unsafe_allow_html=True)

    if not by_wbs.empty:
        b = by_wbs.iloc[::-1]
        fig = charts.slip_bars(b, "wbs_main", "worst_slip_d", C, TPL)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Most consequential delays")
    st.caption("Ranked by originating slip weighted by downstream successors — not by size of slip alone.")
    top = impact[impact["delay_impact"] > 0].head(20)
    if top.empty:
        st.markdown(theme.note("No originating delays detected in the current scope."),
                    unsafe_allow_html=True)
    else:
        st.dataframe(
            top[["task_code", "task_name", "wbs_main", "finish_var_d", "total_float_d",
                 "successors", "cause", "impact_class", "delay_impact"]]
            .rename(columns={
                "task_code": "Activity", "task_name": "Name", "wbs_main": "WBS",
                "finish_var_d": "Slip (d)", "total_float_d": "Float (d)",
                "successors": "Succs", "cause": "Cause", "impact_class": "Class",
                "delay_impact": "Impact"}).round(1),
            use_container_width=True, hide_index=True)

    if by_wbs["finish_var_d"].fillna(0).abs().sum() == 0:
        st.markdown(theme.note(
            "Group finish variance is zero across every WBS because this file's planned dates "
            "track its own forecast. Load the approved baseline as a second file to measure "
            "real variance; until then rely on activity slip and SPI(t).", caution=True),
            unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 5. Critical drivers
# --------------------------------------------------------------------------

with tabs[4]:
    st.markdown("#### Critical drivers")
    lp_only = st.toggle("Longest path only", value=True,
                        help="Excludes level-of-effort and management activities from the ranking.")
    drivers = an.critical_drivers(scope, pred, fallback, longest_path_only=lp_only)
    st.caption("Impact Score = remaining duration (working days) × (1 + successor count)")

    top = drivers.head(15).iloc[::-1]
    if top.empty:
        st.markdown(theme.note("No driving activities found. Check that the schedule has been "
                               "scheduled with the longest-path flag written."), unsafe_allow_html=True)
    else:
        fig = charts.driver_bars(top, C, TPL, dark=(mode == "dark"))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            drivers.head(25)[["task_code", "task_name", "wbs_main", "remain_d",
                              "successors", "impact_score", "forecast_finish"]]
            .rename(columns={"task_code": "Activity", "task_name": "Name", "wbs_main": "WBS",
                             "remain_d": "Rem. dur (d)", "successors": "Succs",
                             "impact_score": "Impact", "forecast_finish": "Forecast finish"}).round(1),
            use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# 6. DCMA
# --------------------------------------------------------------------------

with tabs[5]:
    st.markdown("#### DCMA 14-point assessment")
    d = an.dcma_14(activities, pred, calendars, fallback, data_date)
    passed = int((d["Verdict"] == "PASS").sum())
    c1, c2 = st.columns([1, 3])
    c1.markdown(theme.kpi("Checks passed", f"{passed} / 14",
                          accent=passed >= 12), unsafe_allow_html=True)
    with c2:
        failed = d[d["Verdict"] == "FAIL"]["Check"].tolist()
        if failed:
            st.markdown('<div class="kpi-label">Failing</div>'
                        + " ".join(theme.flag(f, "bad") for f in failed), unsafe_allow_html=True)
    st.dataframe(d, use_container_width=True, hide_index=True)
    st.markdown(theme.note(
        "Assessment runs on the <b>full schedule</b>, not the physical-scope filter — schedule "
        "quality is a property of the whole network."), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 7. Comparison
# --------------------------------------------------------------------------

with tabs[6]:
    st.markdown("#### Update comparison")
    if baseline_file is None:
        st.markdown(theme.note(
            "Upload a second .xer in the sidebar to diff two updates: activities added and "
            "removed, date movement, duration changes and progress regression."),
            unsafe_allow_html=True)
    else:
        b_tables, b_cals, b_fb, b_header, b_wbs, b_acts = load(baseline_file.getvalue(), labour_divisor)
        diff = an.compare_updates(b_acts, activities, fallback)
        k = st.columns(4)
        k[0].markdown(theme.kpi("Added", f"{len(diff['added']):,}"), unsafe_allow_html=True)
        k[1].markdown(theme.kpi("Removed", f"{len(diff['removed']):,}"), unsafe_allow_html=True)
        k[2].markdown(theme.kpi("Changed", f"{len(diff['changed']):,}"), unsafe_allow_html=True)
        reg = int(diff["changed"]["regressed"].sum()) if not diff["changed"].empty else 0
        k[3].markdown(theme.kpi("Progress regressed", f"{reg:,}",
                                "percent complete went backwards"), unsafe_allow_html=True)
        st.caption(f"Comparing **{b_header.short_name}** ({b_header.data_date:%d %b %Y}) → "
                   f"**{header.short_name}** ({data_date:%d %b %Y})")
        if not diff["changed"].empty:
            st.dataframe(diff["changed"].round(3), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# 8. Data health
# --------------------------------------------------------------------------

with tabs[7]:
    st.markdown("#### Data health")
    st.dataframe(an.data_health(activities, pred, data_date),
                 use_container_width=True, hide_index=True)

    ac_total = float(activities["ac"].sum())
    ev_total = float(activities["ev"].sum())
    if ac_total > 0 and abs(ac_total - ev_total) < max(1.0, ac_total * 1e-6):
        st.markdown(theme.note(
            "<b>Actual cost equals earned value on every assignment.</b> P6 is deriving actuals "
            "from percent complete rather than recording them independently, so CPI is fixed at "
            "1.000 and carries no information. Cost performance cannot be assessed from this file.",
            caution=True), unsafe_allow_html=True)

    lab = float(activities["labour_hr"].sum())
    if lab > 0:
        months = max(1, (activities["target_end_date"].max()
                         - activities["target_start_date"].min()).days / 30.4)
        st.markdown(theme.note(
            f"Budgeted labour: <b>{lab:,.0f} hours</b> over ~{months:.0f} months "
            f"(~{lab / months:,.0f} hrs/month). If that peak looks impossible, Units/Time was "
            "likely entered per day while P6 stored it per hour — set the labour divisor to the "
            "calendar hours per day in the sidebar."), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 9. Activities
# --------------------------------------------------------------------------

with tabs[8]:
    st.markdown("#### Activity register")
    seg = xp.taxonomy_frame(scope, ".", ("Project", "Discipline", "Element", "Phase", "Sequence"))
    reg = pd.concat([scope.reset_index(drop=True), seg.reset_index(drop=True)], axis=1)

    f1, f2, f3 = st.columns(3)
    disc = f1.multiselect("Discipline", sorted(reg.get("Discipline", pd.Series(dtype=str)).unique()))
    stat = f2.multiselect("Status", sorted(reg["status_code"].astype(str).unique()))
    search = f3.text_input("Search name or ID")

    if disc:
        reg = reg[reg["Discipline"].isin(disc)]
    if stat:
        reg = reg[reg["status_code"].astype(str).isin(stat)]
    if search:
        s = search.lower()
        reg = reg[reg["task_code"].astype(str).str.lower().str.contains(s)
                  | reg["task_name"].astype(str).str.lower().str.contains(s)]

    reg["Earned %"] = (reg["pct_complete"] * 100).round(1)
    reg["BAC (SAR)"] = reg["bac"].round(0)
    reg["EV (SAR)"] = reg["ev"].round(0)
    st.caption(f"{len(reg):,} activities")
    st.dataframe(
        reg[["task_code", "task_name", "wbs_main", "status_code", "complete_pct_type",
             "Earned %", "BAC (SAR)", "EV (SAR)", "target_end_date", "forecast_finish"]]
        .rename(columns={"task_code": "Activity", "task_name": "Name", "wbs_main": "WBS",
                         "status_code": "Status", "complete_pct_type": "% type",
                         "target_end_date": "Planned finish", "forecast_finish": "Forecast finish"}),
        use_container_width=True, hide_index=True, height=560)
