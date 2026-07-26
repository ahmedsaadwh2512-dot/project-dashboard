"""
evm.py — Earned-value engine.

Rules enforced here
-------------------
1. There is no universal "% complete" column in an XER. Every activity is
   resolved through TASK.complete_pct_type.
2. Rollups are weighted by BAC. Never a plain mean of activity percentages.
3. Actual cost is loaded but never enters a progress figure. Progress is
   EV / BAC. AC appears only in CPI and cost-forecast panels.
4. Days are working days on the owning calendar, never calendar days and
   never a hard-coded 8-hour divisor.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from xer_parser import Calendar

LABOUR = "RT_Labor"
EQUIP = "RT_Equip"
MATERIAL = "RT_Mat"


# --------------------------------------------------------------------------
# 1. Percent complete
# --------------------------------------------------------------------------

def physical_scale(task: pd.DataFrame) -> float:
    """Return 100.0 if phys_complete_pct is stored 0-100, else 1.0."""
    if "phys_complete_pct" not in task.columns:
        return 100.0
    peak = pd.to_numeric(task["phys_complete_pct"], errors="coerce").max()
    return 100.0 if pd.notna(peak) and peak > 1.5 else 1.0


def resolve_percent(task: pd.DataFrame) -> pd.Series:
    """
    Resolve each activity's % complete (0-1) according to its own
    complete_pct_type. This is the single most common source of wrong
    progress figures in XER dashboards.
    """
    scale = physical_scale(task)
    n = len(task)

    phys = pd.to_numeric(task.get("phys_complete_pct", pd.Series(0, index=task.index)),
                         errors="coerce").fillna(0) / scale

    target_hr = pd.to_numeric(task.get("target_drtn_hr_cnt", pd.Series(0, index=task.index)),
                              errors="coerce").fillna(0)
    remain_hr = pd.to_numeric(task.get("remain_drtn_hr_cnt", pd.Series(0, index=task.index)),
                              errors="coerce").fillna(0)
    # P6 measures duration % against at-complete duration, not original duration
    at_complete = np.maximum(target_hr, remain_hr)
    drtn = np.where(at_complete > 0, (at_complete - remain_hr) / at_complete, 0.0)

    act_q = pd.to_numeric(task.get("act_work_qty", pd.Series(0, index=task.index)),
                          errors="coerce").fillna(0)
    rem_q = pd.to_numeric(task.get("remain_work_qty", pd.Series(0, index=task.index)),
                          errors="coerce").fillna(0)
    denom = act_q + rem_q
    units = np.where(denom > 0, act_q / denom, 0.0)

    kind = task.get("complete_pct_type", pd.Series(["CP_Phys"] * n, index=task.index)).astype(str)
    pct = np.select(
        [kind == "CP_Drtn", kind == "CP_Units", kind == "CP_Phys"],
        [drtn, units, phys],
        default=phys,
    )

    # Status always wins over a stale percentage field
    status = task.get("status_code", pd.Series([""] * n, index=task.index)).astype(str)
    pct = np.where(status == "TK_Complete", 1.0, pct)
    pct = np.where(status == "TK_NotStart", 0.0, pct)

    return pd.Series(np.clip(pct, 0.0, 1.0), index=task.index, name="pct_complete")


# --------------------------------------------------------------------------
# 2. Activity frame
# --------------------------------------------------------------------------

def build_activities(
    tables: dict[str, pd.DataFrame],
    wbs: pd.DataFrame,
    labour_divisor: float = 1.0,
) -> pd.DataFrame:
    """
    One row per activity, carrying BAC, EV, AC, budgeted labour hours,
    WBS lineage and the dates needed for time-phasing.

    labour_divisor corrects resource loading where Units/Time was entered
    on a per-day basis while P6 stored it per hour. Default 1.0 = untouched.
    """
    task = tables.get("TASK", pd.DataFrame()).copy()
    if task.empty:
        return pd.DataFrame()

    task["task_id"] = task["task_id"].astype(str)
    task["wbs_id"] = task["wbs_id"].astype(str)
    task["clndr_id"] = task["clndr_id"].astype(str)
    task["pct_complete"] = resolve_percent(task)

    # --- money and hours from resource assignments ------------------------
    tr = tables.get("TASKRSRC", pd.DataFrame()).copy()
    if not tr.empty:
        tr["task_id"] = tr["task_id"].astype(str)
        for c in ("target_cost", "act_reg_cost", "act_ot_cost", "remain_cost",
                  "target_qty", "act_reg_qty", "act_ot_qty", "remain_qty"):
            if c not in tr.columns:
                tr[c] = 0.0
            tr[c] = pd.to_numeric(tr[c], errors="coerce").fillna(0.0)
        tr["ac"] = tr["act_reg_cost"] + tr["act_ot_cost"]

        rtype = tr.get("rsrc_type", pd.Series([""] * len(tr), index=tr.index)).astype(str)
        # Labour hours only. Material-type assignments carry SAR in target_qty
        # on this project, so summing all target_qty inflates hours ~17x.
        tr["labour_hr"] = np.where(rtype == LABOUR, tr["target_qty"], 0.0) / max(labour_divisor, 1e-9)
        tr["equip_hr"] = np.where(rtype == EQUIP, tr["target_qty"], 0.0) / max(labour_divisor, 1e-9)
        tr["act_labour_hr"] = np.where(
            rtype == LABOUR, tr["act_reg_qty"] + tr["act_ot_qty"], 0.0
        ) / max(labour_divisor, 1e-9)

        agg = tr.groupby("task_id").agg(
            bac=("target_cost", "sum"),
            ac=("ac", "sum"),
            remain_cost=("remain_cost", "sum"),
            labour_hr=("labour_hr", "sum"),
            equip_hr=("equip_hr", "sum"),
            act_labour_hr=("act_labour_hr", "sum"),
        )
        task = task.merge(agg, left_on="task_id", right_index=True, how="left")

    for c in ("bac", "ac", "remain_cost", "labour_hr", "equip_hr", "act_labour_hr"):
        if c not in task.columns:
            task[c] = 0.0
        task[c] = task[c].fillna(0.0)

    # --- earned value -----------------------------------------------------
    task["ev"] = task["bac"] * task["pct_complete"]
    task["earned_hr"] = task["labour_hr"] * task["pct_complete"]

    # --- WBS lineage ------------------------------------------------------
    if not wbs.empty:
        lineage = wbs.set_index("wbs_id")
        task = task.merge(
            lineage[["wbs_name", "path", "level", "ancestors"]],
            left_on="wbs_id", right_index=True, how="left", suffixes=("", "_wbs"),
        )
        top = {}
        name_of = dict(zip(wbs["wbs_id"], wbs["wbs_name"].astype(str)))
        for wid, chain in zip(wbs["wbs_id"], wbs["ancestors"]):
            top[wid] = name_of.get(chain[1], name_of.get(chain[0], "—")) if len(chain) > 1 \
                else name_of.get(chain[0], "—")
        task["wbs_main"] = task["wbs_id"].map(top).fillna("—")
    else:
        task["wbs_name"] = "—"
        task["wbs_main"] = "—"
        task["path"] = "—"

    # --- float in working days -------------------------------------------
    task["total_float_hr_cnt"] = pd.to_numeric(
        task.get("total_float_hr_cnt", 0), errors="coerce"
    ).fillna(0)

    # --- forecast finish: reend_date preferred for in-progress work -------
    task["forecast_finish"] = task.get("reend_date")
    task["forecast_finish"] = task["forecast_finish"].fillna(task.get("early_end_date"))
    task["forecast_finish"] = task["forecast_finish"].fillna(task.get("act_end_date"))
    task["forecast_start"] = task.get("restart_date")
    task["forecast_start"] = task["forecast_start"].fillna(task.get("early_start_date"))
    task["forecast_start"] = task["forecast_start"].fillna(task.get("act_start_date"))

    return task


def float_days(task: pd.DataFrame, calendars: dict[str, Calendar], fallback_hpd: float = 8.0) -> pd.Series:
    """Total float converted to working days using each activity's own calendar."""
    hpd = task["clndr_id"].map(
        {k: v.hours_per_day for k, v in calendars.items()}
    ).fillna(fallback_hpd)
    return task["total_float_hr_cnt"] / hpd


# --------------------------------------------------------------------------
# 3. Time phasing
# --------------------------------------------------------------------------

def _spread(cal: Calendar, start, finish, value: float) -> dict[date, float]:
    """Spread a value evenly across the working days between two dates."""
    if value == 0 or pd.isna(start) or pd.isna(finish):
        return {}
    s, f = pd.Timestamp(start).date(), pd.Timestamp(finish).date()
    if f < s:
        f = s
    days = cal.working_days(s, f)
    if not days:
        days = [s]
    share = value / len(days)
    return {d: share for d in days}


def time_phase(
    activities: pd.DataFrame,
    calendars: dict[str, Calendar],
    fallback: Calendar,
    data_date: pd.Timestamp,
    value_col: str = "bac",
) -> pd.DataFrame:
    """
    Daily incremental PV / EV / AC / forecast series.

    PV   spread over baseline (target) dates.
    EV   spread over actual dates up to the data date — a reconstruction,
         since a single XER holds no progress history. Flagged in the UI.
    AC   spread the same way as EV.
    FCST remaining cost spread forward over remaining dates.
    """
    pv: dict[date, float] = {}
    ev: dict[date, float] = {}
    ac: dict[date, float] = {}
    fc: dict[date, float] = {}
    dd = pd.Timestamp(data_date).date() if pd.notna(data_date) else date.today()

    for row in activities.itertuples(index=False):
        cal = calendars.get(getattr(row, "clndr_id", ""), fallback)
        value = float(getattr(row, value_col, 0.0) or 0.0)
        if value <= 0:
            continue

        for d, v in _spread(cal, getattr(row, "target_start_date", None),
                            getattr(row, "target_end_date", None), value).items():
            pv[d] = pv.get(d, 0.0) + v

        earned = value * float(getattr(row, "pct_complete", 0.0))
        actual_cost = float(getattr(row, "ac", 0.0) or 0.0) if value_col == "bac" else 0.0
        a_start = getattr(row, "act_start_date", None)
        a_end = getattr(row, "act_end_date", None)
        if earned > 0:
            e_start = a_start if pd.notna(a_start) else getattr(row, "target_start_date", None)
            e_end = a_end if pd.notna(a_end) else pd.Timestamp(dd)
            for d, v in _spread(cal, e_start, e_end, earned).items():
                if d <= dd:
                    ev[d] = ev.get(d, 0.0) + v
            if actual_cost > 0:
                for d, v in _spread(cal, e_start, e_end, actual_cost).items():
                    if d <= dd:
                        ac[d] = ac.get(d, 0.0) + v

        remaining = value - earned
        if remaining > 0:
            f_start = getattr(row, "forecast_start", None)
            f_start = f_start if pd.notna(f_start) else pd.Timestamp(dd)
            f_start = max(pd.Timestamp(f_start).date(), dd)
            f_end = getattr(row, "forecast_finish", None)
            f_end = pd.Timestamp(f_end).date() if pd.notna(f_end) else f_start
            for d, v in _spread(cal, f_start, max(f_end, f_start), remaining).items():
                fc[d] = fc.get(d, 0.0) + v

    idx = sorted(set(pv) | set(ev) | set(ac) | set(fc))
    if not idx:
        return pd.DataFrame(columns=["day", "pv", "ev", "ac", "fcst"])
    out = pd.DataFrame({
        "day": pd.to_datetime(idx),
        "pv": [pv.get(d, 0.0) for d in idx],
        "ev": [ev.get(d, 0.0) for d in idx],
        "ac": [ac.get(d, 0.0) for d in idx],
        "fcst": [fc.get(d, 0.0) for d in idx],
    })
    return out.sort_values("day").reset_index(drop=True)


def to_periods(daily: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Aggregate the daily series to weekly or monthly buckets, with cumulatives."""
    if daily.empty:
        return daily
    rule = {"W": "W-THU", "M": "ME", "Q": "QE"}.get(freq, "ME")
    g = daily.set_index("day").resample(rule).sum(numeric_only=True).reset_index()
    g = g.rename(columns={"day": "period"})
    for c in ("pv", "ev", "ac", "fcst"):
        g[f"cum_{c}"] = g[c].cumsum()
    # Forecast continues from the earned curve, not from zero
    ev_final = g.loc[g["ev"] > 0, "cum_ev"].max() if (g["ev"] > 0).any() else 0.0
    g["cum_fcst"] = np.where(g["cum_fcst"] > 0, g["cum_fcst"] + ev_final, np.nan)
    return g


# --------------------------------------------------------------------------
# 4. Metrics
# --------------------------------------------------------------------------

def earned_schedule(periods: pd.DataFrame, ev_now: float, data_date: pd.Timestamp) -> dict:
    """
    Earned Schedule: the point on the baseline curve where planned value
    equals the value actually earned. Gives schedule variance in TIME.

    Solves the classic SPI defect where the index drifts to 1.0 near
    completion regardless of how late the project is.
    """
    blank = {"es": None, "at": None, "sv_t": None, "spi_t": None}
    if periods.empty or ev_now <= 0 or pd.isna(data_date):
        return blank

    curve = periods[["period", "cum_pv"]].dropna()
    if curve.empty or curve["cum_pv"].max() < ev_now:
        return blank

    start = curve["period"].iloc[0]
    below = curve[curve["cum_pv"] <= ev_now]
    above = curve[curve["cum_pv"] > ev_now]
    if above.empty:
        return blank

    if below.empty:
        es_months = 0.0
        es_date = start
    else:
        lo, hi = below.iloc[-1], above.iloc[0]
        span = hi["cum_pv"] - lo["cum_pv"]
        frac = (ev_now - lo["cum_pv"]) / span if span > 0 else 0.0
        es_date = lo["period"] + (hi["period"] - lo["period"]) * frac
        es_months = (es_date - start).days / 30.4375

    at_months = (pd.Timestamp(data_date) - start).days / 30.4375
    if at_months <= 0:
        return blank
    return {
        "es": es_date,
        "es_months": es_months,
        "at": at_months,
        "sv_t": es_months - at_months,
        "spi_t": es_months / at_months if at_months else None,
    }


def summarise(activities: pd.DataFrame, periods: pd.DataFrame, data_date: pd.Timestamp) -> dict:
    """Headline EVM figures. Progress percentages never touch actual cost."""
    bac = float(activities["bac"].sum())
    ev = float(activities["ev"].sum())
    ac = float(activities["ac"].sum())
    pv = float(periods.loc[periods["period"] <= data_date, "pv"].sum()) if not periods.empty else 0.0

    def ratio(a, b):
        return a / b if b else None

    spi = ratio(ev, pv)
    cpi = ratio(ev, ac)
    eac = bac / cpi if cpi else None
    es = earned_schedule(periods, ev, data_date)

    return {
        "bac": bac,
        "pv": pv,
        "ev": ev,
        "ac": ac,
        "planned_pct": ratio(pv, bac),
        "earned_pct": ratio(ev, bac),
        "sv": ev - pv,
        "cv": ev - ac,
        "spi": spi,
        "cpi": cpi,
        "eac": eac,
        "etc": (eac - ac) if eac is not None else None,
        "vac": (bac - eac) if eac is not None else None,
        "tcpi": ratio(bac - ev, bac - ac) if (bac - ac) else None,
        **{f"es_{k}": v for k, v in es.items()},
    }


def rollup(activities: pd.DataFrame, periods_by_group: dict[str, float], key: str) -> pd.DataFrame:
    """BAC-weighted rollup of planned vs earned by any grouping column."""
    if activities.empty:
        return pd.DataFrame()
    g = activities.groupby(key, dropna=False).agg(
        bac=("bac", "sum"),
        ev=("ev", "sum"),
        ac=("ac", "sum"),
        labour_hr=("labour_hr", "sum"),
        earned_hr=("earned_hr", "sum"),
        activities=("task_id", "count"),
    ).reset_index()
    g["pv"] = g[key].map(periods_by_group).fillna(0.0)
    g["planned_pct"] = np.where(g["bac"] > 0, g["pv"] / g["bac"], 0.0)
    g["earned_pct"] = np.where(g["bac"] > 0, g["ev"] / g["bac"], 0.0)
    g["variance_pct"] = g["earned_pct"] - g["planned_pct"]
    g["spi"] = np.where(g["pv"] > 0, g["ev"] / g["pv"], np.nan)
    g["cpi"] = np.where(g["ac"] > 0, g["ev"] / g["ac"], np.nan)
    return g.sort_values("bac", ascending=False)
