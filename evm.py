"""
evm.py — Earned-value engine, calculated the way Primavera P6 calculates it.

Every formula below is P6's, and every P6 setting that drives a formula is
read from the file rather than assumed:

  Activity % Complete   per TASK.complete_pct_type
  Performance % / EV    per PROJWBS.ev_compute_type      (EV technique)
  ETC and EAC           per PROJWBS.ev_etc_compute_type  (ETC technique)
  Planned Value         BAC x Schedule % Complete, on the activity calendar

Deviations from P6 present in the previous version, now fixed:

  1. Duration % Complete used at-complete duration as the denominator.
     P6 uses Original (Planned) Duration:  (OD - RD) / OD, clipped at zero.
  2. Units % Complete counted labour units only. P6 counts labour plus
     nonlabour:  Actual Units / At Completion Units.
  3. EV assumed the Activity % Complete technique. P6 applies the technique
     recorded on the owning WBS.
  4. EAC used BAC / CPI. This project is set to EE_Rem_hr, where
     ETC = Remaining Cost and EAC = AC + ETC.
  5. Planned Value was spread over working days counted equally. P6 measures
     Schedule % Complete in working hours, so exception days were mis-weighted.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from xer_parser import Calendar

LABOUR = "RT_Labor"
EQUIP = "RT_Equip"

# PROJWBS.ev_compute_type — how P6 derives Performance % Complete
EV_ACTIVITY_PCT = "EC_Cmp_pct"        # Activity % Complete
EV_0_100 = "EC_0_100"                 # 0 / 100
EV_50_50 = "EC_50_50"                 # 50 / 50
EV_USER_PCT = "EC_Cmp_pct_udf"        # WBS custom percent (ev_user_pct)

# PROJWBS.ev_etc_compute_type — how P6 derives ETC
ETC_REMAINING = "EE_Rem_hr"           # ETC = Remaining Cost
ETC_PF_1 = "EE_Pf_1"                  # ETC = (BAC - EV) / 1
ETC_PF_CPI = "EE_Pf_cpi"              # ETC = (BAC - EV) / CPI
ETC_PF_USER = "EE_Pf_udf"             # ETC = (BAC - EV) / user PF


# --------------------------------------------------------------------------
# 1. Activity % Complete
# --------------------------------------------------------------------------

def physical_scale(task: pd.DataFrame) -> float:
    """100.0 if phys_complete_pct is stored 0-100, else 1.0."""
    if "phys_complete_pct" not in task.columns:
        return 100.0
    peak = pd.to_numeric(task["phys_complete_pct"], errors="coerce").max()
    return 100.0 if pd.notna(peak) and peak > 1.5 else 1.0


def activity_percent(task: pd.DataFrame) -> pd.Series:
    """
    Activity % Complete, 0-1, exactly as P6 computes it.

      Physical   the stored Physical % Complete
      Duration   (Original Duration - Remaining Duration) / Original Duration
      Units      Actual Units / At Completion Units, labour + nonlabour

    P6 clips Duration % at zero when Remaining exceeds Original, and forces
    100% once an activity is marked complete.
    """
    scale = physical_scale(task)
    col = lambda name: pd.to_numeric(
        task.get(name, pd.Series(0.0, index=task.index)), errors="coerce").fillna(0.0)

    phys = col("phys_complete_pct") / scale

    original = col("target_drtn_hr_cnt")
    remaining = col("remain_drtn_hr_cnt")
    drtn = np.where(original > 0, (original - remaining) / original, 0.0)

    act_units = col("act_work_qty") + col("act_equip_qty")
    rem_units = col("remain_work_qty") + col("remain_equip_qty")
    at_completion = act_units + rem_units
    units = np.where(at_completion > 0, act_units / at_completion, 0.0)

    kind = task.get("complete_pct_type", pd.Series("CP_Phys", index=task.index)).astype(str)
    pct = np.select([kind == "CP_Drtn", kind == "CP_Units", kind == "CP_Phys"],
                    [drtn, units, phys], default=phys)

    status = task.get("status_code", pd.Series("", index=task.index)).astype(str)
    pct = np.where(status == "TK_Complete", 1.0, pct)
    return pd.Series(np.clip(pct, 0.0, 1.0), index=task.index, name="pct_complete")


# --------------------------------------------------------------------------
# 2. Schedule % Complete — drives Planned Value
# --------------------------------------------------------------------------

def _calendar_hours(cal: Calendar, start: date, finish: date) -> float:
    if start is None or finish is None or finish < start:
        return 0.0
    total, cur = 0.0, start
    while cur <= finish:
        total += cal.hours_on(cur)
        cur += timedelta(days=1)
    return total


def schedule_percent(task: pd.DataFrame, calendars: dict[str, Calendar],
                     fallback: Calendar, data_date: pd.Timestamp) -> pd.Series:
    """
    Schedule % Complete, 0-1: the share of baseline duration elapsed at the
    data date, measured in working HOURS on the activity's own calendar.

      data date at or before baseline start   0%
      data date at or after  baseline finish  100%
      otherwise                               elapsed hours / baseline hours

    Planned Value is BAC x this figure.
    """
    if pd.isna(data_date):
        return pd.Series(0.0, index=task.index)
    dd = pd.Timestamp(data_date).date()
    out = np.zeros(len(task))

    starts = pd.to_datetime(task.get("target_start_date"), errors="coerce")
    finishes = pd.to_datetime(task.get("target_end_date"), errors="coerce")
    cal_ids = task.get("clndr_id", pd.Series("", index=task.index)).astype(str)

    for i, (cid, s, f) in enumerate(zip(cal_ids, starts, finishes)):
        if pd.isna(s) or pd.isna(f):
            continue
        s, f = s.date(), f.date()
        if dd <= s:
            continue
        if dd >= f:
            out[i] = 1.0
            continue
        cal = calendars.get(cid, fallback)
        total = _calendar_hours(cal, s, f)
        out[i] = min(1.0, _calendar_hours(cal, s, dd) / total) if total > 0 else 0.0

    return pd.Series(out, index=task.index, name="schedule_pct")


# --------------------------------------------------------------------------
# 3. Performance % Complete — the EV technique on the owning WBS
# --------------------------------------------------------------------------

def performance_percent(task: pd.DataFrame, projwbs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Performance % Complete, 0-1, per PROJWBS.ev_compute_type. Earned Value is
    BAC x this figure — not BAC x Activity % Complete, unless the WBS is set
    to the Activity % Complete technique.
    """
    technique = pd.Series(EV_ACTIVITY_PCT, index=task.index)
    user_pct = pd.Series(0.0, index=task.index)

    if projwbs is not None and not projwbs.empty and "ev_compute_type" in projwbs.columns:
        lookup = projwbs.set_index(projwbs["wbs_id"].astype(str))
        wid = task["wbs_id"].astype(str)
        technique = wid.map(lookup["ev_compute_type"].astype(str)).fillna(EV_ACTIVITY_PCT)
        if "ev_user_pct" in lookup.columns:
            user_pct = wid.map(pd.to_numeric(lookup["ev_user_pct"], errors="coerce")).fillna(0.0) / 100.0

    activity = task["pct_complete"].to_numpy()
    status = task.get("status_code", pd.Series("", index=task.index)).astype(str)
    started = (status != "TK_NotStart").to_numpy()
    complete = (status == "TK_Complete").to_numpy()

    zero_hundred = np.where(complete, 1.0, 0.0)
    fifty_fifty = np.where(complete, 1.0, np.where(started, 0.5, 0.0))

    t = technique.to_numpy().astype(str)
    perf = np.select([t == EV_0_100, t == EV_50_50, t == EV_USER_PCT],
                     [zero_hundred, fifty_fifty, user_pct.to_numpy()], default=activity)
    return (pd.Series(np.clip(perf, 0.0, 1.0), index=task.index, name="performance_pct"),
            technique.rename("ev_technique"))


# --------------------------------------------------------------------------
# 4. Activity frame
# --------------------------------------------------------------------------

def build_activities(tables: dict[str, pd.DataFrame], wbs: pd.DataFrame,
                     calendars: dict[str, Calendar] | None = None,
                     fallback: Calendar | None = None,
                     data_date: pd.Timestamp | None = None,
                     labour_divisor: float = 1.0) -> pd.DataFrame:
    """One row per activity carrying BAC, PV, EV, AC and the P6 techniques used."""
    task = tables.get("TASK", pd.DataFrame()).copy()
    if task.empty:
        return pd.DataFrame()

    for c in ("task_id", "wbs_id", "clndr_id"):
        task[c] = task[c].astype(str)
    task["pct_complete"] = activity_percent(task)

    tr = tables.get("TASKRSRC", pd.DataFrame()).copy()
    if not tr.empty:
        tr["task_id"] = tr["task_id"].astype(str)
        for c in ("target_cost", "act_reg_cost", "act_ot_cost", "remain_cost",
                  "target_qty", "act_reg_qty", "act_ot_qty", "remain_qty"):
            tr[c] = pd.to_numeric(tr.get(c, 0.0), errors="coerce").fillna(0.0)
        tr["ac"] = tr["act_reg_cost"] + tr["act_ot_cost"]
        rtype = tr.get("rsrc_type", pd.Series("", index=tr.index)).astype(str)
        div = max(labour_divisor, 1e-9)
        tr["labour_hr"] = np.where(rtype == LABOUR, tr["target_qty"], 0.0) / div
        tr["equip_hr"] = np.where(rtype == EQUIP, tr["target_qty"], 0.0) / div
        tr["act_labour_hr"] = np.where(
            rtype == LABOUR, tr["act_reg_qty"] + tr["act_ot_qty"], 0.0) / div

        agg = tr.groupby("task_id").agg(
            bac=("target_cost", "sum"), ac=("ac", "sum"),
            remain_cost=("remain_cost", "sum"), labour_hr=("labour_hr", "sum"),
            equip_hr=("equip_hr", "sum"), act_labour_hr=("act_labour_hr", "sum"))
        task = task.merge(agg, left_on="task_id", right_index=True, how="left")

    for c in ("bac", "ac", "remain_cost", "labour_hr", "equip_hr", "act_labour_hr"):
        task[c] = pd.to_numeric(task.get(c, 0.0), errors="coerce").fillna(0.0)

    projwbs = tables.get("PROJWBS", pd.DataFrame())
    perf, technique = performance_percent(task, projwbs)
    task["performance_pct"] = perf
    task["ev_technique"] = technique
    task["ev"] = task["bac"] * task["performance_pct"]
    task["earned_hr"] = task["labour_hr"] * task["performance_pct"]

    if calendars is not None and fallback is not None and data_date is not None:
        task["schedule_pct"] = schedule_percent(task, calendars, fallback, data_date)
    else:
        task["schedule_pct"] = 0.0
    task["pv"] = task["bac"] * task["schedule_pct"]

    if not wbs.empty:
        lineage = wbs.set_index("wbs_id")
        task = task.merge(lineage[["wbs_name", "path", "level", "ancestors"]],
                          left_on="wbs_id", right_index=True, how="left", suffixes=("", "_wbs"))
        name_of = dict(zip(wbs["wbs_id"], wbs["wbs_name"].astype(str)))
        top = {wid: (name_of.get(ch[1]) if len(ch) > 1 else name_of.get(ch[0], "—"))
               for wid, ch in zip(wbs["wbs_id"], wbs["ancestors"])}
        task["wbs_main"] = task["wbs_id"].map(top).fillna("—")
    else:
        task["wbs_name"] = task["wbs_main"] = task["path"] = "—"

    task["total_float_hr_cnt"] = pd.to_numeric(
        task.get("total_float_hr_cnt", 0), errors="coerce").fillna(0)

    task["forecast_finish"] = task.get("reend_date")
    for col in ("early_end_date", "act_end_date"):
        task["forecast_finish"] = task["forecast_finish"].fillna(task.get(col))
    task["forecast_start"] = task.get("restart_date")
    for col in ("early_start_date", "act_start_date"):
        task["forecast_start"] = task["forecast_start"].fillna(task.get(col))

    return task


def float_days(task: pd.DataFrame, calendars: dict[str, Calendar], fallback_hpd: float = 8.0) -> pd.Series:
    hpd = task["clndr_id"].map({k: v.hours_per_day for k, v in calendars.items()}).fillna(fallback_hpd)
    return task["total_float_hr_cnt"] / hpd


# --------------------------------------------------------------------------
# 5. Time phasing — reconciles to the P6 figures at the data date
# --------------------------------------------------------------------------

def _spread_hours(cal: Calendar, start, finish, value: float) -> dict[date, float]:
    """Spread a value across working HOURS, so the curve matches Schedule %."""
    if value == 0 or pd.isna(start) or pd.isna(finish):
        return {}
    s, f = pd.Timestamp(start).date(), pd.Timestamp(finish).date()
    if f < s:
        f = s
    hours, cur = {}, s
    while cur <= f:
        h = cal.hours_on(cur)
        if h > 0:
            hours[cur] = h
        cur += timedelta(days=1)
    total = sum(hours.values())
    if total <= 0:
        return {s: value}
    return {d: value * h / total for d, h in hours.items()}


def time_phase(activities: pd.DataFrame, calendars: dict[str, Calendar],
               fallback: Calendar, data_date: pd.Timestamp) -> pd.DataFrame:
    """
    Daily PV / EV / AC / forecast. PV is spread over baseline dates by working
    hours, so cumulative PV at the data date equals the sum of BAC x Schedule %
    that P6 reports. The earned curve before the data date is a reconstruction —
    a single XER holds no progress history.
    """
    pv: dict[date, float] = {}
    ev: dict[date, float] = {}
    ac: dict[date, float] = {}
    fc: dict[date, float] = {}
    dd = pd.Timestamp(data_date).date() if pd.notna(data_date) else date.today()

    for row in activities.itertuples(index=False):
        cal = calendars.get(getattr(row, "clndr_id", ""), fallback)
        bac = float(getattr(row, "bac", 0.0) or 0.0)
        if bac <= 0:
            continue

        for d, v in _spread_hours(cal, getattr(row, "target_start_date", None),
                                  getattr(row, "target_end_date", None), bac).items():
            pv[d] = pv.get(d, 0.0) + v

        earned = float(getattr(row, "ev", 0.0) or 0.0)
        actual = float(getattr(row, "ac", 0.0) or 0.0)
        a_start = getattr(row, "act_start_date", None)
        a_end = getattr(row, "act_end_date", None)
        if earned > 0:
            e_start = a_start if pd.notna(a_start) else getattr(row, "target_start_date", None)
            e_end = a_end if pd.notna(a_end) else pd.Timestamp(dd)
            for d, v in _spread_hours(cal, e_start, e_end, earned).items():
                if d <= dd:
                    ev[d] = ev.get(d, 0.0) + v
            for d, v in _spread_hours(cal, e_start, e_end, actual).items():
                if d <= dd:
                    ac[d] = ac.get(d, 0.0) + v

        remaining = float(getattr(row, "remain_cost", 0.0) or 0.0)
        if remaining > 0:
            f_start = getattr(row, "forecast_start", None)
            f_start = max(pd.Timestamp(f_start).date(), dd) if pd.notna(f_start) else dd
            f_end = getattr(row, "forecast_finish", None)
            f_end = pd.Timestamp(f_end).date() if pd.notna(f_end) else f_start
            for d, v in _spread_hours(cal, f_start, max(f_end, f_start), remaining).items():
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
    if daily.empty:
        return daily
    rule = {"W": "W-THU", "M": "ME", "Q": "QE"}.get(freq, "ME")
    g = daily.set_index("day").resample(rule).sum(numeric_only=True).reset_index()
    g = g.rename(columns={"day": "period"})
    for c in ("pv", "ev", "ac", "fcst"):
        g[f"cum_{c}"] = g[c].cumsum()
    ev_final = g.loc[g["ev"] > 0, "cum_ev"].max() if (g["ev"] > 0).any() else 0.0
    g["cum_fcst"] = np.where(g["cum_fcst"] > 0, g["cum_fcst"] + ev_final, np.nan)
    return g


# --------------------------------------------------------------------------
# 6. Forecast — per the ETC technique recorded on the WBS
# --------------------------------------------------------------------------

def forecast(activities: pd.DataFrame, projwbs: pd.DataFrame,
             bac: float, ev: float, ac: float, cpi: float | None) -> dict:
    """
    ETC and EAC the way P6 computes them, per PROJWBS.ev_etc_compute_type.

      EE_Rem_hr   ETC = Remaining Cost                  EAC = AC + ETC
      EE_Pf_1     ETC = BAC - EV
      EE_Pf_cpi   ETC = (BAC - EV) / CPI
      EE_Pf_udf   ETC = (BAC - EV) / user performance factor

    Using BAC / CPI where the file says EE_Rem_hr is the most common reason a
    dashboard's EAC disagrees with P6.
    """
    technique, user_pf = ETC_REMAINING, 1.0
    if projwbs is not None and not projwbs.empty and "ev_etc_compute_type" in projwbs.columns:
        modes = projwbs["ev_etc_compute_type"].astype(str).value_counts()
        if not modes.empty:
            technique = modes.index[0]
        if "ev_etc_user_value" in projwbs.columns:
            vals = pd.to_numeric(projwbs["ev_etc_user_value"], errors="coerce").dropna()
            vals = vals[vals > 0]
            if not vals.empty:
                user_pf = float(vals.mode().iloc[0])

    remaining_cost = float(activities["remain_cost"].sum()) if "remain_cost" in activities else 0.0
    if technique == ETC_REMAINING:
        etc = remaining_cost
    elif technique == ETC_PF_CPI:
        etc = (bac - ev) / cpi if cpi else None
    elif technique == ETC_PF_USER:
        etc = (bac - ev) / user_pf if user_pf else None
    else:
        etc = bac - ev

    eac = (ac + etc) if etc is not None else None
    return {"etc_technique": technique, "etc_user_pf": user_pf, "etc": etc,
            "eac": eac, "vac": (bac - eac) if eac is not None else None}


def earned_schedule(periods: pd.DataFrame, ev_now: float, data_date: pd.Timestamp) -> dict:
    """
    Earned Schedule. Not a P6 field — an ANSI EIA-748 extension expressing
    schedule variance in time rather than money, reported alongside SPI because
    cost-based SPI drifts to 1.0 near completion however late the work is.
    """
    blank = {"es": None, "es_months": None, "at": None, "sv_t": None, "spi_t": None}
    if periods.empty or ev_now <= 0 or pd.isna(data_date):
        return blank
    curve = periods[["period", "cum_pv"]].dropna()
    if curve.empty or curve["cum_pv"].max() < ev_now:
        return blank
    start = curve["period"].iloc[0]
    below, above = curve[curve["cum_pv"] <= ev_now], curve[curve["cum_pv"] > ev_now]
    if above.empty:
        return blank
    if below.empty:
        es_date, es_months = start, 0.0
    else:
        lo, hi = below.iloc[-1], above.iloc[0]
        span = hi["cum_pv"] - lo["cum_pv"]
        frac = (ev_now - lo["cum_pv"]) / span if span > 0 else 0.0
        es_date = lo["period"] + (hi["period"] - lo["period"]) * frac
        es_months = (es_date - start).days / 30.4375
    at = (pd.Timestamp(data_date) - start).days / 30.4375
    if at <= 0:
        return blank
    return {"es": es_date, "es_months": es_months, "at": at,
            "sv_t": es_months - at, "spi_t": es_months / at}


def summarise(activities: pd.DataFrame, periods: pd.DataFrame,
              data_date: pd.Timestamp, projwbs: pd.DataFrame | None = None) -> dict:
    """
    Headline figures. PV and EV are summed from the per-activity P6 values, not
    read off the curve, so they match P6 column for column.
    """
    bac = float(activities["bac"].sum())
    ev = float(activities["ev"].sum())
    ac = float(activities["ac"].sum())
    pv = float(activities["pv"].sum()) if "pv" in activities else 0.0

    ratio = lambda a, b: (a / b) if b else None
    spi, cpi = ratio(ev, pv), ratio(ev, ac)
    fc = forecast(activities, projwbs, bac, ev, ac, cpi)
    es = earned_schedule(periods, ev, data_date)

    return {
        "bac": bac, "pv": pv, "ev": ev, "ac": ac,
        "schedule_pct": ratio(pv, bac),      # P6 Schedule % Complete, cost weighted
        "performance_pct": ratio(ev, bac),   # P6 Performance % Complete
        "sv": ev - pv, "cv": ev - ac, "spi": spi, "cpi": cpi,
        "tcpi": ratio(bac - ev, bac - ac) if (bac - ac) else None,
        **fc,
        **{f"es_{k}": v for k, v in es.items()},
    }


def rollup(activities: pd.DataFrame, key: str) -> pd.DataFrame:
    """BAC-weighted rollup. PV is summed from activity PV, matching P6."""
    if activities.empty:
        return pd.DataFrame()
    g = activities.groupby(key, dropna=False).agg(
        bac=("bac", "sum"), pv=("pv", "sum"), ev=("ev", "sum"), ac=("ac", "sum"),
        labour_hr=("labour_hr", "sum"), earned_hr=("earned_hr", "sum"),
        activities=("task_id", "count")).reset_index()
    g["planned_pct"] = np.where(g["bac"] > 0, g["pv"] / g["bac"], 0.0)
    g["earned_pct"] = np.where(g["bac"] > 0, g["ev"] / g["bac"], 0.0)
    g["variance_pct"] = g["earned_pct"] - g["planned_pct"]
    g["spi"] = np.where(g["pv"] > 0, g["ev"] / g["pv"], np.nan)
    g["cpi"] = np.where(g["ac"] > 0, g["ev"] / g["ac"], np.nan)
    return g.sort_values("bac", ascending=False)
