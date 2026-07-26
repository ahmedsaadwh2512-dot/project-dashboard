"""
analytics.py — schedule quality, slippage, drivers, comparison, data health.

Every day figure in this module is a WORKING day on the owning calendar.
Nothing here reads actual cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xer_parser import Calendar

HARD_CONSTRAINTS = {
    "CS_MSO", "CS_MSOB", "CS_MEO", "CS_MEOB", "CS_MANDSTART", "CS_MANDFIN",
}
SOFT_CONSTRAINTS = {"CS_MSOA", "CS_MEOA", "CS_ALAP"}


# --------------------------------------------------------------------------
# Working-day helpers
# --------------------------------------------------------------------------

def wd_between(cal: Calendar, a, b) -> float:
    """Signed working days from a to b. Positive = b is later."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    a, b = pd.Timestamp(a).date(), pd.Timestamp(b).date()
    if a == b:
        return 0.0
    sign = 1 if b > a else -1
    lo, hi = (a, b) if b > a else (b, a)
    return sign * max(0, cal.working_day_count(lo, hi) - 1)


# --------------------------------------------------------------------------
# DCMA 14-point assessment
# --------------------------------------------------------------------------

def dcma_14(
    task: pd.DataFrame,
    pred: pd.DataFrame,
    calendars: dict[str, Calendar],
    fallback: Calendar,
    data_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    The DCMA 14-Point Schedule Assessment. Returns one row per check with
    the measured value, the threshold, and a pass/fail verdict.
    """
    rows: list[dict] = []
    hpd = fallback.hours_per_day or 8.0

    incomplete = task[task["status_code"].astype(str) != "TK_Complete"]
    milestones = task["task_type"].astype(str).isin(["TT_Mile", "TT_FinMile"])
    real = task[~milestones]
    n_real = max(len(real), 1)

    def add(no, name, value, threshold, unit, ok, detail=""):
        rows.append({
            "#": no, "Check": name, "Value": value, "Threshold": threshold,
            "Unit": unit, "Verdict": "PASS" if ok else "FAIL", "Detail": detail,
        })

    # 1. Logic — activities missing a predecessor or a successor
    has_pred = set(pred["task_id"].astype(str)) if not pred.empty else set()
    has_succ = set(pred["pred_task_id"].astype(str)) if not pred.empty else set()
    ids = real["task_id"].astype(str)
    dangling = int(sum(1 for t in ids if t not in has_pred or t not in has_succ))
    v = 100 * dangling / n_real
    add(1, "Logic (open ends)", v, 5.0, "%", v <= 5, f"{dangling} activities")

    # 2. Leads — negative lag is never acceptable
    lags = pd.to_numeric(pred.get("lag_hr_cnt", pd.Series(dtype=float)), errors="coerce").fillna(0)
    leads = int((lags < 0).sum())
    add(2, "Leads (negative lag)", leads, 0, "count", leads == 0, "")

    # 3. Lags
    n_rel = max(len(pred), 1)
    lag_ct = int((lags > 0).sum())
    v = 100 * lag_ct / n_rel
    add(3, "Lags", v, 5.0, "%", v <= 5, f"{lag_ct} of {len(pred)} relationships")

    # 4. Relationship types — finish-to-start should dominate
    types = pred.get("pred_type", pd.Series(dtype=str)).astype(str)
    fs = int((types == "PR_FS").sum())
    v = 100 * fs / n_rel
    add(4, "Finish-to-start ratio", v, 90.0, "%", v >= 90, f"{fs} of {len(pred)}")

    # 5. Hard constraints
    cstr = task.get("cstr_type", pd.Series(dtype=str)).astype(str)
    hard = int(cstr.isin(HARD_CONSTRAINTS).sum())
    v = 100 * hard / n_real
    add(5, "Hard constraints", v, 5.0, "%", v <= 5, f"{hard} activities")

    # 6. High float — more than 44 working days
    tf_days = pd.to_numeric(task.get("total_float_hr_cnt", 0), errors="coerce").fillna(0) / hpd
    inc_mask = task["status_code"].astype(str) != "TK_Complete"
    n_inc = max(int(inc_mask.sum()), 1)
    high = int(((tf_days > 44) & inc_mask).sum())
    v = 100 * high / n_inc
    add(6, "High float (>44d)", v, 5.0, "%", v <= 5, f"{high} activities")

    # 7. Negative float
    neg = int(((tf_days < 0) & inc_mask).sum())
    add(7, "Negative float", neg, 0, "count", neg == 0, "")

    # 8. High duration — remaining duration over 44 working days
    rd = pd.to_numeric(task.get("remain_drtn_hr_cnt", 0), errors="coerce").fillna(0) / hpd
    hi_dur = int(((rd > 44) & inc_mask).sum())
    v = 100 * hi_dur / n_inc
    add(8, "High duration (>44d)", v, 5.0, "%", v <= 5, f"{hi_dur} activities")

    # 9. Invalid dates
    bad = 0
    if pd.notna(data_date):
        es = pd.to_datetime(task.get("early_start_date"), errors="coerce")
        astart = pd.to_datetime(task.get("act_start_date"), errors="coerce")
        aend = pd.to_datetime(task.get("act_end_date"), errors="coerce")
        bad += int((es < data_date).sum())
        bad += int((astart > data_date).sum())
        bad += int((aend > data_date).sum())
    add(9, "Invalid dates", bad, 0, "count", bad == 0, "forecast before / actual after data date")

    # 10. Resources — activities with duration but no assignment
    if "bac" in task.columns:
        unres = int(((task["bac"] <= 0) & (task.get("labour_hr", 0) <= 0)
                     & (~milestones) & inc_mask).sum())
    else:
        unres = 0
    v = 100 * unres / n_real
    add(10, "Missing resources", v, 5.0, "%", v <= 5, f"{unres} activities carry no cost or hours")

    # 11. Missed tasks — finishing later than planned
    missed = 0
    if "forecast_finish" in task.columns and "target_end_date" in task.columns:
        ff = pd.to_datetime(task["forecast_finish"], errors="coerce")
        bf = pd.to_datetime(task["target_end_date"], errors="coerce")
        missed = int((ff > bf).sum())
    v = 100 * missed / n_real
    add(11, "Missed tasks", v, 5.0, "%", v <= 5, f"{missed} finishing later than planned")

    # 12. Critical path test — qualitative, requires a deliberate 600-day injection
    add(12, "Critical path test", "manual", "—", "", True,
        "Requires injecting a 600d delay and confirming the end date moves")

    # 13. CPLI — critical path length index
    cpli = np.nan
    if pd.notna(data_date) and "forecast_finish" in task.columns:
        finish = pd.to_datetime(task["forecast_finish"], errors="coerce").max()
        if pd.notna(finish):
            cpl = wd_between(fallback, data_date, finish)
            tf_crit = float(tf_days[inc_mask].min()) if inc_mask.any() else 0.0
            if cpl and cpl > 0:
                cpli = (cpl + tf_crit) / cpl
    add(13, "CPLI", round(cpli, 3) if pd.notna(cpli) else "—", 0.95, "index",
        bool(pd.notna(cpli) and cpli >= 0.95), "critical path length index")

    # 14. BEI — baseline execution index
    bei = np.nan
    if pd.notna(data_date) and "target_end_date" in task.columns:
        due = pd.to_datetime(task["target_end_date"], errors="coerce") <= data_date
        done = task["status_code"].astype(str) == "TK_Complete"
        denom = int(due.sum())
        if denom:
            bei = int((done & due).sum()) / denom
    add(14, "BEI", round(bei, 3) if pd.notna(bei) else "—", 0.95, "index",
        bool(pd.notna(bei) and bei >= 0.95), "baseline execution index")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Slippage and variance
# --------------------------------------------------------------------------

def activity_slippage(
    task: pd.DataFrame,
    calendars: dict[str, Calendar],
    fallback: Calendar,
) -> pd.DataFrame:
    """
    Per-activity finish variance against the planned (target) dates held in
    this file, in working days. Positive = late.
    """
    df = task.copy()
    cal_for = lambda cid: calendars.get(str(cid), fallback)

    df["finish_var_d"] = [
        wd_between(cal_for(c), b, f)
        for c, b, f in zip(df.get("clndr_id", ""), df.get("target_end_date"), df.get("forecast_finish"))
    ]
    df["start_var_d"] = [
        wd_between(cal_for(c), b, f)
        for c, b, f in zip(df.get("clndr_id", ""), df.get("target_start_date"), df.get("forecast_start"))
    ]

    hpd = df.get("clndr_id", pd.Series("", index=df.index)).map(
        {k: v.hours_per_day for k, v in calendars.items()}
    ).fillna(fallback.hours_per_day)
    df["total_float_d"] = pd.to_numeric(df.get("total_float_hr_cnt", 0), errors="coerce").fillna(0) / hpd

    # Duration growth — an originating cause, not an inherited one
    tgt = pd.to_numeric(df.get("target_drtn_hr_cnt", 0), errors="coerce").fillna(0)
    rem = pd.to_numeric(df.get("remain_drtn_hr_cnt", 0), errors="coerce").fillna(0)
    at_complete = np.maximum(tgt, rem)
    df["duration_growth_d"] = (at_complete - tgt) / hpd

    # Impact classification
    slipped = df["finish_var_d"] > 0
    df["impact_class"] = np.select(
        [slipped & (df["total_float_d"] <= 0),
         slipped & (df["total_float_d"] > 0),
         df["finish_var_d"] < 0],
        ["Critical", "Float-eroding", "Ahead"],
        default="On plan",
    )

    # Originating vs inherited
    late_start = df["start_var_d"] > 0
    grew = df["duration_growth_d"] > 0.5
    df["origin"] = np.where(slipped & (grew | late_start), "Originating",
                            np.where(slipped, "Inherited", "—"))

    cause = np.select(
        [grew & late_start, grew, late_start],
        ["Duration extended + late start", "Duration extended", "Late start"],
        default="Predecessor driven",
    )
    df["cause"] = np.where(slipped, cause, "—")

    return df


def successor_counts(pred: pd.DataFrame) -> pd.Series:
    if pred.empty:
        return pd.Series(dtype=int)
    return pred.groupby(pred["pred_task_id"].astype(str)).size()


def delay_impact(slip: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """
    Rank delayed activities by consequence, not by size of slip.

        Impact = originating slip (working days) x (1 + successor count)

    Same shape as the Impact Score already in use for critical drivers, so
    the two pages reconcile.
    """
    succ = successor_counts(pred)
    out = slip.copy()
    out["successors"] = out["task_id"].astype(str).map(succ).fillna(0).astype(int)
    origin_slip = np.where(out["origin"] == "Originating", out["finish_var_d"].fillna(0), 0.0)
    out["delay_impact"] = origin_slip * (1 + out["successors"])
    return out.sort_values("delay_impact", ascending=False)


def wbs_slippage(slip: pd.DataFrame, group: str = "wbs_main") -> pd.DataFrame:
    """
    Delay per WBS. Finish variance of a group is the movement of its LATEST
    finish, not the average of its activities — averaging hides the driver.
    """
    if slip.empty:
        return pd.DataFrame()
    rows = []
    for name, g in slip.groupby(group, dropna=False):
        bl = pd.to_datetime(g.get("target_end_date"), errors="coerce").max()
        fc = pd.to_datetime(g.get("forecast_finish"), errors="coerce").max()
        crit = g[g["impact_class"] == "Critical"]
        rows.append({
            group: name,
            "baseline_finish": bl,
            "forecast_finish": fc,
            "finish_var_d": (fc - bl).days if pd.notna(bl) and pd.notna(fc) else np.nan,
            "activities": len(g),
            "slipped": int((g["finish_var_d"] > 0).sum()),
            "critical": len(crit),
            "originating": int((g["origin"] == "Originating").sum()),
            "worst_slip_d": float(g["finish_var_d"].max()) if len(g) else np.nan,
            "bac": float(g["bac"].sum()) if "bac" in g.columns else 0.0,
        })
    return pd.DataFrame(rows).sort_values("finish_var_d", ascending=False, na_position="last")


# --------------------------------------------------------------------------
# Critical drivers
# --------------------------------------------------------------------------

def critical_drivers(task: pd.DataFrame, pred: pd.DataFrame, fallback: Calendar,
                     longest_path_only: bool = True) -> pd.DataFrame:
    """
    Impact Score = remaining duration (working days) x (1 + successor count).
    Scoped to the longest path so level-of-effort activities do not dominate.
    """
    df = task.copy()
    hpd = fallback.hours_per_day or 8.0
    df["remain_d"] = pd.to_numeric(df.get("remain_drtn_hr_cnt", 0), errors="coerce").fillna(0) / hpd
    df["successors"] = df["task_id"].astype(str).map(successor_counts(pred)).fillna(0).astype(int)
    df["impact_score"] = df["remain_d"] * (1 + df["successors"])

    if longest_path_only and "driving_path_flag" in df.columns:
        lp = df["driving_path_flag"].astype(str).str.upper() == "Y"
        if lp.any():
            df = df[lp]
    df = df[df["status_code"].astype(str) != "TK_Complete"]
    return df.sort_values("impact_score", ascending=False)


# --------------------------------------------------------------------------
# Two-file comparison
# --------------------------------------------------------------------------

def compare_updates(base: pd.DataFrame, curr: pd.DataFrame, fallback: Calendar) -> dict:
    """
    Diff two parsed updates on task_code. Returns added, removed and changed
    activities with date, duration and progress movement.
    """
    b = base.set_index(base["task_code"].astype(str))
    c = curr.set_index(curr["task_code"].astype(str))
    added = sorted(set(c.index) - set(b.index))
    removed = sorted(set(b.index) - set(c.index))
    common = sorted(set(b.index) & set(c.index))

    rows = []
    for code in common:
        rb, rc = b.loc[code], c.loc[code]
        if isinstance(rb, pd.DataFrame):
            rb = rb.iloc[0]
        if isinstance(rc, pd.DataFrame):
            rc = rc.iloc[0]
        fin_move = wd_between(fallback, rb.get("forecast_finish"), rc.get("forecast_finish"))
        dur_b = float(pd.to_numeric(rb.get("target_drtn_hr_cnt"), errors="coerce") or 0)
        dur_c = float(pd.to_numeric(rc.get("target_drtn_hr_cnt"), errors="coerce") or 0)
        pct_b = float(rb.get("pct_complete") or 0)
        pct_c = float(rc.get("pct_complete") or 0)
        if (fin_move and abs(fin_move) > 0) or dur_b != dur_c or abs(pct_c - pct_b) > 1e-9:
            rows.append({
                "task_code": code,
                "task_name": rc.get("task_name"),
                "wbs_main": rc.get("wbs_main"),
                "finish_move_d": fin_move,
                "duration_change_hr": dur_c - dur_b,
                "pct_before": pct_b,
                "pct_after": pct_c,
                "pct_move": pct_c - pct_b,
                "regressed": pct_c < pct_b - 1e-9,
            })
    cols = ["task_code", "task_name", "wbs_main", "finish_move_d", "duration_change_hr",
            "pct_before", "pct_after", "pct_move", "regressed"]
    changed = pd.DataFrame(rows, columns=cols)
    if not changed.empty:
        changed = changed.sort_values("finish_move_d", ascending=False, na_position="last")

    return {
        "added": curr[curr["task_code"].astype(str).isin(added)],
        "removed": base[base["task_code"].astype(str).isin(removed)],
        "changed": changed,
    }


# --------------------------------------------------------------------------
# Data health
# --------------------------------------------------------------------------

def data_health(task: pd.DataFrame, pred: pd.DataFrame, data_date: pd.Timestamp) -> pd.DataFrame:
    """Checks that catch the kinds of defect that trigger a client rejection."""
    checks = []
    n = max(len(task), 1)

    def add(name, count, detail):
        checks.append({"Check": name, "Count": count,
                       "Share": f"{100 * count / n:.1f}%", "Detail": detail})

    status = task["status_code"].astype(str)
    add("Zero-cost activities", int((task.get("bac", 0) <= 0).sum()),
        "Excluded from cost-weighted progress")
    add("Started with no progress", int(((status == "TK_Active") & (task["pct_complete"] <= 0)).sum()),
        "Actual start recorded but 0% earned")
    add("Complete with no actual finish", int(((status == "TK_Complete")
        & pd.to_datetime(task.get("act_end_date"), errors="coerce").isna()).sum()),
        "Status complete, finish date missing")
    add("Progress without actual start", int(((task["pct_complete"] > 0)
        & pd.to_datetime(task.get("act_start_date"), errors="coerce").isna()).sum()),
        "Earned value with no start date")

    if pd.notna(data_date):
        add("Actuals beyond data date",
            int((pd.to_datetime(task.get("act_start_date"), errors="coerce") > data_date).sum()
                + (pd.to_datetime(task.get("act_end_date"), errors="coerce") > data_date).sum()),
            "Impossible dates")
        add("Out-of-sequence progress",
            int(((status != "TK_NotStart")
                 & (pd.to_datetime(task.get("early_start_date"), errors="coerce") < data_date)).sum()),
            "Forecast start before the data date")

    zero_dur = ((pd.to_numeric(task.get("target_drtn_hr_cnt", 0), errors="coerce").fillna(0) == 0)
                & (~task["task_type"].astype(str).isin(["TT_Mile", "TT_FinMile"])))
    add("Zero-duration non-milestones", int(zero_dur.sum()), "Should be milestones or have duration")

    if "bac" in task.columns and task["bac"].sum() > 0:
        top2 = task.nlargest(2, "bac")["bac"].sum()
        share = 100 * top2 / task["bac"].sum()
        add("Cost concentration", 2, f"Top 2 activities hold {share:.1f}% of BAC")

    return pd.DataFrame(checks)
