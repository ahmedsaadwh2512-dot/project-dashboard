"""
xer_parser.py — Primavera P6 XER reader.

Reads an .xer file into typed pandas DataFrames, one per P6 table, and parses
the CALENDAR blob into usable working-day rules.

Design notes
------------
* encoding='utf-8-sig' — XER files carry a BOM that corrupts the first field name.
* Durations and float in XER are stored in HOURS. Conversion to days is done
  against the owning calendar's hours-per-day, never a hard-coded 8.
* Nothing here computes progress or money. Parsing stays separate from EVM so
  the calculations can be unit-tested against P6 output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

# --------------------------------------------------------------------------
# Field typing
# --------------------------------------------------------------------------

DATE_SUFFIXES = ("_date",)
NUMERIC_HINTS = (
    "_cnt", "_qty", "_cost", "_pct", "_num", "_flag_num", "_id",
    "day_hr_cnt", "week_hr_cnt", "month_hr_cnt", "year_hr_cnt",
)
# Columns that look numeric by suffix but must stay strings (identifiers used as keys)
FORCE_STR = {"task_code", "wbs_short_name", "rsrc_short_name", "proj_short_name"}

# P6 day index -> python weekday() index
_P6_DAY_TO_PY = {1: 6, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
_P6_EPOCH = date(1899, 12, 30)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in FORCE_STR:
            df[col] = df[col].astype("string")
            continue
        if col.endswith(DATE_SUFFIXES):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif any(col.endswith(h) for h in NUMERIC_HINTS):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_xer(path_or_buffer) -> dict[str, pd.DataFrame]:
    """Parse an XER file into {table_name: DataFrame}."""
    if hasattr(path_or_buffer, "read"):
        raw = path_or_buffer.read()
        text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
    else:
        with open(path_or_buffer, "r", encoding="utf-8-sig", errors="replace") as fh:
            text = fh.read()

    tables: dict[str, pd.DataFrame] = {}
    current_name: str | None = None
    current_fields: list[str] = []
    current_rows: list[list[str]] = []

    def flush() -> None:
        if current_name and current_fields:
            width = len(current_fields)
            norm = [r[:width] + [""] * (width - len(r)) for r in current_rows]
            tables[current_name] = _coerce(pd.DataFrame(norm, columns=current_fields))

    for line in text.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        tag = parts[0]
        if tag == "%T":
            flush()
            current_name = parts[1].strip() if len(parts) > 1 else None
            current_fields, current_rows = [], []
        elif tag == "%F":
            current_fields = [p.strip() for p in parts[1:]]
        elif tag == "%R":
            current_rows.append(parts[1:])
        elif tag == "%E":
            flush()
            current_name, current_fields, current_rows = None, [], []
    flush()
    return tables


# --------------------------------------------------------------------------
# Calendars
# --------------------------------------------------------------------------

@dataclass
class Calendar:
    """Working-time rules for one P6 calendar."""

    clndr_id: str
    name: str
    day_hours: dict[int, float] = field(default_factory=dict)   # python weekday -> hours
    holidays: set[date] = field(default_factory=set)
    exception_hours: dict[date, float] = field(default_factory=dict)

    @property
    def hours_per_day(self) -> float:
        working = [h for h in self.day_hours.values() if h > 0]
        return sum(working) / len(working) if working else 8.0

    @property
    def days_per_week(self) -> int:
        return sum(1 for h in self.day_hours.values() if h > 0)

    def hours_on(self, day: date) -> float:
        if day in self.exception_hours:
            return self.exception_hours[day]
        if day in self.holidays:
            return 0.0
        return self.day_hours.get(day.weekday(), 0.0)

    def is_working(self, day: date) -> bool:
        return self.hours_on(day) > 0

    def working_days(self, start: date, finish: date) -> list[date]:
        if start is None or finish is None or finish < start:
            return []
        out, cur = [], start
        while cur <= finish:
            if self.is_working(cur):
                out.append(cur)
            cur += timedelta(days=1)
        return out

    def working_day_count(self, start: date, finish: date) -> int:
        return len(self.working_days(start, finish))


def _balanced(text: str, open_idx: int) -> tuple[str, int]:
    """Return the contents of the parenthesis group starting at open_idx."""
    depth, i = 0, open_idx
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i], i + 1
        i += 1
    return "", len(text)


_SHIFT_RE = re.compile(r"s\|(\d{1,2}):(\d{2})\|f\|(\d{1,2}):(\d{2})")


def _shift_hours(block: str) -> float:
    total = 0.0
    for sh, sm, fh, fm in _SHIFT_RE.findall(block):
        start = int(sh) + int(sm) / 60
        finish = int(fh) + int(fm) / 60
        if finish < start:          # shift crosses midnight
            finish += 24
        total += finish - start
    return total


def parse_calendar_blob(clndr_id: str, name: str, blob: str, default_day_hrs: float = 8.0) -> Calendar:
    """Decode the CALENDAR.clndr_data string into a Calendar."""
    cal = Calendar(clndr_id=str(clndr_id), name=str(name))
    if not isinstance(blob, str) or not blob:
        cal.day_hours = {d: default_day_hrs for d in range(5)}   # Mon-Fri fallback
        return cal

    text = blob.replace("\r", "").replace("\n", "")

    # --- DaysOfWeek -------------------------------------------------------
    anchor = text.find("DaysOfWeek")
    if anchor != -1:
        open_idx = text.find("(", text.find("(", anchor) + 1)
        section, _ = _balanced(text, open_idx) if open_idx != -1 else ("", 0)
        for m in re.finditer(r"0\|\|(\d)\(\)", section):
            p6_day = int(m.group(1))
            grp_start = section.find("(", m.end())
            body, _ = _balanced(section, grp_start) if grp_start != -1 else ("", 0)
            hours = _shift_hours(body)
            py_day = _P6_DAY_TO_PY.get(p6_day)
            if py_day is not None:
                cal.day_hours[py_day] = hours

    if not any(h > 0 for h in cal.day_hours.values()):
        cal.day_hours = {d: default_day_hrs for d in range(5)}

    # --- Exceptions -------------------------------------------------------
    anchor = text.find("Exceptions")
    if anchor != -1:
        open_idx = text.find("(", text.find("(", anchor) + 1)
        section, _ = _balanced(text, open_idx) if open_idx != -1 else ("", 0)
        for m in re.finditer(r"0\|\|\d+\(d\|(\d+)\)", section):
            serial = int(m.group(1))
            day = _P6_EPOCH + timedelta(days=serial)
            grp_start = section.find("(", m.end())
            body, _ = _balanced(section, grp_start) if grp_start != -1 else ("", 0)
            hours = _shift_hours(body)
            if hours > 0:
                cal.exception_hours[day] = hours
            else:
                cal.holidays.add(day)

    return cal


def build_calendars(tables: dict[str, pd.DataFrame]) -> dict[str, Calendar]:
    """Build {clndr_id: Calendar} from the CALENDAR table."""
    out: dict[str, Calendar] = {}
    df = tables.get("CALENDAR")
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        default = float(row.get("day_hr_cnt") or 8.0)
        cal = parse_calendar_blob(
            row.get("clndr_id"),
            row.get("clndr_name", ""),
            row.get("clndr_data", ""),
            default_day_hrs=default,
        )
        out[str(row.get("clndr_id"))] = cal
    return out


def default_calendar(calendars: dict[str, Calendar], tables: dict[str, pd.DataFrame]) -> Calendar:
    """Pick the calendar used by the most activities — the project's real working pattern."""
    task = tables.get("TASK")
    if task is not None and "clndr_id" in task.columns and not task.empty:
        top = task["clndr_id"].astype(str).value_counts()
        for cid in top.index:
            if cid in calendars:
                return calendars[cid]
    if calendars:
        return next(iter(calendars.values()))
    return Calendar(clndr_id="-", name="Fallback 5-day", day_hours={d: 8.0 for d in range(5)})


# --------------------------------------------------------------------------
# Project header
# --------------------------------------------------------------------------

@dataclass
class ProjectHeader:
    proj_id: str
    short_name: str
    data_date: pd.Timestamp | None
    plan_start: pd.Timestamp | None
    plan_finish: pd.Timestamp | None
    scd_finish: pd.Timestamp | None


def project_header(tables: dict[str, pd.DataFrame]) -> ProjectHeader:
    df = tables.get("PROJECT")
    if df is None or df.empty:
        return ProjectHeader("-", "-", None, None, None, None)
    # Exclude the reserved default project node if several rows are present
    row = df.iloc[0]
    if len(df) > 1 and "export_flag" in df.columns:
        flagged = df[df["export_flag"].astype(str).str.upper() == "Y"]
        if not flagged.empty:
            row = flagged.iloc[0]
    return ProjectHeader(
        proj_id=str(row.get("proj_id")),
        short_name=str(row.get("proj_short_name", "-")),
        data_date=pd.to_datetime(row.get("last_recalc_date"), errors="coerce"),
        plan_start=pd.to_datetime(row.get("plan_start_date"), errors="coerce"),
        plan_finish=pd.to_datetime(row.get("plan_end_date"), errors="coerce"),
        scd_finish=pd.to_datetime(row.get("scd_end_date"), errors="coerce"),
    )


# --------------------------------------------------------------------------
# WBS tree
# --------------------------------------------------------------------------

def wbs_tree(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """PROJWBS flattened with level, full path and ordered display label."""
    df = tables.get("PROJWBS")
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["wbs_id", "parent_wbs_id", "wbs_short_name", "wbs_name", "level", "path", "label"]
        )
    df = df.copy()
    df["wbs_id"] = df["wbs_id"].astype(str)
    df["parent_wbs_id"] = df["parent_wbs_id"].astype(str)
    name_of = dict(zip(df["wbs_id"], df["wbs_name"].astype(str)))
    parent_of = dict(zip(df["wbs_id"], df["parent_wbs_id"]))

    def climb(wid: str) -> list[str]:
        chain, seen = [], set()
        cur = wid
        while cur in name_of and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = parent_of.get(cur, "")
        return list(reversed(chain))

    chains = {wid: climb(wid) for wid in df["wbs_id"]}
    df["level"] = df["wbs_id"].map(lambda w: len(chains[w]))
    df["path"] = df["wbs_id"].map(lambda w: " › ".join(name_of[c] for c in chains[w]))
    df["ancestors"] = df["wbs_id"].map(lambda w: chains[w])
    df["label"] = df.apply(
        lambda r: f"{'   ' * max(0, int(r['level']) - 1)}{r['wbs_name']}", axis=1
    )
    return df.sort_values(["level", "seq_num"] if "seq_num" in df.columns else ["level"])


# --------------------------------------------------------------------------
# Activity ID taxonomy
# --------------------------------------------------------------------------

def split_activity_id(code: str, separator: str = ".", labels: tuple[str, ...] = ()) -> dict[str, str]:
    """
    Split a structured Activity ID into named segments.

    Built for the WB.* five-segment convention, but works for any consistent
    separator scheme. Missing segments come back empty rather than raising.
    """
    if not isinstance(code, str):
        return {}
    parts = code.split(separator)
    labels = labels or tuple(f"Segment {i + 1}" for i in range(len(parts)))
    return {labels[i] if i < len(labels) else f"Segment {i + 1}": p for i, p in enumerate(parts)}


def taxonomy_frame(task: pd.DataFrame, separator: str = ".", labels: tuple[str, ...] = ()) -> pd.DataFrame:
    """Expand TASK.task_code into one column per ID segment."""
    if task.empty or "task_code" not in task.columns:
        return pd.DataFrame(index=task.index)
    split = task["task_code"].astype(str).str.split(re.escape(separator), regex=True, expand=True)
    n = split.shape[1]
    names = [labels[i] if i < len(labels) else f"Segment {i + 1}" for i in range(n)]
    split.columns = names
    return split.fillna("")
