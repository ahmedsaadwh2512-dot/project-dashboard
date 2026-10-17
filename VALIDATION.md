# Release validation — 17 October 2026

## Supplied schedules

| Item | Current update | Reference baseline |
|---|---:|---:|
| Project | QC-02-001-BLN-1 | QC-02-001-BLN-101 |
| Data date | 08-Oct-2026 20:00 | 14-Apr-2026 07:00 |
| Activities | 757 | 757 |
| Resource assignments | 3,986 | 3,986 |
| Calendars | 4 | 4 |

Exact Activity ID matches: **757 / 757**. Added: 0. Removed: 0.
Activity budgets reconcile to the raw assignment cost sum:
**264,666,062.4561** source currency units. The few sub-cent digits are retained
from the export, not rounded activity by activity.

Full scope, budgeted-baseline-cost basis:

| Metric | Dashboard result |
|---|---:|
| Measurable activities | 470 |
| No positive cost basis | 287 |
| Planned value at data date | 57,705,917.02545 |
| Earned value at data date | 43,818,264.02590 |
| Cost-weighted planned progress | 21.803293% |
| Cost-weighted earned progress | 16.556057% |
| SPI | 0.759337 |

These are reproducible dashboard calculations, **not a claimed reconciliation to
an independent P6 report**. No P6 reference column export was provided. The
activity-based working-hour plan, selected cost basis and supported EV methods
are documented in README. Filters change these figures.

## Defects corrected

1. Calendar shifts in the supplied exports use `f|16:00|s|08:00`; the previous
   parser expected the opposite order and silently used a five-day calendar.
   The supplied Wadi calendars now read as six-day weeks with Friday off and
   all exported exception records.
2. Uploaded baseline now drives reference dates, budget and planned progress
   throughout analysis. Different internal task/calendar IDs across files do
   not break matching.
3. Partial-day hours are retained; the data date is no longer truncated to date.
4. Physical progress no longer guesses the scale from the maximum value.
5. Completed activity dates prefer actual dates rather than stale forecast dates.
6. Missing values remain missing; missing cost basis is not zero progress.
7. Resource/expense totals, selected project boundaries and duplicate Activity
   ID checks are explicit.
8. No synthetic history, automatic critical-path-test PASS, heuristic labour
   adjustment or unsupported delay-causation statements.
9. Literal search safely accepts `[`. Empty filters show an explanatory state.
10. CSV export prefixes spreadsheet formula-like text to prevent formula execution.

## Executed checks

- **10 automated tests passed**, including the supplied schedules, assignment
  total reconciliation, exact baseline mapping, partial-day hours, Friday off,
  unknown EV exclusion, added/removed scope and curve reconciliation at the data
  date and completion.
- Streamlit AppTest: all eight pages rendered without app exceptions with both
  supplied files; status filtering, literal search, empty results and upload-free
  landing state also passed.
- Chromium: uploaded both actual files through the interface; overview charts
  rendered without application exceptions. Landing and overview layouts were
  inspected at 1600 × 1100. No deployment to GitHub/Streamlit Cloud was performed.

## Operational limits

No network rescheduling, forensic delay analysis or certified DCMA assessment.
Unsupported EV techniques are unmeasured. Full exported calendar definitions are
required; inherited partial definitions are not reconstructed. Assignment curves,
manual future-period distributions and P6 administrative baseline settings need
separate reconciliation if exact P6 period-report equivalence is required.
Actual cost is auditable but not used as progress. No earned history can be proven
from the single current snapshot. Read the calculation contract before reporting.

## Version / environment

Python 3.12. Runtime package versions are pinned in requirements.txt to the
versions used for the successful app checks. Tests require pytest separately.
