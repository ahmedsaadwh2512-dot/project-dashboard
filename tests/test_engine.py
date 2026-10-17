from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engine as e
import xer_parser as xp

BLOB='(0||CalendarData()((0||DaysOfWeek()((0||1()((0||0(f|16:00|s|08:00)())))(0||2()((0||0(s|08:00|f|16:00)())))(0||3()((0||0(f|16:00|s|08:00)())))(0||4()((0||0(f|16:00|s|08:00)())))(0||5()((0||0(f|16:00|s|08:00)())))(0||6()())(0||7()((0||0(f|16:00|s|08:00)())))))(0||Exceptions()())))'

def test_calendar_reads_both_shift_orders_and_friday_off():
    c=xp.parse_calendar_blob('1','Six day',BLOB)
    assert c.days_per_week==6
    assert c.hours_on(pd.Timestamp('2026-10-09').date())==0
    assert c.hours_between(pd.Timestamp('2026-10-08 12:00'),pd.Timestamp('2026-10-10 12:00'))==8

def test_no_silent_calendar_substitution():
    with pytest.raises(ValueError):xp.parse_calendar_blob('1','Broken','')

def test_partial_day():
    c=xp.parse_calendar_blob('1','Six day',BLOB)
    assert c.hours_between(pd.Timestamp('2026-10-08 08:00'),pd.Timestamp('2026-10-08 10:00'))==2

def test_csv_formula_protection():
    assert "'=HYPERLINK" in e.export_csv(pd.DataFrame({'name':['=HYPERLINK("bad")']})).decode('utf-8-sig')

def test_empty_totals():
    a=pd.DataFrame(columns=['bac','ev','pv','actual_cost','budget'])
    assert np.isnan(e.totals(a)['earned'])

@pytest.fixture
def real_pair():
    import os
    folder=Path(os.environ.get('XER_TEST_DIR','/nonexistent'))
    if not folder.exists():pytest.skip('Set XER_TEST_DIR to run private schedule integration checks.')
    u=e.load((folder/'QC-BLN-001-08.Oct.26.xer').read_bytes(),'7866')
    b=e.load((folder/'QC-02-001-BLN-001-Final Baseline(2).xer').read_bytes(),'5708')
    return u,b

def test_real_costs_match_raw_assignments(real_pair):
    u,b=real_pair
    for project in [u,b]:
        assert project['activities'].budget.sum()==pytest.approx(project['tables']['TASKRSRC'].target_cost.sum())
    a,r=e.analyse(u,b)
    assert len(a)==757 and a.matched.all() and r.empty
    assert e.totals(a)['bac']==pytest.approx(264666062.4561)
    assert a.ev.sum()==pytest.approx((a.bac*a.performance_pct).sum())
    assert a.base_finish.equals(a.task_code.map(b['activities'].set_index('task_code').finish))

def test_curve_reconciles_at_data_date_and_completion(real_pair):
    u,b=real_pair;a,_=e.analyse(u,b);dd=u['header'].data_date
    c=e.curve(a,b['calendars'],dd)
    assert c.loc[c.date.eq(dd),'planned'].iloc[0]==pytest.approx(e.totals(a)['pv'])
    assert c.planned.iloc[-1]==pytest.approx(e.totals(a)['bac'])
    assert c.planned.diff().dropna().min()>=-.01

def test_unknown_ev_does_not_become_zero(real_pair):
    u,b=real_pair;u['activities'].loc[0,'performance_pct']=np.nan
    a,_=e.analyse(u,b)
    assert pd.isna(a.ev.iloc[0])

def test_added_activity_is_not_assigned_fake_baseline(real_pair):
    u,b=real_pair;u['activities'].loc[0,'task_code']='NEW'
    a,r=e.analyse(u,b)
    assert not a.matched.iloc[0] and pd.isna(a.bac.iloc[0]) and len(r)==1

def test_physical_one_percent_is_not_one_hundred(real_pair):
    u,_=real_pair
    # Engine always reads XER physical percent in 0–100 units.
    a=u['activities'];mask=a.complete_pct_type.eq('CP_Phys')&a.status.eq('In progress')
    assert np.allclose(a.loc[mask,'activity_pct'],a.loc[mask,'phys_complete_pct']/100)
