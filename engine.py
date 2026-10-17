"""Auditable XER snapshot analytics. No scheduling or delay-causation claims."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import xer_parser as xp


def num(df, key, default=0):
    return pd.to_numeric(df.get(key, pd.Series(default, index=df.index)), errors='coerce').fillna(default)


def dates(df, key):
    return pd.to_datetime(df.get(key, pd.Series(pd.NaT, index=df.index)), errors='coerce')


def projects(raw):
    return xp.read_xer(io.BytesIO(raw))['PROJECT'][['proj_id','proj_short_name']]


def load(raw, project_id):
    tables = xp.read_xer(io.BytesIO(raw))
    for k, df in tables.items():
        if 'proj_id' in df and k != 'CALENDAR':
            tables[k] = df[df.proj_id.eq(str(project_id))].copy()
    task = tables['TASK'].copy().reset_index(drop=True)
    if task.empty: raise ValueError('The selected project has no activities.')
    if task.task_code.duplicated().any():
        raise ValueError('Duplicate Activity IDs in the selected project. Correct them before comparison.')
    cals = xp.build_calendars(tables)
    missing = set(task.clndr_id) - set(cals)
    if missing: raise ValueError(f'Missing activity calendars: {sorted(missing)}')
    wbs = xp.wbs_tree(tables)
    header = xp.project_header(tables)
    if pd.isna(header.data_date): raise ValueError('Project data date is missing.')
    for name in ['budget','actual_cost','remaining_cost','labour_hours','equipment_hours','assignments']:
        task[name] = 0.0
    tr = tables.get('TASKRSRC', pd.DataFrame())
    if not tr.empty:
        tr = tr.copy()
        tr['budget'] = num(tr,'target_cost')
        tr['actual_cost'] = num(tr,'act_reg_cost') + num(tr,'act_ot_cost')
        tr['remaining_cost'] = num(tr,'remain_cost')
        rt = tr.get('rsrc_type', pd.Series('',index=tr.index))
        if 'RSRC' in tables and 'rsrc_type' in tables['RSRC']:
            rt = rt.replace('',pd.NA).fillna(tr.rsrc_id.map(tables['RSRC'].set_index('rsrc_id').rsrc_type))
        tr['labour_hours'] = num(tr,'target_qty').where(rt.eq('RT_Labor'),0)
        tr['equipment_hours'] = num(tr,'target_qty').where(rt.eq('RT_Equip'),0)
        tr['assignments'] = 1
        cols = ['budget','actual_cost','remaining_cost','labour_hours','equipment_hours','assignments']
        agg = tr.groupby('task_id')[cols].sum()
        for k in cols: task[k] = task.task_id.map(agg[k]).fillna(0)
    expenses = tables.get('PROJCOST',pd.DataFrame())
    if not expenses.empty:
        for out,key in [('budget','target_cost'),('actual_cost','act_cost'),('remaining_cost','remain_cost')]:
            vals = num(expenses,key).groupby(expenses.task_id).sum()
            task[out] += task.task_id.map(vals).fillna(0)
    physical = num(task,'phys_complete_pct') / 100.0
    od, rd = num(task,'target_drtn_hr_cnt'), num(task,'remain_drtn_hr_cnt')
    duration = (od-rd).div(od.replace(0,np.nan)).fillna(0)
    au = num(task,'act_work_qty')+num(task,'act_equip_qty')
    ru = num(task,'remain_work_qty')+num(task,'remain_equip_qty')
    units = au.div((au+ru).replace(0,np.nan)).fillna(0)
    task['activity_pct'] = np.select([task.complete_pct_type.eq('CP_Phys'),task.complete_pct_type.eq('CP_Drtn'),task.complete_pct_type.eq('CP_Units')],[physical,duration,units],default=np.nan)
    task['activity_pct'] = task.activity_pct.clip(0,1).mask(task.status_code.eq('TK_Complete'),1).mask(task.status_code.eq('TK_NotStart'),0)
    task['start'] = dates(task,'act_start_date').combine_first(dates(task,'restart_date')).combine_first(dates(task,'early_start_date'))
    task['finish'] = dates(task,'act_end_date').combine_first(dates(task,'reend_date')).combine_first(dates(task,'early_end_date'))
    task['remaining_start'] = dates(task,'restart_date').combine_first(dates(task,'early_start_date'))
    task['calendar_hours'] = task.clndr_id.map({k:c.hours_per_day for k,c in cals.items()})
    task['float_days'] = num(task,'total_float_hr_cnt',np.nan)/task.calendar_hours
    task['remaining_days'] = rd/task.calendar_hours
    task['original_days'] = od/task.calendar_hours
    task['status'] = task.status_code.map({'TK_NotStart':'Not started','TK_Active':'In progress','TK_Complete':'Completed'}).fillna('Unknown')
    task['longest_path'] = task.get('driving_path_flag',pd.Series('',index=task.index)).eq('Y')
    task['milestone'] = task.task_type.isin(['TT_Mile','TT_FinMile'])
    if not wbs.empty:
        tree = wbs.set_index('wbs_id')
        task['wbs_path'] = task.wbs_id.map(tree.path).fillna('Unassigned')
        name = tree.wbs_name.to_dict()
        task['wbs_main'] = task.wbs_id.map({wid:name[ch[min(1,len(ch)-1)]] for wid,ch in zip(wbs.wbs_id,wbs.ancestors)})
        task['ev_method'] = task.wbs_id.map(tree.ev_compute_type)
        task['user_pct'] = task.wbs_id.map(num(tree,'ev_user_pct'))/100
    else:
        task['wbs_path'] = task['wbs_main'] = 'Unassigned'
        task['ev_method'] = ''; task['user_pct'] = np.nan
    task['performance_pct'] = np.select([task.ev_method.eq('EC_Cmp_pct'),task.ev_method.eq('EC_0_100'),task.ev_method.eq('EC_50_50'),task.ev_method.eq('EC_Cmp_pct_udf')],[task.activity_pct,task.status.eq('Completed').astype(float),np.select([task.status.eq('Completed'),task.status.eq('In progress')],[1,.5],default=0),task.user_pct],default=np.nan)
    return {'tables':tables,'activities':task,'calendars':cals,'wbs':wbs,'header':header}


def analyse(current, baseline=None, budget_basis='Budgeted cost'):
    a = current['activities'].copy()
    base = baseline or current
    b = base['activities'].set_index('task_code')
    a['matched'] = a.task_code.isin(b.index)
    a['base_start'] = a.task_code.map(b.start)
    a['base_finish'] = a.task_code.map(b.finish)
    a['base_calendar'] = a.task_code.map(b.clndr_id)
    base_budget = b.budget if budget_basis == 'Budgeted cost' else b.actual_cost+b.remaining_cost
    a['bac'] = a.task_code.map(base_budget)
    a['ev'] = a.bac * a.performance_pct
    dd = current['header'].data_date
    planned = []
    for r in a.itertuples():
        if not r.matched or pd.isna(r.base_start) or pd.isna(r.base_finish) or r.base_finish < r.base_start:
            planned.append(np.nan); continue
        if dd >= r.base_finish: planned.append(1.0); continue
        if dd <= r.base_start: planned.append(0.0); continue
        cal = base['calendars'][r.base_calendar]
        total = cal.hours_between(r.base_start,r.base_finish)
        planned.append(cal.hours_between(r.base_start,dd)/total if total>0 else np.nan)
    a['planned_pct'] = planned
    a['pv'] = a.bac*a.planned_pct
    a['variance_cd'] = (a.finish.dt.normalize()-a.base_finish.dt.normalize()).dt.days
    a['variance_wd'] = [signed_hours(base['calendars'].get(r.base_calendar),r.base_finish,r.finish) for r in a.itertuples()]
    a['earned_pct'] = a.ev.div(a.bac.where(a.bac>0))
    a['variance_pp'] = 100*(a.earned_pct-a.planned_pct)
    removed = base['activities'][~base['activities'].task_code.isin(a.task_code)].copy() if baseline else base['activities'].iloc[:0].copy()
    return a, removed


def signed_hours(cal,a,b):
    if cal is None or pd.isna(a) or pd.isna(b): return np.nan
    return (1 if b>=a else -1)*cal.hours_between(min(a,b),max(a,b))/cal.hours_per_day


def totals(a):
    measured = a[a.bac.gt(0) & a.ev.notna() & a.pv.notna()]
    bac,pv,ev = [float(measured[k].sum()) for k in ['bac','pv','ev']]
    return dict(bac=bac,pv=pv,ev=ev,planned=pv/bac if bac else np.nan,earned=ev/bac if bac else np.nan,spi=ev/pv if pv else np.nan,actual=float(a.actual_cost.sum()),budget=float(a.budget.sum()),measured=len(measured),excluded=len(a)-len(measured))


def rollup(a,key='wbs_main'):
    rows=[]
    for name,g in a.groupby(key,dropna=False):
        m=totals(g); rows.append({key:name,**m,'activities':len(g),'variance_pp':100*(m['earned']-m['planned']),'finish':g.finish.max()})
    return pd.DataFrame(rows).sort_values('bac',ascending=False) if rows else pd.DataFrame()


def curve(a,cals,dd,interval='Monthly'):
    """Working-hour weighted plan. EV is a snapshot, never fabricated history."""
    bucket={}
    flows={}
    for r in a.itertuples():
        if pd.isna(r.bac) or r.bac<=0 or pd.isna(r.pv) or pd.isna(r.ev): continue
        cal=cals[r.base_calendar]; s,f=r.base_start,r.base_finish
        total=cal.hours_between(s,f)
        if total<=0:
            bucket[f]=bucket.get(f,0)+r.bac
            flows[f.normalize()]=flows.get(f.normalize(),0)+r.bac
            continue
        day=s.normalize()
        while day<=f.normalize():
            end=min(day+pd.Timedelta(days=1),f)
            value=r.bac*cal.hours_between(max(day,s),end)/total
            bucket[end]=bucket.get(end,0)+value
            flows[day]=flows.get(day,0)+value
            day+=pd.Timedelta(days=1)
    if not bucket:return pd.DataFrame(columns=['date','planned'])
    series=pd.Series(bucket).sort_index().cumsum()
    rule={'Weekly':'W-THU','Monthly':'ME','Quarterly':'QE'}[interval]
    sampled=series.resample(rule).last().ffill()
    sampled=sampled[sampled.index<=series.index.max()]
    points=pd.concat([sampled,pd.Series({a.base_start.min():0.0,dd:totals(a)['pv'],series.index.max():series.iloc[-1]})]).sort_index()
    points=points[~points.index.duplicated(keep='last')]
    result=points.rename_axis('date').reset_index(name='planned')
    frequency={'Weekly':'W-THU','Monthly':'M','Quarterly':'Q'}[interval]
    increments=pd.Series(flows).sort_index()
    grouped=increments.groupby(increments.index.to_period(frequency)).sum()
    complete_index=pd.period_range(grouped.index.min(),grouped.index.max(),freq=frequency)
    grouped=grouped.reindex(complete_index,fill_value=0)
    periods=pd.DataFrame({'date':grouped.index.to_timestamp(how='end'),'period_value':grouped.values,'cumulative':grouped.cumsum().values})
    result.attrs['periods']=periods.to_dict('list')
    result.attrs['interval']=interval
    return result


def quality(a,tables,dd):
    issues=[]
    def add(label,mask,detail):
        g=a.loc[mask.fillna(False)]
        issues.append({'Check':label,'Activities':len(g),'Explanation':detail,'IDs':', '.join(g.task_code)})
    inc=a.status.ne('Completed');real=~a.milestone & ~a.task_type.isin(['TT_LOE','TT_WBS'])
    pred=tables.get('TASKPRED',pd.DataFrame())
    haspred=set(pred.get('task_id',[]));hassucc=set(pred.get('pred_task_id',[]))
    add('Open ends',inc&real&(~a.task_id.isin(haspred)|~a.task_id.isin(hassucc)),'Incomplete non-milestones missing predecessor or successor; review boundary activities.')
    add('Negative float',inc&a.float_days.lt(0),'Remaining work with negative total float.')
    add('High float',inc&real&a.float_days.gt(44),'Screening threshold: 44 calendar-conversion working days.')
    add('Long remaining duration',inc&real&a.remaining_days.gt(44),'Screening threshold: 44 working days; inspect work packaging.')
    add('Actual dates after data date',dates(a,'act_start_date').gt(dd)|dates(a,'act_end_date').gt(dd),'Actual dates later than the selected project data date.')
    add('Completed without actual finish',a.status.eq('Completed')&dates(a,'act_end_date').isna(),'Missing actual finish.')
    add('Completed with remaining duration',a.status.eq('Completed')&num(a,'remain_drtn_hr_cnt').gt(0),'Remaining duration should be reviewed.')
    add('Completed with remaining cost',a.status.eq('Completed')&a.remaining_cost.abs().gt(.01),'Review residual resource or expense costs.')
    add('Missing baseline match',~a.matched,'No identical Activity ID in the selected reference; excluded from EVM.')
    add('No positive cost basis',a.bac.le(0)|a.bac.isna(),'Unmeasured on a cost-weighted basis; not zero progress.')
    add('Unsupported EV technique',a.performance_pct.isna(),'No substitute formula is assumed.')
    add('Invalid reference dates',a.base_start.isna()|a.base_finish.isna()|a.base_finish.lt(a.base_start),'Cannot establish a valid planned duration.')
    add('No resource assignment',real&a.assignments.eq(0),'Resource presence is checked independently of cost.')
    return pd.DataFrame(issues)


def export_csv(frame):
    safe=frame.copy()
    for c in safe.select_dtypes(include=['object','string']).columns:
        safe[c]=safe[c].map(lambda x:"'"+x if isinstance(x,str) and x.startswith(('=','+','-','@')) else x)
    return safe.to_csv(index=False).encode('utf-8-sig')


def schedule_attention(a, behind_basis='Either condition', critical_basis='Float ≤ 0', progress_tolerance=.1):
    """Flag incomplete work only; missing progress/float never implies zero."""
    out=a.copy()
    incomplete=out.status.ne('Completed')
    progress=out.earned_pct.notna() & out.planned_pct.notna() & ((out.planned_pct-out.earned_pct)*100 > progress_tolerance)
    finish=out.variance_cd.gt(0)
    if behind_basis=='Progress below plan':behind=progress
    elif behind_basis=='Finish later than baseline':behind=finish
    else:behind=progress|finish
    floating=out.float_days.le(0)
    longest=out.longest_path.fillna(False)
    critical={'Float ≤ 0':floating,'Exported longest path':longest,'Either criterion':floating|longest}[critical_basis]
    out['behind_schedule']=incomplete & behind.fillna(False)
    out['critical']=incomplete & critical.fillna(False)
    out['behind_and_critical']=out.behind_schedule & out.critical
    out['progress_gap_pp']=(out.earned_pct-out.planned_pct)*100
    out['behind_reason']=np.select([incomplete&progress&finish,incomplete&progress,incomplete&finish],['Progress below plan + finish later','Progress below plan','Finish later than baseline'],default='—')
    out['critical_reason']=np.select([incomplete&floating&longest,incomplete&floating,incomplete&longest],['Float ≤ 0 + longest path','Float ≤ 0','Exported longest path'],default='—')
    return out
