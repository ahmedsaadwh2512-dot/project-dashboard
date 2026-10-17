"""Project Controls / XER Intelligence — Streamlit entry point."""
from html import escape
import pandas as pd
import streamlit as st
import engine as en
import theme
import charts

st.set_page_config(page_title='Project Controls | Ahmed Saad',page_icon='◈',layout='wide')
with st.sidebar:
    st.markdown('### ◈ PROJECT CONTROLS')
    st.caption('XER INTELLIGENCE  /  '+theme.BUILD)
    dark=True
    page=st.selectbox('Workspace',['Overview','WBS explorer','S-curve','Milestones & variance','Critical activities','Baseline comparison','Data quality','Activity register'])
    st.divider()
    current_file=st.file_uploader('01 · Current update',type=['xer'])
    baseline_file=st.file_uploader('02 · Approved baseline',type=['xer'])
C=theme.apply(st,dark)
st.markdown('<div class="eyebrow">PROJECT CONTROLS / DECISION WORKSPACE</div>',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def choices(raw):return en.projects(raw)
@st.cache_data(show_spinner=False)
def load(raw,pid):return en.load(raw,pid)
@st.cache_data(show_spinner=False)
def analysis(current,base,basis):return en.analyse(current,base,basis)
@st.cache_data(show_spinner=False)
def plan_curve(a,cals,dd,interval):return en.curve(a,cals,dd,interval)

def fmt(x,spec=',.2f'):
    return '—' if x is None or pd.isna(x) else format(x,spec)
def datefmt(x):return '—' if pd.isna(x) else pd.Timestamp(x).strftime('%d %b %Y')
def chart(fig):st.plotly_chart(fig,width='stretch',config={'displaylogo':False,'toImageButtonOptions':{'format':'png','scale':2}})
def table(df,key):
    st.dataframe(df,width='stretch',hide_index=True)
    st.download_button('↓ Download CSV',en.export_csv(df),key+'.csv','text/csv',key='dl_'+key)

def select_project(raw,label):
    p=choices(raw)
    names=dict(zip(p.proj_id,p.proj_short_name))
    return st.sidebar.selectbox(label,list(names),format_func=lambda x:names[x])

if current_file is None:
    st.title('Clarity across every workfront.')
    st.markdown('''<div class="hero"><div class="eyebrow">FROM SCHEDULE TO DECISION</div>
    <h2>Your programme.<br>Your baseline.<br>One clear picture.</h2>
    <p>Understand progress, see date movement and inspect the activities behind the numbers. Upload your current Primavera XER and approved baseline to begin.</p></div>''',unsafe_allow_html=True)
    for col,title,body in zip(st.columns(3),['01 / Measure','02 / Investigate','03 / Act'],['Cost-weighted planned and earned progress with a visible calculation basis.','Explore WBS, milestones, variance and the exported longest path.','Filter the register, inspect data quality and export the evidence.']):
        with col:
            st.subheader(title);st.write(body)
    st.info('Project files are supplied through the sidebar. No example figures are shown as project results.')
    st.stop()

try:
    raw=current_file.getvalue(); pid=select_project(raw,'Current project')
    with st.spinner('Reading activities, resources and calendars…'):current=load(raw,pid)
    base=None
    if baseline_file:
        br=baseline_file.getvalue();bid=select_project(br,'Baseline project')
        base=load(br,bid)
    basis=st.sidebar.selectbox('Baseline cost basis',['Budgeted cost','At-completion cost'],help='Choose the cost basis used in your P6 earned-value settings. Budgeted cost is the default here; exact P6 reconciliation requires the same setting.')
    with st.spinner('Matching the baseline and calculating progress…'):all_a,removed=analysis(current,base,basis)
except (ValueError,KeyError,TypeError,UnicodeError) as exc:
    st.error(f'Cannot analyse this export: {exc}');st.stop()

h=current['header'];dd=h.data_date
st.title(h.short_name)
st.caption(f'DATA DATE  {dd:%d %b %Y · %H:%M}   /   {len(all_a):,} ACTIVITIES   /   '+('BASELINE CONNECTED' if base else 'CURRENT PLAN ONLY'))
if base is None:
    st.warning('No approved baseline uploaded. Planned values use the current snapshot’s dates and costs; they are not approved-baseline performance.')
else:
    st.caption(f'Reference: {base["header"].short_name} · exact Activity ID matching · {int(all_a.matched.sum()):,} matched / {len(all_a):,} current · {len(removed):,} baseline-only')
    if removed.shape[0] or not all_a.matched.all():st.warning('Scope differs between files. EVM covers matched, measurable activities only. Added and removed work is shown in Baseline comparison.')

with st.sidebar:
    st.divider();st.markdown('**Explore your scope**')
    groups=st.multiselect('Main WBS',sorted(all_a.wbs_main.dropna().unique()))
    candidate=all_a[all_a.wbs_main.isin(groups)] if groups else all_a
    paths=st.multiselect('Detailed WBS',sorted(candidate.wbs_path.dropna().unique()))
    statuses=st.multiselect('Activity status',['Not started','In progress','Completed'])
    search=st.text_input('Activity ID or name',placeholder='Search activities…')
    exclude_loe=st.checkbox('Exclude level of effort',value=False)
    st.divider();st.caption('Ahmed Saad · Project Controls\nAll values follow the active scope except Data quality and Baseline comparison, which inspect the full project.')
a=candidate.copy()
if paths:a=a[a.wbs_path.isin(paths)]
if statuses:a=a[a.status.isin(statuses)]
if exclude_loe:a=a[~a.task_type.isin(['TT_LOE','TT_WBS'])]
if search:a=a[a.task_code.str.contains(search,case=False,regex=False,na=False)|a.task_name.str.contains(search,case=False,regex=False,na=False)]
m=en.totals(a)
st.caption(f'ACTIVE SCOPE  {len(a):,} activities · {m["measured"]:,} measured · {m["excluded"]:,} excluded from cost-weighted progress')
if a.empty and page not in ['Data quality','Baseline comparison']:
    st.info('No activities match these filters. Clear a filter to continue.');st.stop()

if page=='Overview':
    cols=st.columns(4)
    cols[0].metric('EARNED PROGRESS',fmt(m['earned']*100)+ '%',help='Earned value / measurable baseline budget.')
    cols[1].metric('PLANNED PROGRESS',fmt(m['planned']*100)+'%')
    cols[2].metric('PROGRESS VARIANCE',fmt((m['earned']-m['planned'])*100,'+.2f')+' pp')
    cols[3].metric('SCHEDULE PERFORMANCE',fmt(m['spi'],'.3f'),help='SPI = earned value / planned value; it is not a forecast completion-date measure.')
    st.write('')
    l,r=st.columns([2.1,1])
    g=en.rollup(a)
    with l:st.subheader('Progress by workstream');chart(charts.progress(g,'wbs_main',C))
    with r:
        st.subheader('Budget composition');chart(charts.budget(g,'wbs_main',C));st.caption('Measured baseline budget · '+fmt(m['bac']/1e6)+'M')
    l,r=st.columns([2.1,1])
    with l:
        st.subheader('The plan and where we stand')
        curve=plan_curve(a,(base or current)['calendars'],dd,'Monthly')
        if not curve.empty:chart(charts.scurve(curve,dd,m,C))
    with r:
        st.subheader('Delivery snapshot')
        st.metric('Forecast scope finish',datefmt(a.finish.max()))
        st.metric('Incomplete · negative float',str(int((a.status.ne('Completed')&a.float_days.lt(0)).sum())))
        st.metric('Exported longest path',str(int((a.longest_path&a.status.ne('Completed')).sum())))
    st.caption('Earned progress is a data-date snapshot. Historical earned curves are not invented from a single update.')

elif page=='WBS explorer':
    st.subheader('Explore the work breakdown')
    tree=current['wbs'];levels=sorted(tree.level.unique()) if not tree.empty else [1]
    level=st.select_slider('WBS depth',levels,value=levels[min(1,len(levels)-1)])
    lookup=tree.set_index('wbs_id').path.to_dict()
    nodes={wid:lookup[ch[min(int(level)-1,len(ch)-1)]] for wid,ch in zip(tree.wbs_id,tree.ancestors)} if not tree.empty else {}
    a['WBS']=a.wbs_id.map(nodes).fillna('Unassigned')
    g=en.rollup(a,'WBS')
    metric=st.radio('View',['Progress','Budget & values'],horizontal=True)
    if metric=='Progress':chart(charts.progress(g,'WBS',C));st.caption('Largest 15 groups by measured budget; full detail below.')
    display=g[['WBS','activities','bac','pv','ev','planned','earned','variance_pp','spi','excluded']].copy()
    display[['planned','earned']]*=100
    table(display.rename(columns={'bac':'BAC','pv':'PV','ev':'EV','planned':'Planned %','earned':'Earned %','variance_pp':'Variance (pp)'}),'wbs')

elif page=='S-curve':
    st.subheader('Baseline value profile')
    c1,c2=st.columns(2)
    interval=c1.radio('Interval',['Weekly','Monthly','Quarterly'],horizontal=True)
    percent=c2.radio('Display',['Percent','Value (millions)'],horizontal=True)=='Percent'
    curve=plan_curve(a,(base or current)['calendars'],dd,interval)
    if not curve.empty:chart(charts.scurve(curve,dd,m,C,percent))
    cols=st.columns(4)
    for c,title,v in zip(cols,['BASELINE BUDGET','PLANNED VALUE','EARNED VALUE','SCHEDULE VARIANCE'],[m['bac'],m['pv'],m['ev'],m['ev']-m['pv']]):c.metric(title,fmt(v/1e6)+'M')
    st.info('Plan is weighted by working hours between reference activity start and finish, including calendar exceptions. Resource curves and assignment-specific spreads are not reproduced. EV is shown only at the data date; historical snapshots are required for an actual progress trend.')
    if not curve.empty:table(curve,'planned_curve')

elif page=='Milestones & variance':
    st.subheader('Dates that need attention')
    milestone_only=st.toggle('Milestones only',value=True)
    unit=st.radio('Variance unit',['Calendar days','Working days'],horizontal=True)
    v=a[a.milestone].copy() if milestone_only else a.copy()
    if not v.empty:
        chart(charts.variance(v,C,unit));table(v[['task_code','task_name','status','base_finish','finish','variance_cd','variance_wd','float_days']].sort_values('variance_cd',ascending=False),'variance')
    else:st.info('No milestones in this scope.')
    st.caption('Positive variance means later than the reference. Working-day variance uses the baseline calendar’s work hours and hours/day conversion. No delay causation is inferred.')

elif page=='Critical activities':
    st.subheader('Remaining work under pressure')
    criterion=st.radio('Selection',['Exported longest path','Negative float','Near critical'],horizontal=True)
    v=a[a.status.ne('Completed')&~a.task_type.isin(['TT_LOE','TT_WBS'])].copy()
    if criterion=='Exported longest path':v=v[v.longest_path]
    elif criterion=='Negative float':v=v[v.float_days.lt(0)]
    else:
        threshold=st.slider('Float threshold (working days)',0,60,14)
        v=v[v.float_days.between(0,threshold)]
    if v.empty:st.info('No matching activities. The app does not substitute other activities for a missing longest-path flag.')
    else:
        table(v[['task_code','task_name','wbs_main','start','finish','remaining_days','float_days']].sort_values(['float_days','finish']),'critical')
        chart(charts.timeline(v,C));st.caption('Timeline shows the first 60 activities by start. Float is read from the XER; the network is not rescheduled here.')

elif page=='Baseline comparison':
    st.subheader('Scope and date movement · full project')
    if base is None:st.info('Upload the baseline to compare files.');st.stop()
    added=all_a[~all_a.matched]
    cols=st.columns(4)
    for c,label,v in zip(cols,['MATCHED','ADDED','REMOVED','LATER FINISH'],[all_a.matched.sum(),len(added),len(removed),all_a.variance_cd.gt(0).sum()]):c.metric(label,int(v))
    view=st.radio('Show',['Matched','Added','Removed'],horizontal=True)
    if view=='Matched':table(all_a.loc[all_a.matched,['task_code','task_name','base_start','start','base_finish','finish','variance_cd','budget','bac']],'comparison')
    else:
        v=added if view=='Added' else removed
        table(v[['task_code','task_name','wbs_main','start','finish','budget']],view.lower())
    st.caption('An Activity ID change appears as added/removed. No fuzzy name matching is used. Baseline-only costs are not silently reassigned to current work.')

elif page=='Data quality':
    st.subheader('Trust the inputs · full project')
    q=en.quality(all_a,current['tables'],dd)
    table(q.drop(columns='IDs'),'quality')
    label=st.selectbox('Inspect affected activities',q.Check)
    ids=q.loc[q.Check.eq(label),'IDs'].iloc[0].split(', ')
    table(all_a[all_a.task_code.isin(ids)][['task_code','task_name','status','budget','actual_cost','remaining_cost','float_days']],'quality_activities')
    with st.expander('Calendars and calculation basis'):
        table(pd.DataFrame([{'Calendar':c.name,'ID':k,'Workdays/week':c.days_per_week,'Hours/day conversion':c.hours_per_day,'Exception dates':len(c.exception_hours)} for k,c in current['calendars'].items()]),'calendars')
        st.write('BAC basis: '+basis+'. EV = BAC × WBS performance percent. PV = BAC × elapsed reference working hours / total reference working hours. Unknown EV techniques remain unmeasured.')
        st.write('This is a schedule-health screen, not a certified DCMA 14-point assessment. Critical-path stress tests and forensic delay attribution require a scheduling engine and are not marked PASS.')
        st.write('Cost currency is inherited from your source system; no currency conversion is performed. Labour/equipment quantities are reported as exported; no automatic unit divisor is applied.')
        st.write('EAC/CPI, synthetic earned-history curves and speculative impact scores have been removed from executive reporting. Actual costs remain available in the register for audit.')

elif page=='Activity register':
    st.subheader('Every number has an activity behind it')
    view=st.radio('Register view',['Progress & value','Dates & float','Resources'],horizontal=True)
    cols={'Progress & value':['activity_pct','earned_pct','planned_pct','bac','pv','ev','actual_cost','ev_method'],
          'Dates & float':['base_start','base_finish','start','finish','variance_cd','float_days','remaining_days'],
          'Resources':['labour_hours','equipment_hours','assignments','budget','remaining_cost']}[view]
    display=a[['task_code','task_name','wbs_path','status']+cols].copy()
    for col in ['activity_pct','earned_pct','planned_pct']:
        if col in display:display[col]=(display[col]*100).round(2)
    display=display.rename(columns={'task_code':'Activity ID','task_name':'Activity name','wbs_path':'WBS','status':'Status','activity_pct':'Activity %','earned_pct':'Earned %','planned_pct':'Planned %','bac':'BAC','pv':'PV','ev':'EV','actual_cost':'Actual cost','float_days':'Float (WD)','variance_cd':'Variance (CD)','remaining_days':'Remaining (WD)'})
    table(display,'activities')
    st.caption('Percentages use a 0–100 scale. Monetary values are in source currency units.')
    with st.expander('Activity detail'):
        code=st.selectbox('Activity',a.task_code.tolist())
        row=a.set_index('task_code').loc[code]
        st.write(row.task_name)
        st.dataframe(row.rename('Value').astype(str),width='stretch')

st.divider()
st.caption('PROJECT CONTROLS / Ahmed Saad  ·  v'+theme.BUILD+'  ·  Baseline-linked, cost-weighted reporting')
