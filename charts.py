import plotly.graph_objects as go
import plotly.express as px

def style(fig,C,height=390):
    fig.update_layout(template='plotly_dark' if C['bg']=='#0B1220' else 'plotly_white',height=height,
        paper_bgcolor=C['panel'],plot_bgcolor=C['panel'],font=dict(family='Arial, sans-serif',color=C['ink'],size=12),
        margin=dict(l=20,r=24,t=35,b=30),legend=dict(orientation='h',y=1.13,x=0),
        hoverlabel=dict(bgcolor=C['panel'],font_size=13),hovermode='closest')
    fig.update_xaxes(gridcolor=C['line'],zeroline=False)
    fig.update_yaxes(gridcolor=C['line'],zeroline=False)
    return fig

def progress(g,key,C):
    g=g[g.bac.gt(0)].sort_values('bac').tail(15)
    fig=go.Figure()
    for col,title,color in [('planned','Planned',C['blue']),('earned','Earned',C['teal'])]:
        fig.add_bar(y=g[key],x=g[col]*100,name=title,orientation='h',marker_color=color,
            hovertemplate='%{y}<br>'+title+': %{x:.2f}%<extra></extra>')
    fig.update_layout(barmode='group')
    fig.update_xaxes(title='Cost-weighted progress (%)',range=[0,105])
    return style(fig,C,max(330,min(650,len(g)*65)))

def scurve(curve,dd,m,C,percent=True):
    scale=100/m['bac'] if percent and m['bac'] else 1e-6
    suffix='%' if percent else 'M'
    fig=go.Figure()
    fig.add_scatter(x=curve.date,y=curve.planned*scale,name='Baseline plan',mode='lines',line=dict(color=C['blue'],width=3))
    fig.add_scatter(x=[dd],y=[m['pv']*scale],name='Planned at data date',mode='markers',marker=dict(color=C['blue'],size=11))
    fig.add_scatter(x=[dd],y=[m['ev']*scale],name='Earned at data date',mode='markers',marker=dict(color=C['teal'],size=14,line=dict(color=C['panel'],width=3)))
    fig.add_shape(type='line',x0=dd,x1=dd,y0=0,y1=1,yref='paper',line=dict(color=C['muted'],dash='dot'))
    fig.update_traces(hovertemplate='%{x|%d %b %Y}<br>%{y:,.2f}'+suffix+'<extra>%{fullData.name}</extra>')
    fig.update_yaxes(title='Progress (%)' if percent else 'Value (millions)',rangemode='tozero')
    return style(fig,C,440)

def budget(g,key,C):
    g=g[g.bac.gt(0)].copy()
    labels=[f'{v:.1f}%' if v>=2 else '' for v in g.bac/g.bac.sum()*100]
    fig=go.Figure(go.Pie(labels=g[key],values=g.bac,hole=.72,sort=False,
        marker=dict(colors=[C['teal'],C['blue'],C['amber'],C['red'],'#558FB5','#9D7DCE','#82B4A4']),
        text=labels,textinfo='text',textposition='inside',hovertemplate='%{label}<br>%{value:,.0f}<br>%{percent}<extra></extra>'))
    fig.update_layout(showlegend=True,annotations=[dict(text='BUDGET<br>MIX',x=.5,y=.5,showarrow=False,font=dict(size=15,color=C['muted']))])
    style(fig,C,400)
    fig.update_layout(legend=dict(orientation='h',y=-.05,x=0,font=dict(size=10)),margin=dict(l=15,r=15,t=25,b=80))
    return fig

def variance(a,C,unit='Calendar days'):
    col='variance_cd' if unit=='Calendar days' else 'variance_wd'
    g=a.dropna(subset=[col]).nlargest(15,col).sort_values(col)
    fig=go.Figure(go.Bar(y=g.task_code,x=g[col],orientation='h',marker_color=[C['red'] if v>0 else C['teal'] for v in g[col]],customdata=g.task_name,hovertemplate='%{y}<br>%{customdata}<br>%{x:.1f} days<extra></extra>'))
    fig.update_xaxes(title=unit+' · positive = late')
    return style(fig,C,470)

def timeline(a,C):
    g=a.dropna(subset=['start','finish']).sort_values('start').head(60).copy()
    g['label']=g.task_code+' · '+g.task_name.str.slice(0,65)
    fig=px.timeline(g,x_start='start',x_end='finish',y='label',color='status',color_discrete_map={'Completed':C['teal'],'In progress':C['blue'],'Not started':C['muted']},hover_data=['task_name','float_days'])
    fig.update_yaxes(autorange='reversed',title='')
    return style(fig,C,max(400,min(1400,80+len(g)*27)))
