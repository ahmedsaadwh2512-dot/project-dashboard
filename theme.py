"""Restrained navy, teal and amber design system."""
BUILD='2.0.0'
COLORS={'teal':'#11B9A3','blue':'#7189FF','amber':'#F3B65B','red':'#F27785','muted':'#92A3BA'}
def apply(st,dark=True):
    bg,panel,ink,line=('#0B1220','#121E30','#ECF2FC','#26354B') if dark else ('#F3F6FB','#FFFFFF','#172840','#DBE3EE')
    st.markdown(f'''<style>
    .stApp {{background:{bg};color:{ink}}}
    [data-testid="stSidebar"] {{background:{panel};border-right:1px solid {line}}}
    .block-container {{max-width:1560px;padding-top:4.5rem;padding-bottom:3rem}}
    h1 {{font-size:2.1rem!important;letter-spacing:-.06rem;font-weight:750!important}}
    h2,h3 {{letter-spacing:-.025rem}}
    [data-testid="stMetric"] {{background:{panel};border:1px solid {line};border-radius:14px;padding:18px 20px;min-height:120px}}
    [data-testid="stMetricLabel"] {{color:#92A3BA;font-size:.8rem}}
    [data-testid="stMetricValue"] {{font-size:1.85rem;font-weight:650}}
    .eyebrow {{color:#11B9A3;font-size:.72rem;font-weight:700;letter-spacing:.18rem;margin-bottom:.5rem}}
    .hero {{border:1px solid {line};border-radius:18px;background:{panel};padding:38px;margin:24px 0}}
    .hero h2 {{font-size:2.6rem;max-width:750px;line-height:1.14}}
    .hero p {{color:#92A3BA;max-width:680px;font-size:1.05rem;line-height:1.7}}
    .stButton button,.stDownloadButton button {{border-radius:9px}}
    [data-testid="stDataFrame"] {{border:1px solid {line};border-radius:12px}}
    </style>''',unsafe_allow_html=True)
    return {'bg':bg,'panel':panel,'ink':ink,'line':line,**COLORS}
