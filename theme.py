"""
theme.py — visual identity.

Direction: drawing office, not analytics SaaS. Ink surfaces, limestone text,
a single ochre accent reserved for earned value. Status colours follow
engineering convention (oxide red, survey green) rather than web convention,
so screenshots survive being printed into a client report.

Every numeral is set in a monospaced face with tabular figures, so columns
align the way they do in a P6 report. That is the signature of this dashboard.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

DARK = {
    "ground": "#0E1621",
    "panel": "#16212E",
    "panel_high": "#1B2836",
    "rule": "#233040",
    "text": "#E8E2D5",
    "muted": "#7C94AC",
    "accent": "#C8922F",
    "accent_soft": "#8A6520",
    "good": "#4E9A6A",
    "warn": "#C8922F",
    "bad": "#C0504D",
    "plan": "#4A5C70",
    "grid": "#1E2B3A",
}

LIGHT = {
    "ground": "#F2EFE8",
    "panel": "#FFFFFF",
    "panel_high": "#FAF8F3",
    "rule": "#D8D2C6",
    "text": "#16212E",
    "muted": "#5C6B7C",
    "accent": "#9A6B14",
    "accent_soft": "#C8922F",
    "good": "#3D7A53",
    "warn": "#9A6B14",
    "bad": "#A33F3C",
    "plan": "#9AA7B4",
    "grid": "#E4DFD5",
}

BUILD = "2026-07-30-b"  # shown in the sidebar so you can confirm which file is live

FONT_STACK = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif"
MONO_STACK = "'IBM Plex Mono', 'SF Mono', Consolas, monospace"


def palette(mode: str = "dark") -> dict:
    return DARK if mode == "dark" else LIGHT


def inject_css(st, mode: str = "dark") -> None:
    """
    Inject the stylesheet.

    Streamlit renders st.markdown through a Markdown parser even when
    unsafe_allow_html is set. Two things break a <style> block there: a
    blank line, which ends raw-HTML mode, and four-space indentation,
    which is read as a code fence. Either one dumps the whole stylesheet
    onto the page as text. The CSS is therefore flattened to unindented,
    blank-line-free lines before it is emitted, and the fonts arrive by
    @import rather than a <link> tag, which Streamlit strips.
    """
    c = palette(mode)
    css = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
  :root {{
    --ground:{c['ground']}; --panel:{c['panel']}; --panel-high:{c['panel_high']};
    --rule:{c['rule']}; --text:{c['text']}; --muted:{c['muted']};
    --accent:{c['accent']}; --good:{c['good']}; --warn:{c['warn']};
    --bad:{c['bad']}; --plan:{c['plan']};
  }}

  [data-testid="stHeader"], [data-testid="stToolbar"], .stDeployButton {{ display:none !important; }}

  .stApp {{ background: var(--ground); color: var(--text); font-family: {FONT_STACK}; }}
  section[data-testid="stSidebar"] {{ background: var(--panel); border-right: 1px solid var(--rule); }}

  h1, h2, h3, h4 {{ font-family: {FONT_STACK}; letter-spacing: -0.015em; color: var(--text); }}
  h1 {{ font-size: 1.45rem; font-weight: 600; }}
  h2 {{ font-size: 1.1rem;  font-weight: 600; }}
  h3 {{ font-size: 0.95rem; font-weight: 600; }}

  /* Every numeral is monospaced with tabular figures — the signature */
  .num, .kpi-value, .strip-value, td.num, .stDataFrame td {{
    font-family: {MONO_STACK};
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1;
  }}

  /* Masthead ---------------------------------------------------------- */
  .masthead {{
    border-bottom: 1px solid var(--rule);
    padding: 0 0 0.85rem 0; margin-bottom: 1.1rem;
    display:flex; align-items:baseline; gap:1.1rem; flex-wrap:wrap;
  }}
  .masthead .title {{ font-size:1.3rem; font-weight:600; letter-spacing:-0.02em; }}
  .masthead .meta {{
    font-family:{MONO_STACK}; font-size:0.72rem; color:var(--muted);
    text-transform:uppercase; letter-spacing:0.09em;
  }}
  .masthead .meta b {{ color:var(--text); font-weight:500; }}

  /* KPI card ---------------------------------------------------------- */
  .kpi {{
    background: var(--panel); border:1px solid var(--rule);
    border-radius:3px; padding:0.85rem 0.95rem 0.75rem; height:100%;
  }}
  .kpi-label {{
    font-size:0.66rem; text-transform:uppercase; letter-spacing:0.11em;
    color:var(--muted); font-weight:500; margin-bottom:0.35rem;
  }}
  .kpi-value {{ font-size:1.5rem; font-weight:600; line-height:1.1; color:var(--text); }}
  .kpi-value.accent {{ color: var(--accent); }}
  .kpi-sub {{ font-size:0.72rem; color:var(--muted); margin-top:0.3rem; font-family:{MONO_STACK}; }}

  /* Gauge strip — one idiom, used everywhere a % appears --------------- */
  .strip {{ margin-top:0.55rem; }}
  .strip-track {{
    position:relative; height:6px; background:var(--rule);
    border-radius:1px; overflow:visible;
  }}
  .strip-fill {{ position:absolute; left:0; top:0; height:6px; background:var(--accent); border-radius:1px; }}
  .strip-tick {{
    position:absolute; top:-3px; width:2px; height:12px;
    background:var(--text); opacity:0.85;
  }}
  .strip-legend {{
    display:flex; justify-content:space-between; margin-top:0.32rem;
    font-family:{MONO_STACK}; font-size:0.66rem; color:var(--muted);
  }}

  /* Flags ------------------------------------------------------------- */
  .flag {{
    display:inline-block; font-family:{MONO_STACK}; font-size:0.63rem;
    text-transform:uppercase; letter-spacing:0.08em; padding:0.16rem 0.42rem;
    border:1px solid currentColor; border-radius:2px; font-weight:500;
  }}
  .flag.good {{ color:var(--good); }}
  .flag.warn {{ color:var(--warn); }}
  .flag.bad  {{ color:var(--bad);  }}
  .flag.none {{ color:var(--muted); }}

  /* Notes ------------------------------------------------------------- */
  .note {{
    border-left:2px solid var(--accent); background:var(--panel);
    padding:0.6rem 0.85rem; margin:0.7rem 0; font-size:0.8rem; color:var(--muted);
  }}
  .note b {{ color:var(--text); font-weight:500; }}
  .note.caution {{ border-left-color:var(--bad); }}

  .stTabs [data-baseweb="tab-list"] {{ gap:0; border-bottom:1px solid var(--rule); }}
  .stTabs [data-baseweb="tab"] {{
    font-family:{FONT_STACK}; font-size:0.78rem; font-weight:500;
    letter-spacing:0.02em; padding:0.55rem 0.9rem; color:var(--muted);
  }}
  .stTabs [aria-selected="true"] {{ color:var(--accent); border-bottom:2px solid var(--accent); }}

  [data-testid="stDataFrame"] {{ border:1px solid var(--rule); border-radius:3px; }}
  footer, #MainMenu {{ visibility:hidden; }}

  @media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}
"""
    flat = "\n".join(line.strip() for line in css.splitlines() if line.strip())
    _emit(st, f"<style>{flat}</style>")


def _emit(st, html: str) -> None:
    """
    Write raw HTML.

    st.html renders without going through the Markdown parser, which is what
    we want: Markdown terminates an HTML block at the first blank line unless
    the block opens with <style>, <pre> or <script>, so a stray tag ahead of
    the stylesheet dumps the CSS onto the page as text. st.html arrived in
    Streamlit 1.33; older versions fall back to st.markdown.
    """
    writer = getattr(st, "html", None)
    if callable(writer):
        writer(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def register_plotly(mode: str = "dark") -> str:
    """Register a plotly template matching the palette. Returns its name."""
    c = palette(mode)
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=12, color=c["text"]),
        title=dict(font=dict(size=13, color=c["text"]), x=0, xanchor="left"),
        xaxis=dict(gridcolor=c["grid"], zerolinecolor=c["rule"], linecolor=c["rule"],
                   tickfont=dict(family=MONO_STACK, size=10, color=c["muted"])),
        yaxis=dict(gridcolor=c["grid"], zerolinecolor=c["rule"], linecolor=c["rule"],
                   tickfont=dict(family=MONO_STACK, size=10, color=c["muted"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=10, color=c["muted"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=34, b=10),
        hoverlabel=dict(font=dict(family=MONO_STACK, size=11),
                        bgcolor=c["panel_high"], bordercolor=c["rule"]),
        colorway=[c["accent"], c["plan"], c["good"], c["bad"], c["muted"]],
    )
    name = f"qc_{mode}"
    pio.templates[name] = tpl
    return name


# --------------------------------------------------------------------------
# HTML fragments
# --------------------------------------------------------------------------

def gauge_strip(planned: float | None, earned: float | None) -> str:
    """
    The signature element. Filled bar = earned, tick mark = planned.
    Both values are fractions (0-1). None renders as an unmeasured track.
    """
    if earned is None:
        return (
            '<div class="strip"><div class="strip-track"></div>'
            '<div class="strip-legend"><span>no cost basis</span><span>&mdash;</span></div></div>'
        )
    e = max(0.0, min(1.0, earned)) * 100
    p = None if planned is None else max(0.0, min(1.0, planned)) * 100
    tick = f'<div class="strip-tick" style="left:{p:.1f}%"></div>' if p is not None else ""
    plan_txt = "&mdash;" if p is None else f"plan {p:.1f}%"
    return (
        f'<div class="strip"><div class="strip-track">'
        f'<div class="strip-fill" style="width:{e:.1f}%"></div>{tick}</div>'
        f'<div class="strip-legend"><span>earned {e:.1f}%</span><span>{plan_txt}</span></div></div>'
    )


def kpi(label: str, value: str, sub: str = "", strip: str = "", accent: bool = False) -> str:
    cls = "kpi-value accent" if accent else "kpi-value"
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="{cls}">{value}</div>{sub_html}{strip}</div>'
    )


def flag(text: str, tone: str = "none") -> str:
    return f'<span class="flag {tone}">{text}</span>'


def note(text: str, caution: bool = False) -> str:
    cls = "note caution" if caution else "note"
    return f'<div class="{cls}">{text}</div>'
