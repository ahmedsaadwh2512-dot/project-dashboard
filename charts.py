"""
charts.py — every figure in the dashboard.

Separated from app.py so the plotting can be changed without touching the
page logic, and so each chart can be exercised on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def scurve(periods: pd.DataFrame, data_date, C: dict, template: str,
           period_label: str = "Monthly", show_ac: bool = False) -> go.Figure:
    """
    Columns for value earned and planned in each period, lines for the
    cumulative curves. Actual cost is available as a third series but stays
    off by default — on a schedule where P6 derives actuals from percent
    complete, the AC line sits exactly on top of the EV line and tells you
    nothing.
    """
    fig = go.Figure()
    p = periods

    fig.add_bar(x=p["period"], y=p["pv"], name=f"{period_label} planned",
                marker_color=C["plan"], opacity=0.65)
    fig.add_bar(x=p["period"], y=p["ev"], name=f"{period_label} earned",
                marker_color=C["accent"], opacity=0.9)
    if show_ac:
        fig.add_bar(x=p["period"], y=p["ac"], name=f"{period_label} actual",
                    marker_color=C["muted"], opacity=0.55)

    fig.add_trace(go.Scatter(x=p["period"], y=p["cum_pv"], name="Cumulative planned",
                             mode="lines", yaxis="y2",
                             line=dict(color=C["plan"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=p["period"], y=p["cum_ev"].where(p["cum_ev"] > 0),
                             name="Cumulative earned", mode="lines", yaxis="y2",
                             line=dict(color=C["accent"], width=3)))
    if show_ac:
        fig.add_trace(go.Scatter(x=p["period"], y=p["cum_ac"].where(p["cum_ac"] > 0),
                                 name="Cumulative actual", mode="lines", yaxis="y2",
                                 line=dict(color=C["muted"], width=2)))
    fig.add_trace(go.Scatter(x=p["period"], y=p["cum_fcst"], name="Forecast",
                             mode="lines", yaxis="y2",
                             line=dict(color=C["good"], width=2, dash="dash")))

    if pd.notna(data_date):
        fig.add_vline(x=data_date, line_width=1, line_dash="dot", line_color=C["text"],
                      annotation_text="data date", annotation_position="top",
                      annotation_font_size=10)

    fig.update_layout(
        template=template, height=480, barmode="group", bargap=0.25,
        yaxis=dict(title=f"{period_label} value (SAR)"),
        yaxis2=dict(title="Cumulative (SAR)", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)"),
        hovermode="x unified")
    return fig


def wbs_bars(rows: pd.DataFrame, label_col: str, C: dict, template: str) -> go.Figure:
    """Earned as a filled bar, planned as a tick — the gauge strip, as a chart."""
    fig = go.Figure()
    fig.add_bar(y=rows[label_col], x=rows["earned_pct"] * 100, orientation="h",
                name="Earned", marker_color=C["accent"],
                customdata=np.stack([rows["bac"] / 1e6, rows["activities"]], axis=-1),
                hovertemplate="%{y}<br>Earned %{x:.1f}%<br>BAC SAR %{customdata[0]:.1f}M"
                              "<br>%{customdata[1]} activities<extra></extra>")
    fig.add_trace(go.Scatter(
        y=rows[label_col], x=rows["planned_pct"] * 100, mode="markers", name="Planned",
        marker=dict(symbol="line-ns", size=18, line=dict(width=2, color=C["text"])),
        hovertemplate="Planned %{x:.1f}%<extra></extra>"))
    fig.update_layout(template=template, height=max(320, 26 * len(rows)),
                      barmode="overlay", xaxis_title="% of budget", yaxis_title=None)
    return fig


def slip_bars(rows: pd.DataFrame, label_col: str, value_col: str,
              C: dict, template: str) -> go.Figure:
    """Diverging bars — late to the right in oxide red, ahead to the left in green."""
    vals = rows[value_col].fillna(0)
    fig = go.Figure(go.Bar(
        y=rows[label_col], x=vals, orientation="h",
        marker_color=[C["bad"] if v > 0 else C["good"] for v in vals],
        hovertemplate="%{y}<br>%{x:.0f} working days<extra></extra>"))
    fig.update_layout(template=template, height=max(280, 34 * len(rows)),
                      xaxis_title="working days late", yaxis_title=None)
    return fig


def driver_bars(rows: pd.DataFrame, C: dict, template: str, dark: bool = True) -> go.Figure:
    fig = go.Figure(go.Bar(
        y=rows["task_code"], x=rows["impact_score"], orientation="h",
        marker_color=C["accent"], text=rows["task_name"].str.slice(0, 42),
        textposition="inside", insidetextanchor="start",
        textfont=dict(size=10, color=C["ground"] if dark else "#FFFFFF")))
    fig.update_layout(template=template, height=max(320, 30 * len(rows)),
                      xaxis_title="Impact score", yaxis_title=None)
    return fig


def value_donut(rows: pd.DataFrame, label_col: str, C: dict, template: str) -> go.Figure:
    """
    Budget distribution. A donut rather than a pie — the centre carries the
    total, and it is only used at the top WBS level. Below that the category
    count makes any circular chart unreadable, and wbs_bars is used instead.
    """
    r = rows[rows["bac"] > 0]
    shades = [C["accent"], C["plan"], C["good"], C["muted"], C["bad"],
              C["accent_soft"], C["rule"]]
    fig = go.Figure(go.Pie(
        labels=r[label_col], values=r["bac"], hole=0.62, sort=True,
        marker=dict(colors=shades[:len(r)], line=dict(color=C["ground"], width=1.5)),
        textinfo="percent", textfont=dict(size=11, color=C["ground"]),
        hovertemplate="%{label}<br>SAR %{value:,.0f}<br>%{percent}<extra></extra>"))
    total = r["bac"].sum() / 1e6
    fig.update_layout(
        template=template, height=340, showlegend=True,
        annotations=[dict(text=f"SAR<br><b>{total:,.0f}M</b>", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=15, color=C["text"]))])
    return fig


def cost_vs_performance(rows: pd.DataFrame, label_col: str, C: dict, template: str) -> go.Figure:
    """
    Budget against schedule performance. Position on the x axis is how much
    money is at stake; position on the y axis is whether it is being earned on
    plan. Anything large and low on the chart is where attention belongs.
    """
    r = rows[(rows["bac"] > 0) & rows["spi"].notna()].copy()
    if r.empty:
        return go.Figure()
    size = r["activities"].clip(lower=1)
    fig = go.Figure(go.Scatter(
        x=r["bac"] / 1e6, y=r["spi"], mode="markers+text",
        text=r[label_col].str.slice(0, 18), textposition="top center",
        textfont=dict(size=9, color=C["muted"]),
        marker=dict(size=size, sizemode="area",
                    sizeref=2.0 * size.max() / (38 ** 2), sizemin=6,
                    color=r["variance_pct"] * 100, colorscale=[[0, C["bad"]], [0.5, C["warn"]], [1, C["good"]]],
                    cmid=0, line=dict(width=1, color=C["rule"]),
                    colorbar=dict(title=dict(text="Var %", font=dict(size=9)), thickness=10)),
        customdata=np.stack([r["activities"], r["earned_pct"] * 100, r["planned_pct"] * 100], axis=-1),
        hovertemplate="%{text}<br>BAC SAR %{x:.1f}M<br>SPI %{y:.3f}"
                      "<br>Earned %{customdata[1]:.1f}% vs planned %{customdata[2]:.1f}%"
                      "<br>%{customdata[0]} activities<extra></extra>"))
    fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color=C["text"],
                  annotation_text="on plan", annotation_font_size=9)
    fig.update_layout(template=template, height=340,
                      xaxis_title="Budget at completion (SAR M)", yaxis_title="SPI")
    return fig


def progress_trend(periods: pd.DataFrame, bac: float, data_date, C: dict, template: str) -> go.Figure:
    """Cumulative percent complete, planned against earned — the S-curve in percent."""
    if periods.empty or bac <= 0:
        return go.Figure()
    p = periods
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["period"], y=p["cum_pv"] / bac * 100, name="Planned",
                             mode="lines", line=dict(color=C["plan"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=p["period"], y=(p["cum_ev"] / bac * 100).where(p["cum_ev"] > 0),
                             name="Earned", mode="lines+markers",
                             line=dict(color=C["accent"], width=3),
                             marker=dict(size=5)))
    if pd.notna(data_date):
        fig.add_vline(x=data_date, line_width=1, line_dash="dot", line_color=C["text"])
    fig.update_layout(template=template, height=340, hovermode="x unified",
                      yaxis_title="% complete", xaxis_title=None,
                      yaxis=dict(ticksuffix="%"))
    return fig
