import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced Project Dashboard",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# SIDEBAR (Controls)
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")

project = st.sidebar.selectbox(
    "Select Project",
    ["Project A", "Project B", "Project C"]
)

date_range = st.sidebar.date_input("Select Date Range")

metric = st.sidebar.selectbox(
    "Select Metric",
    ["Cost", "Progress", "Performance"]
)

# ─────────────────────────────────────────────
# MOCK DATA (replace with your real data)
# ─────────────────────────────────────────────
np.random.seed(42)
df = pd.DataFrame({
    "Task": [f"Task {i}" for i in range(1, 11)],
    "Cost": np.random.randint(1000, 10000, 10),
    "Progress": np.random.randint(10, 100, 10),
    "Performance": np.random.uniform(0.5, 1.5, 10)
})

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("## 📊 Project Controls Dashboard")
st.markdown(f"### {project}")

# ─────────────────────────────────────────────
# KPIs (Top Cards)
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Cost", f"${df['Cost'].sum():,.0f}")
col2.metric("📈 Avg Progress", f"{df['Progress'].mean():.1f}%")
col3.metric("⚡ Performance Index", f"{df['Performance'].mean():.2f}")

# ─────────────────────────────────────────────
# CHARTS ROW 1
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig_bar = px.bar(
        df,
        x="Task",
        y=metric,
        color=metric,
        text=metric,
        title=f"{metric} by Task",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        hovermode="x unified",
        transition_duration=500
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    fig_pie = px.pie(
        df,
        names="Task",
        values="Cost",
        title="Cost Distribution"
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────────────────
# CHARTS ROW 2
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig_line = px.line(
        df,
        x="Task",
        y="Progress",
        markers=True,
        title="Progress Trend"
    )
    fig_line.update_layout(hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    fig_scatter = px.scatter(
        df,
        x="Cost",
        y="Performance",
        size="Progress",
        color="Performance",
        hover_name="Task",
        title="Cost vs Performance"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ─────────────────────────────────────────────
# DATA TABLE (Interactive)
# ─────────────────────────────────────────────
st.markdown("### 📋 Detailed Data")
st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & Plotly")
