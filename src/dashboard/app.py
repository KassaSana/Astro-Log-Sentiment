"""Streamlit dashboard entry point — Astronaut Log Sentiment Analyzer."""

import sqlite3
from pathlib import Path

import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Astronaut Log Sentiment Analyzer",
    page_icon="\U0001f6f8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── NASA Theme CSS ───────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* Main background */
    .stApp { background-color: #0B1D33; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0D2137; }

    /* Headers */
    h1, h2, h3, h4 { color: #E0E8F0 !important; }

    /* Body text */
    p, li, span, label { color: #B0BEC5 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A3A5C, #0D2137);
        border: 1px solid #2196F3;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricValue"] { color: #E0E8F0 !important; }
    [data-testid="stMetricLabel"] { color: #90CAF9 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #B0BEC5; }
    .stTabs [aria-selected="true"] { color: #2196F3 !important; }

    /* Dataframes */
    .stDataFrame { border: 1px solid #1A3A5C; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Database Path ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "astro_sentiment.db"


@st.cache_resource
def get_db_connection():
    """Cached database connection for the dashboard."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── Plotly Theme ─────────────────────────────────────────────────────

NASA_COLORS = [
    "#2196F3",  # Blue
    "#00BCD4",  # Cyan
    "#4CAF50",  # Green
    "#FFC107",  # Amber
    "#FF5722",  # Deep Orange
    "#9C27B0",  # Purple
    "#607D8B",  # Blue Grey
]

NASA_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,29,51,0.8)",
    font=dict(color="#B0BEC5", family="Roboto, sans-serif"),
    xaxis=dict(gridcolor="#1A3A5C", zerolinecolor="#1A3A5C"),
    yaxis=dict(gridcolor="#1A3A5C", zerolinecolor="#1A3A5C"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=40, r=40, t=50, b=40),
)

# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("\U0001f6f8 Astro Sentiment")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Analyzing NASA astronaut communications for sentiment drift over mission duration.**

Two data sources:
- ISS Blog (operational tone)
- Oral History Transcripts (personal sentiment)
"""
)

# Show data stats
try:
    conn = get_db_connection()
    blog_count = conn.execute("SELECT COUNT(*) FROM blog_posts").fetchone()[0]
    oh_count = conn.execute(
        "SELECT COUNT(*) FROM oral_histories WHERE speaker='astronaut'"
    ).fetchone()[0]
    st.sidebar.markdown("---")
    st.sidebar.metric("Blog Posts", f"{blog_count:,}")
    st.sidebar.metric("Oral History Segments", f"{oh_count:,}")
except Exception:
    st.sidebar.warning("Database not found. Run scrapers first.")

# ── Main Page ────────────────────────────────────────────────────────

st.title("Astronaut Log Sentiment Analyzer")
st.markdown(
    """
Welcome to the Astronaut Log Sentiment Analyzer. This dashboard explores
sentiment patterns across two NASA text corpora:

- **ISS Blog Archive** — Operational communications written during active missions
- **NASA Oral Histories** — Personal retrospective interviews with ISS participants

Use the sidebar to navigate between analysis views.
"""
)

st.info(
    "Navigate to the analysis pages using the sidebar. "
    "Each page offers a different analytical lens on the data.",
    icon="\u2139\ufe0f",
)
