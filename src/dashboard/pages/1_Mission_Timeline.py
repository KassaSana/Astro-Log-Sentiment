"""Page 1: Mission Timeline — Sentiment over time with expedition boundaries."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.app import NASA_COLORS, NASA_LAYOUT, get_db_connection

st.header("Mission Timeline")
st.markdown("Tracking operational tone across the ISS blog archive over time.")

conn = get_db_connection()

# ── Load Data ────────────────────────────────────────────────────────

query = """
SELECT bp.published_date, bp.expedition_id, bp.title,
       sr.positive_score, sr.negative_score, sr.neutral_score, sr.label,
       er.anger_score, er.disgust_score, er.fear_score, er.joy_score,
       er.neutral_score as emo_neutral, er.sadness_score, er.surprise_score,
       er.dominant_emotion
FROM blog_posts bp
JOIN sentiment_results sr ON sr.source_type='blog' AND sr.source_id=bp.id
JOIN emotion_results er ON er.source_type='blog' AND er.source_id=bp.id
ORDER BY bp.published_date
"""

df = pd.read_sql_query(query, conn)

if df.empty:
    st.warning("No analyzed blog data found. Run the scraper and analysis pipeline first.")
    st.stop()

df["published_date"] = pd.to_datetime(df["published_date"])

# ── Controls ─────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 3])
with col1:
    window = st.radio(
        "Rolling Average Window",
        options=[7, 14, 30, 60],
        index=2,
        format_func=lambda x: f"{x} days",
    )

# ── Chart 1: Sentiment Time Series ──────────────────────────────────

st.subheader("Sentiment Over Time (Rolling Average)")

df_sorted = df.sort_values("published_date").set_index("published_date")
rolling_pos = df_sorted["positive_score"].rolling(f"{window}D").mean()
rolling_neg = df_sorted["negative_score"].rolling(f"{window}D").mean()
rolling_neu = df_sorted["neutral_score"].rolling(f"{window}D").mean()

fig_sentiment = go.Figure()

fig_sentiment.add_trace(
    go.Scatter(
        x=rolling_pos.index,
        y=rolling_pos.values,
        name="Positive",
        line=dict(color="#4CAF50", width=2),
        hovertemplate="Date: %{x}<br>Positive: %{y:.3f}<extra></extra>",
    )
)
fig_sentiment.add_trace(
    go.Scatter(
        x=rolling_neg.index,
        y=rolling_neg.values,
        name="Negative",
        line=dict(color="#FF5722", width=2),
        hovertemplate="Date: %{x}<br>Negative: %{y:.3f}<extra></extra>",
    )
)
fig_sentiment.add_trace(
    go.Scatter(
        x=rolling_neu.index,
        y=rolling_neu.values,
        name="Neutral",
        line=dict(color="#607D8B", width=1, dash="dot"),
        hovertemplate="Date: %{x}<br>Neutral: %{y:.3f}<extra></extra>",
    )
)

# Add expedition boundary lines
if "expedition_id" in df.columns:
    expedition_boundaries = (
        df.dropna(subset=["expedition_id"])
        .groupby("expedition_id")["published_date"]
        .min()
    )
    for exp_id, start_date in expedition_boundaries.items():
        fig_sentiment.add_vline(
            x=start_date,
            line_dash="dash",
            line_color="rgba(33, 150, 243, 0.3)",
            annotation_text=f"E{int(exp_id)}",
            annotation_position="top",
            annotation_font_size=9,
            annotation_font_color="#90CAF9",
        )

fig_sentiment.update_layout(
    **NASA_LAYOUT,
    title=f"Blog Post Sentiment ({window}-Day Rolling Average)",
    yaxis_title="Score",
    xaxis_title="Date",
    height=450,
    legend=dict(orientation="h", y=-0.15),
)

st.plotly_chart(fig_sentiment, use_container_width=True)

# ── Chart 2: Emotion Heatmap ────────────────────────────────────────

st.subheader("Emotion Heatmap Over Time")

emotions = ["anger_score", "disgust_score", "fear_score", "joy_score",
            "emo_neutral", "sadness_score", "surprise_score"]
emotion_labels = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]

# Resample to monthly means
df_monthly = df.set_index("published_date")[emotions].resample("ME").mean().dropna()

if not df_monthly.empty:
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=df_monthly.values.T,
            x=df_monthly.index.strftime("%Y-%m"),
            y=emotion_labels,
            colorscale=[
                [0, "#0B1D33"],
                [0.25, "#1A3A5C"],
                [0.5, "#2196F3"],
                [0.75, "#00BCD4"],
                [1.0, "#4CAF50"],
            ],
            hovertemplate="Month: %{x}<br>Emotion: %{y}<br>Score: %{z:.3f}<extra></extra>",
        )
    )

    fig_heatmap.update_layout(
        **NASA_LAYOUT,
        title="Monthly Average Emotion Scores",
        xaxis_title="Month",
        yaxis_title="Emotion",
        height=350,
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

# ── Summary Stats ────────────────────────────────────────────────────

st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Posts Analyzed", f"{len(df):,}")
with col2:
    st.metric("Avg Positive Score", f"{df['positive_score'].mean():.3f}")
with col3:
    st.metric("Avg Negative Score", f"{df['negative_score'].mean():.3f}")
with col4:
    most_common = df["dominant_emotion"].mode().iloc[0] if not df["dominant_emotion"].empty else "N/A"
    st.metric("Most Common Emotion", most_common.title())
