"""Page 4: Oral Histories Deep Dive — Per-astronaut sentiment profiles and key quotes."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.app import NASA_COLORS, NASA_LAYOUT, get_db_connection

st.header("Oral Histories Deep Dive")
st.markdown(
    "Personal sentiment analysis of NASA oral history transcripts. "
    "Each interview segment is analyzed for emotional content."
)

conn = get_db_connection()

# ── Get Available Astronauts ─────────────────────────────────────────

astronauts_query = """
SELECT DISTINCT oh.astronaut_name, COUNT(*) as segment_count
FROM oral_histories oh
JOIN sentiment_results sr ON sr.source_type='oral_history' AND sr.source_id=oh.id
WHERE oh.speaker = 'astronaut'
GROUP BY oh.astronaut_name
ORDER BY oh.astronaut_name
"""
astronauts_df = pd.read_sql_query(astronauts_query, conn)

if astronauts_df.empty:
    st.warning(
        "No analyzed oral history data found. Run the scraper and analysis pipeline first."
    )
    st.stop()

# ── Astronaut Selector ───────────────────────────────────────────────

selected = st.selectbox(
    "Select Astronaut",
    options=astronauts_df["astronaut_name"].tolist(),
    format_func=lambda x: f"{x} ({astronauts_df[astronauts_df['astronaut_name']==x]['segment_count'].iloc[0]} segments)",
)

# ── Load Data for Selected Astronaut ─────────────────────────────────

query = """
SELECT oh.segment_index, oh.speaker, oh.text, oh.word_count,
       sr.label, sr.positive_score, sr.negative_score, sr.neutral_score,
       er.dominant_emotion, er.anger_score, er.disgust_score, er.fear_score,
       er.joy_score, er.neutral_score as emo_neutral, er.sadness_score, er.surprise_score
FROM oral_histories oh
JOIN sentiment_results sr ON sr.source_type='oral_history' AND sr.source_id=oh.id
JOIN emotion_results er ON er.source_type='oral_history' AND er.source_id=oh.id
WHERE oh.astronaut_name = ? AND oh.speaker = 'astronaut'
ORDER BY oh.segment_index
"""

df = pd.read_sql_query(query, conn, params=(selected,))

if df.empty:
    st.warning(f"No analyzed segments for {selected}.")
    st.stop()

# ── Summary Metrics ──────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Segments Analyzed", len(df))
with col2:
    st.metric("Avg Positive", f"{df['positive_score'].mean():.3f}")
with col3:
    st.metric("Avg Negative", f"{df['negative_score'].mean():.3f}")
with col4:
    dominant = df["dominant_emotion"].mode().iloc[0] if not df["dominant_emotion"].empty else "N/A"
    st.metric("Dominant Emotion", dominant.title())

# ── Chart 1: Sentiment Arc ──────────────────────────────────────────

st.subheader("Sentiment Arc Across Interview")

fig_arc = go.Figure()

fig_arc.add_trace(
    go.Scatter(
        x=df["segment_index"],
        y=df["positive_score"],
        name="Positive",
        line=dict(color="#4CAF50", width=2),
        hovertemplate="Segment %{x}<br>Positive: %{y:.3f}<br>%{customdata}<extra></extra>",
        customdata=df["text"].str[:100] + "...",
    )
)
fig_arc.add_trace(
    go.Scatter(
        x=df["segment_index"],
        y=df["negative_score"],
        name="Negative",
        line=dict(color="#FF5722", width=2),
        hovertemplate="Segment %{x}<br>Negative: %{y:.3f}<extra></extra>",
    )
)
fig_arc.add_trace(
    go.Scatter(
        x=df["segment_index"],
        y=df["neutral_score"],
        name="Neutral",
        line=dict(color="#607D8B", width=1, dash="dot"),
        hovertemplate="Segment %{x}<br>Neutral: %{y:.3f}<extra></extra>",
    )
)

fig_arc.update_layout(
    **NASA_LAYOUT,
    title=f"Sentiment Progression — {selected}",
    xaxis_title="Interview Segment",
    yaxis_title="Score",
    height=400,
    legend=dict(orientation="h", y=-0.15),
)

st.plotly_chart(fig_arc, use_container_width=True)

# ── Chart 2: Emotion Progression ────────────────────────────────────

st.subheader("Emotion Progression")

emotion_cols = [
    ("anger_score", "Anger", "#FF5722"),
    ("fear_score", "Fear", "#9C27B0"),
    ("joy_score", "Joy", "#4CAF50"),
    ("sadness_score", "Sadness", "#2196F3"),
    ("surprise_score", "Surprise", "#FFC107"),
]

fig_emotions = go.Figure()

for col_name, label, color in emotion_cols:
    fig_emotions.add_trace(
        go.Scatter(
            x=df["segment_index"],
            y=df[col_name],
            name=label,
            stackgroup="one",
            line=dict(color=color, width=0.5),
            fillcolor=color.replace(")", ", 0.3)").replace("rgb", "rgba")
            if "rgb" in color
            else color,
            hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
        )
    )

fig_emotions.update_layout(
    **NASA_LAYOUT,
    title=f"Emotion Stack — {selected}",
    xaxis_title="Interview Segment",
    yaxis_title="Cumulative Score",
    height=400,
    legend=dict(orientation="h", y=-0.15),
)

st.plotly_chart(fig_emotions, use_container_width=True)

# ── Quotes Table ─────────────────────────────────────────────────────

st.subheader("Key Quotes")

# Emotion filter
emotion_filter = st.multiselect(
    "Filter by emotion",
    options=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    default=[],
)

quotes_df = df[["segment_index", "text", "dominant_emotion", "positive_score", "negative_score"]].copy()
quotes_df["text_preview"] = quotes_df["text"].str[:250] + "..."

if emotion_filter:
    quotes_df = quotes_df[quotes_df["dominant_emotion"].isin(emotion_filter)]

# Sort by strongest sentiment signal (furthest from neutral)
quotes_df["sentiment_strength"] = abs(quotes_df["positive_score"] - quotes_df["negative_score"])
quotes_df = quotes_df.sort_values("sentiment_strength", ascending=False)

st.dataframe(
    quotes_df[["segment_index", "text_preview", "dominant_emotion", "positive_score", "negative_score"]].rename(
        columns={
            "segment_index": "Segment",
            "text_preview": "Quote",
            "dominant_emotion": "Emotion",
            "positive_score": "Pos Score",
            "negative_score": "Neg Score",
        }
    ),
    use_container_width=True,
    hide_index=True,
    height=400,
)
