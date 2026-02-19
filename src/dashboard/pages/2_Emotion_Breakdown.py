"""Page 2: Emotion Breakdown — Per-expedition emotion profiles."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.app import NASA_COLORS, NASA_LAYOUT, get_db_connection

st.header("Emotion Breakdown")
st.markdown("Emotional fingerprints of ISS expeditions based on blog post analysis.")

conn = get_db_connection()

# ── Load Data ────────────────────────────────────────────────────────

query = """
SELECT bp.expedition_id, e.name as expedition_name,
       AVG(er.anger_score) as avg_anger,
       AVG(er.disgust_score) as avg_disgust,
       AVG(er.fear_score) as avg_fear,
       AVG(er.joy_score) as avg_joy,
       AVG(er.neutral_score) as avg_neutral,
       AVG(er.sadness_score) as avg_sadness,
       AVG(er.surprise_score) as avg_surprise,
       COUNT(*) as post_count
FROM blog_posts bp
JOIN emotion_results er ON er.source_type='blog' AND er.source_id=bp.id
JOIN expeditions e ON e.number=bp.expedition_id
WHERE bp.expedition_id IS NOT NULL
GROUP BY bp.expedition_id
ORDER BY bp.expedition_id
"""

df = pd.read_sql_query(query, conn)

if df.empty:
    st.warning("No expedition emotion data found. Run the scraper and analysis pipeline first.")
    st.stop()

emotions = ["avg_anger", "avg_disgust", "avg_fear", "avg_joy",
            "avg_neutral", "avg_sadness", "avg_surprise"]
emotion_labels = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]

# ── Chart 1: Grouped Bar Chart ──────────────────────────────────────

st.subheader("Emotion Scores by Expedition")

fig_bars = go.Figure()

for i, (col, label) in enumerate(zip(emotions, emotion_labels)):
    fig_bars.add_trace(
        go.Bar(
            name=label,
            x=df["expedition_name"],
            y=df[col],
            marker_color=NASA_COLORS[i],
            hovertemplate=f"{label}: %{{y:.3f}}<br>Posts: %{{customdata}}<extra></extra>",
            customdata=df["post_count"],
        )
    )

fig_bars.update_layout(
    **NASA_LAYOUT,
    barmode="group",
    title="Average Emotion Scores per Expedition",
    xaxis_title="Expedition",
    yaxis_title="Average Score",
    height=500,
    xaxis_tickangle=-45,
    legend=dict(orientation="h", y=-0.25),
)

st.plotly_chart(fig_bars, use_container_width=True)

# ── Chart 2: Radar / Spider Chart ───────────────────────────────────

st.subheader("Expedition Emotion Radar Comparison")

expedition_options = df["expedition_name"].tolist()
selected_expeditions = st.multiselect(
    "Select expeditions to compare (2-4 recommended)",
    options=expedition_options,
    default=expedition_options[:3] if len(expedition_options) >= 3 else expedition_options,
    max_selections=4,
)

if selected_expeditions:
    fig_radar = go.Figure()

    for i, exp_name in enumerate(selected_expeditions):
        row = df[df["expedition_name"] == exp_name].iloc[0]
        values = [row[col] for col in emotions]
        # Close the radar shape
        values.append(values[0])
        labels = emotion_labels + [emotion_labels[0]]

        fig_radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels,
                fill="toself",
                name=exp_name,
                line_color=NASA_COLORS[i % len(NASA_COLORS)],
                fillcolor=f"rgba{tuple(list(int(NASA_COLORS[i % len(NASA_COLORS)].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + [0.15])}",
            )
        )

    fig_radar.update_layout(
        **NASA_LAYOUT,
        polar=dict(
            bgcolor="rgba(11,29,51,0.8)",
            radialaxis=dict(visible=True, gridcolor="#1A3A5C", color="#B0BEC5"),
            angularaxis=dict(gridcolor="#1A3A5C", color="#B0BEC5"),
        ),
        title="Emotion Profile Comparison",
        height=500,
    )

    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Select at least one expedition to see the radar chart.")

# ── Insight Table ────────────────────────────────────────────────────

st.subheader("Expedition Emotion Summary")

# Find dominant non-neutral emotion per expedition
display_df = df.copy()
non_neutral_emotions = ["avg_anger", "avg_disgust", "avg_fear", "avg_joy", "avg_sadness", "avg_surprise"]
non_neutral_labels = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]

display_df["dominant_emotion"] = display_df[non_neutral_emotions].idxmax(axis=1).map(
    dict(zip(non_neutral_emotions, non_neutral_labels))
)

st.dataframe(
    display_df[["expedition_name", "post_count", "dominant_emotion", "avg_joy", "avg_fear", "avg_surprise"]].rename(
        columns={
            "expedition_name": "Expedition",
            "post_count": "Posts",
            "dominant_emotion": "Dominant Emotion",
            "avg_joy": "Avg Joy",
            "avg_fear": "Avg Fear",
            "avg_surprise": "Avg Surprise",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
