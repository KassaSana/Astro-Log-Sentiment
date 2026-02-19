"""Page 3: Expedition Compare — Side-by-side expedition analysis with word clouds."""

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from src.dashboard.app import NASA_COLORS, NASA_LAYOUT, get_db_connection

st.header("Expedition Compare")
st.markdown("Side-by-side comparison of two ISS expeditions.")

conn = get_db_connection()

# ── Get Available Expeditions ────────────────────────────────────────

exp_query = """
SELECT DISTINCT bp.expedition_id, e.name
FROM blog_posts bp
JOIN expeditions e ON e.number = bp.expedition_id
WHERE bp.expedition_id IS NOT NULL
ORDER BY bp.expedition_id
"""
expeditions = pd.read_sql_query(exp_query, conn)

if expeditions.empty:
    st.warning("No expedition data found. Run the scraper and analysis pipeline first.")
    st.stop()

exp_options = dict(zip(expeditions["name"], expeditions["expedition_id"]))

# ── Expedition Selectors ─────────────────────────────────────────────

col_select_1, col_select_2 = st.columns(2)
exp_names = list(exp_options.keys())
with col_select_1:
    exp_a_name = st.selectbox("Expedition A", options=exp_names, index=0)
with col_select_2:
    default_b = min(1, len(exp_names) - 1)
    exp_b_name = st.selectbox("Expedition B", options=exp_names, index=default_b)

exp_a_id = exp_options[exp_a_name]
exp_b_id = exp_options[exp_b_name]


# ── Data Loader ──────────────────────────────────────────────────────

def load_expedition_data(exp_id: int) -> pd.DataFrame:
    query = """
    SELECT bp.text, bp.word_count, bp.title,
           sr.positive_score, sr.negative_score, sr.neutral_score, sr.label,
           er.dominant_emotion, er.joy_score, er.fear_score, er.surprise_score,
           lf.flesch_reading_ease, lf.avg_sentence_length,
           lf.lexical_diversity, lf.first_person_ratio
    FROM blog_posts bp
    JOIN sentiment_results sr ON sr.source_type='blog' AND sr.source_id=bp.id
    JOIN emotion_results er ON er.source_type='blog' AND er.source_id=bp.id
    LEFT JOIN linguistic_features lf ON lf.source_type='blog' AND lf.source_id=bp.id
    WHERE bp.expedition_id = ?
    """
    return pd.read_sql_query(query, conn, params=(exp_id,))


df_a = load_expedition_data(exp_a_id)
df_b = load_expedition_data(exp_b_id)


def generate_wordcloud(texts: pd.Series) -> bytes:
    """Generate a word cloud image and return as PNG bytes."""
    all_text = " ".join(texts.dropna())
    wc = WordCloud(
        width=500,
        height=300,
        background_color="#0B1D33",
        colormap="cool",
        max_words=80,
        contour_width=1,
        contour_color="#2196F3",
    ).generate(all_text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()


# ── Side-by-Side Display ─────────────────────────────────────────────

col_a, col_b = st.columns(2)

for col, df, exp_name in [(col_a, df_a, exp_a_name), (col_b, df_b, exp_b_name)]:
    with col:
        st.subheader(exp_name)

        if df.empty:
            st.warning("No data available for this expedition.")
            continue

        # Metric cards
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Posts", len(df))
            st.metric("Avg Positive", f"{df['positive_score'].mean():.3f}")
        with m2:
            dominant = df["dominant_emotion"].mode().iloc[0] if not df["dominant_emotion"].empty else "N/A"
            st.metric("Dominant Emotion", dominant.title())
            st.metric("Avg Negative", f"{df['negative_score'].mean():.3f}")

        # Word cloud
        st.markdown("**Word Cloud**")
        if not df["text"].empty:
            wc_bytes = generate_wordcloud(df["text"])
            st.image(wc_bytes, use_container_width=True)

        # Linguistic features comparison bar chart
        if "flesch_reading_ease" in df.columns and df["flesch_reading_ease"].notna().any():
            st.markdown("**Linguistic Features**")
            features = {
                "Readability": df["flesch_reading_ease"].mean(),
                "Avg Sent. Len": df["avg_sentence_length"].mean(),
                "Lexical Div.": df["lexical_diversity"].mean() * 100,  # Scale for visibility
                "1st Person %": df["first_person_ratio"].mean() * 100,
            }

            fig_ling = go.Figure(
                go.Bar(
                    x=list(features.values()),
                    y=list(features.keys()),
                    orientation="h",
                    marker_color=NASA_COLORS[0],
                    hovertemplate="%{y}: %{x:.1f}<extra></extra>",
                )
            )
            fig_ling.update_layout(
                **NASA_LAYOUT,
                height=250,
                margin=dict(l=100, r=20, t=20, b=20),
                xaxis_title="Score",
            )
            st.plotly_chart(fig_ling, use_container_width=True)
