"""Orchestrate sentiment, emotion, and linguistic analysis across all data."""

import logging
from datetime import datetime

from tqdm import tqdm

from src.data.db import (
    get_connection,
    get_unanalyzed,
    init_db,
    insert_emotion,
    insert_linguistic,
    insert_sentiment,
)
from src.data.models import EmotionResult, SentimentResult

from .emotion import EmotionAnalyzer
from .linguistic import LinguisticAnalyzer
from .sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)

MIN_WORDS_FOR_ANALYSIS = 10


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = 400,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping token windows for models with 512-token limits.

    Uses a sliding window approach:
    - Window size: max_tokens (400, leaving room for special tokens)
    - Step size: max_tokens - overlap (300)
    - Each window is decoded back to a text string

    Returns a list of text strings, each within the token limit.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        return [text]

    chunks = []
    step = max_tokens - overlap
    for start in range(0, len(token_ids), step):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_str = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_str)
        if end == len(token_ids):
            break

    return chunks


def aggregate_sentiment(
    results: list[SentimentResult],
    source_type: str,
    source_id: int,
    model_name: str,
) -> SentimentResult:
    """Aggregate sentiment scores from multiple chunks by averaging."""
    n = len(results)
    pos = sum(r.positive_score for r in results) / n
    neg = sum(r.negative_score for r in results) / n
    neu = sum(r.neutral_score for r in results) / n

    scores = {"positive": pos, "negative": neg, "neutral": neu}
    label = max(scores, key=scores.get)

    return SentimentResult(
        source_type=source_type,
        source_id=source_id,
        label=label,
        positive_score=pos,
        negative_score=neg,
        neutral_score=neu,
        model_name=model_name,
        analyzed_at=datetime.utcnow(),
    )


def aggregate_emotions(
    results: list[EmotionResult],
    source_type: str,
    source_id: int,
    model_name: str,
) -> EmotionResult:
    """Aggregate emotion scores from multiple chunks by averaging."""
    n = len(results)
    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    means = {}
    for e in emotions:
        means[e] = sum(getattr(r, f"{e}_score") for r in results) / n

    dominant = max(means, key=means.get)

    return EmotionResult(
        source_type=source_type,
        source_id=source_id,
        anger_score=means["anger"],
        disgust_score=means["disgust"],
        fear_score=means["fear"],
        joy_score=means["joy"],
        neutral_score=means["neutral"],
        sadness_score=means["sadness"],
        surprise_score=means["surprise"],
        dominant_emotion=dominant,
        model_name=model_name,
        analyzed_at=datetime.utcnow(),
    )


def _analyze_rows(
    rows,
    source_type: str,
    sentiment_analyzer: SentimentAnalyzer | None,
    emotion_analyzer: EmotionAnalyzer | None,
    linguistic_analyzer: LinguisticAnalyzer | None,
    conn,
) -> None:
    """Run all enabled analyzers on a list of database rows."""
    for row in tqdm(rows, desc=f"Analyzing {source_type}"):
        row_id = row["id"]
        text = row["text"]
        word_count = row["word_count"]

        if word_count < MIN_WORDS_FOR_ANALYSIS:
            continue

        # Sentiment analysis with chunking
        if sentiment_analyzer:
            try:
                chunks = chunk_text(text, sentiment_analyzer.tokenizer)
                chunk_results = [
                    sentiment_analyzer.analyze_text(c, source_type, row_id)
                    for c in chunks
                ]
                if len(chunk_results) == 1:
                    result = chunk_results[0]
                else:
                    result = aggregate_sentiment(
                        chunk_results,
                        source_type,
                        row_id,
                        sentiment_analyzer.model_name,
                    )
                insert_sentiment(conn, result)
            except Exception as e:
                logger.error(f"Sentiment failed for {source_type}:{row_id}: {e}")

        # Emotion analysis with chunking
        if emotion_analyzer:
            try:
                chunks = chunk_text(text, emotion_analyzer.tokenizer)
                chunk_results = [
                    emotion_analyzer.analyze_text(c, source_type, row_id)
                    for c in chunks
                ]
                if len(chunk_results) == 1:
                    result = chunk_results[0]
                else:
                    result = aggregate_emotions(
                        chunk_results,
                        source_type,
                        row_id,
                        emotion_analyzer.model_name,
                    )
                insert_emotion(conn, result)
            except Exception as e:
                logger.error(f"Emotion failed for {source_type}:{row_id}: {e}")

        # Linguistic analysis (no chunking needed)
        if linguistic_analyzer:
            try:
                features = linguistic_analyzer.analyze(text, source_type, row_id)
                insert_linguistic(conn, features)
            except Exception as e:
                logger.error(f"Linguistic failed for {source_type}:{row_id}: {e}")


def run_analysis(
    db_path: str,
    skip_sentiment: bool = False,
    skip_emotion: bool = False,
    skip_linguistic: bool = False,
    device: str = "cpu",
) -> None:
    """Run the full analysis pipeline on all unanalyzed data.

    Loads models, finds unanalyzed rows, runs all three analysis types,
    and stores results in the database. Idempotent via UNIQUE constraints.
    """
    conn = get_connection(db_path)
    init_db(conn)

    # Initialize analyzers
    sentiment_analyzer = None
    emotion_analyzer = None
    linguistic_analyzer = None

    if not skip_sentiment:
        sentiment_analyzer = SentimentAnalyzer(device=device)
    if not skip_emotion:
        emotion_analyzer = EmotionAnalyzer(device=device)
    if not skip_linguistic:
        linguistic_analyzer = LinguisticAnalyzer()

    # Process blog posts
    logger.info("Finding unanalyzed blog posts...")
    if sentiment_analyzer:
        blog_rows = get_unanalyzed(
            conn, "blog_posts", "sentiment_results", "blog", sentiment_analyzer.model_name
        )
    elif emotion_analyzer:
        blog_rows = get_unanalyzed(
            conn, "blog_posts", "emotion_results", "blog", emotion_analyzer.model_name
        )
    elif linguistic_analyzer:
        blog_rows = get_unanalyzed(
            conn, "blog_posts", "linguistic_features", "blog"
        )
    else:
        blog_rows = []

    if blog_rows:
        logger.info(f"Analyzing {len(blog_rows)} blog posts...")
        _analyze_rows(
            blog_rows,
            "blog",
            sentiment_analyzer,
            emotion_analyzer,
            linguistic_analyzer,
            conn,
        )

    # Process oral history segments (astronaut segments only)
    logger.info("Finding unanalyzed oral history segments...")
    if sentiment_analyzer:
        oh_rows = get_unanalyzed(
            conn,
            "oral_histories",
            "sentiment_results",
            "oral_history",
            sentiment_analyzer.model_name,
        )
    elif emotion_analyzer:
        oh_rows = get_unanalyzed(
            conn,
            "oral_histories",
            "emotion_results",
            "oral_history",
            emotion_analyzer.model_name,
        )
    elif linguistic_analyzer:
        oh_rows = get_unanalyzed(
            conn, "oral_histories", "linguistic_features", "oral_history"
        )
    else:
        oh_rows = []

    # Filter to astronaut segments only (skip interviewer questions)
    oh_rows = [r for r in oh_rows if r["speaker"] == "astronaut"]

    if oh_rows:
        logger.info(f"Analyzing {len(oh_rows)} oral history segments...")
        _analyze_rows(
            oh_rows,
            "oral_history",
            sentiment_analyzer,
            emotion_analyzer,
            linguistic_analyzer,
            conn,
        )

    logger.info("Analysis pipeline complete.")
    conn.close()
