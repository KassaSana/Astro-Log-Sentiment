"""SQLite database setup and helper functions."""

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path

from .models import (
    BlogPost,
    EmotionResult,
    Expedition,
    LinguisticFeatures,
    OralHistorySegment,
    SentimentResult,
)

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS expeditions (
    number          INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    crew            TEXT NOT NULL,
    patch_url       TEXT
);

CREATE TABLE IF NOT EXISTS blog_posts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    author          TEXT,
    published_date  TEXT NOT NULL,
    text            TEXT NOT NULL,
    word_count      INTEGER NOT NULL,
    expedition_id   INTEGER,
    scraped_at      TEXT NOT NULL,
    FOREIGN KEY (expedition_id) REFERENCES expeditions(number)
);

CREATE TABLE IF NOT EXISTS oral_histories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    astronaut_name  TEXT NOT NULL,
    pdf_url         TEXT NOT NULL,
    interview_date  TEXT,
    segment_index   INTEGER NOT NULL,
    speaker         TEXT NOT NULL,
    text            TEXT NOT NULL,
    word_count      INTEGER NOT NULL,
    scraped_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sentiment_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type     TEXT NOT NULL CHECK(source_type IN ('blog', 'oral_history')),
    source_id       INTEGER NOT NULL,
    label           TEXT NOT NULL,
    positive_score  REAL NOT NULL,
    negative_score  REAL NOT NULL,
    neutral_score   REAL NOT NULL,
    model_name      TEXT NOT NULL,
    analyzed_at     TEXT NOT NULL,
    UNIQUE(source_type, source_id, model_name)
);

CREATE TABLE IF NOT EXISTS emotion_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type     TEXT NOT NULL CHECK(source_type IN ('blog', 'oral_history')),
    source_id       INTEGER NOT NULL,
    anger_score     REAL NOT NULL,
    disgust_score   REAL NOT NULL,
    fear_score      REAL NOT NULL,
    joy_score       REAL NOT NULL,
    neutral_score   REAL NOT NULL,
    sadness_score   REAL NOT NULL,
    surprise_score  REAL NOT NULL,
    dominant_emotion TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    analyzed_at     TEXT NOT NULL,
    UNIQUE(source_type, source_id, model_name)
);

CREATE TABLE IF NOT EXISTS linguistic_features (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type         TEXT NOT NULL CHECK(source_type IN ('blog', 'oral_history')),
    source_id           INTEGER NOT NULL,
    flesch_reading_ease REAL NOT NULL,
    avg_sentence_length REAL NOT NULL,
    lexical_diversity   REAL NOT NULL,
    first_person_ratio  REAL NOT NULL,
    exclamation_count   INTEGER NOT NULL,
    question_count      INTEGER NOT NULL,
    analyzed_at         TEXT NOT NULL,
    UNIQUE(source_type, source_id)
);

CREATE INDEX IF NOT EXISTS idx_blog_posts_date ON blog_posts(published_date);
CREATE INDEX IF NOT EXISTS idx_blog_posts_expedition ON blog_posts(expedition_id);
CREATE INDEX IF NOT EXISTS idx_oral_histories_astronaut ON oral_histories(astronaut_name);
CREATE INDEX IF NOT EXISTS idx_sentiment_source ON sentiment_results(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_emotion_source ON emotion_results(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_linguistic_source ON linguistic_features(source_type, source_id);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with row factory enabled."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they don't exist."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    logger.info("Database schema initialized.")


# ── Expeditions ──────────────────────────────────────────────────────


def load_expeditions(json_path: str | Path) -> list[Expedition]:
    """Load expedition metadata from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return [Expedition(**item) for item in data]


def insert_expeditions(conn: sqlite3.Connection, expeditions: list[Expedition]) -> int:
    """Insert expeditions into the database. Returns count inserted."""
    count = 0
    for exp in expeditions:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO expeditions (number, name, start_date, end_date, crew, patch_url) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    exp.number,
                    exp.name,
                    exp.start_date.isoformat(),
                    exp.end_date.isoformat(),
                    json.dumps(exp.crew),
                    exp.patch_url,
                ),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return count


def map_date_to_expedition(post_date: date, expeditions: list[Expedition]) -> int | None:
    """Map a blog post date to an expedition number.

    If the date falls during a crew handover (overlapping expeditions),
    returns the later expedition number.
    """
    matches = [e for e in expeditions if e.start_date <= post_date <= e.end_date]
    if not matches:
        return None
    return max(matches, key=lambda e: e.number).number


# ── Blog Posts ───────────────────────────────────────────────────────


def insert_blog_post(conn: sqlite3.Connection, post: BlogPost) -> int:
    """Insert a blog post. Returns rowid or -1 if duplicate."""
    try:
        cursor = conn.execute(
            "INSERT INTO blog_posts (url, title, author, published_date, text, word_count, expedition_id, scraped_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                post.url,
                post.title,
                post.author,
                post.published_date.isoformat(),
                post.text,
                post.word_count,
                post.expedition_id,
                post.scraped_at.isoformat(),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        logger.debug(f"Skipping duplicate blog post: {post.url}")
        return -1


# ── Oral Histories ───────────────────────────────────────────────────


def insert_oral_history(conn: sqlite3.Connection, segment: OralHistorySegment) -> int:
    """Insert an oral history segment. Returns rowid."""
    cursor = conn.execute(
        "INSERT INTO oral_histories (astronaut_name, pdf_url, interview_date, segment_index, speaker, text, word_count, scraped_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            segment.astronaut_name,
            segment.pdf_url,
            segment.interview_date.isoformat() if segment.interview_date else None,
            segment.segment_index,
            segment.speaker,
            segment.text,
            segment.word_count,
            segment.scraped_at.isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


# ── Analysis Results ─────────────────────────────────────────────────


def insert_sentiment(conn: sqlite3.Connection, result: SentimentResult) -> int:
    """Insert a sentiment result. Uses INSERT OR IGNORE for idempotency."""
    cursor = conn.execute(
        "INSERT OR IGNORE INTO sentiment_results "
        "(source_type, source_id, label, positive_score, negative_score, neutral_score, model_name, analyzed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            result.source_type,
            result.source_id,
            result.label,
            result.positive_score,
            result.negative_score,
            result.neutral_score,
            result.model_name,
            result.analyzed_at.isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def insert_emotion(conn: sqlite3.Connection, result: EmotionResult) -> int:
    """Insert an emotion result. Uses INSERT OR IGNORE for idempotency."""
    cursor = conn.execute(
        "INSERT OR IGNORE INTO emotion_results "
        "(source_type, source_id, anger_score, disgust_score, fear_score, joy_score, "
        "neutral_score, sadness_score, surprise_score, dominant_emotion, model_name, analyzed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            result.source_type,
            result.source_id,
            result.anger_score,
            result.disgust_score,
            result.fear_score,
            result.joy_score,
            result.neutral_score,
            result.sadness_score,
            result.surprise_score,
            result.dominant_emotion,
            result.model_name,
            result.analyzed_at.isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def insert_linguistic(conn: sqlite3.Connection, features: LinguisticFeatures) -> int:
    """Insert linguistic features. Uses INSERT OR IGNORE for idempotency."""
    cursor = conn.execute(
        "INSERT OR IGNORE INTO linguistic_features "
        "(source_type, source_id, flesch_reading_ease, avg_sentence_length, "
        "lexical_diversity, first_person_ratio, exclamation_count, question_count, analyzed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            features.source_type,
            features.source_id,
            features.flesch_reading_ease,
            features.avg_sentence_length,
            features.lexical_diversity,
            features.first_person_ratio,
            features.exclamation_count,
            features.question_count,
            features.analyzed_at.isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


# ── Queries ──────────────────────────────────────────────────────────


def get_unanalyzed(
    conn: sqlite3.Connection,
    source_table: str,
    analysis_table: str,
    source_type: str,
    model_name: str | None = None,
) -> list[sqlite3.Row]:
    """Get rows from source_table that have no matching row in analysis_table.

    For linguistic_features (no model_name column), pass model_name=None.
    """
    if model_name:
        query = f"""
            SELECT s.* FROM {source_table} s
            LEFT JOIN {analysis_table} a
                ON a.source_type = ? AND a.source_id = s.id AND a.model_name = ?
            WHERE a.id IS NULL
        """
        return conn.execute(query, (source_type, model_name)).fetchall()
    else:
        query = f"""
            SELECT s.* FROM {source_table} s
            LEFT JOIN {analysis_table} a
                ON a.source_type = ? AND a.source_id = s.id
            WHERE a.id IS NULL
        """
        return conn.execute(query, (source_type,)).fetchall()
