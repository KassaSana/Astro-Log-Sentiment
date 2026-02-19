"""Shared pytest fixtures for the astro-sentiment test suite."""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.db import SCHEMA_SQL, get_connection, init_db, insert_expeditions, load_expeditions
from src.data.models import (
    BlogPost,
    EmotionResult,
    Expedition,
    LinguisticFeatures,
    OralHistorySegment,
    SentimentResult,
)

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Database Fixtures ────────────────────────────────────────────────


@pytest.fixture
def db_conn():
    """In-memory SQLite connection with schema initialized."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def db_with_expeditions(db_conn):
    """In-memory DB pre-loaded with a few test expeditions."""
    expeditions = [
        Expedition(
            number=64,
            name="Expedition 64",
            start_date=date(2020, 10, 21),
            end_date=date(2021, 4, 17),
            crew=["Kate Rubins", "Sergey Ryzhikov", "Sergey Kud-Sverchkov"],
        ),
        Expedition(
            number=65,
            name="Expedition 65",
            start_date=date(2021, 4, 9),
            end_date=date(2021, 10, 17),
            crew=["Shannon Walker", "Michael Hopkins", "Victor Glover", "Soichi Noguchi"],
        ),
        Expedition(
            number=66,
            name="Expedition 66",
            start_date=date(2021, 10, 5),
            end_date=date(2022, 3, 30),
            crew=["Anton Shkaplerov", "Pyotr Dubrov", "Mark Vande Hei"],
        ),
    ]
    insert_expeditions(db_conn, expeditions)
    return db_conn, expeditions


@pytest.fixture
def tmp_db(tmp_path):
    """A temporary on-disk SQLite database."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    return db_path, conn


# ── Sample Data Fixtures ─────────────────────────────────────────────


@pytest.fixture
def sample_blog_post():
    """A minimal valid BlogPost."""
    return BlogPost(
        url="https://blogs.nasa.gov/spacestation/2021/06/15/crew-conducts-science/",
        title="Crew Conducts Science Aboard the Station",
        author="Mark Garcia",
        published_date=date(2021, 6, 15),
        text=(
            "The Expedition 65 crew focused on science operations today aboard "
            "the International Space Station. The astronauts conducted a series "
            "of microgravity experiments in the Destiny laboratory module. "
            "Commander Shannon Walker oversaw the operations while microbiologist "
            "results were analyzed. The crew also performed routine maintenance."
        ),
        word_count=47,
        expedition_id=65,
    )


@pytest.fixture

def sample_oral_history():
    """A minimal valid OralHistorySegment."""
    return OralHistorySegment(
        astronaut_name="Peggy A. Whitson",
        pdf_url="https://www.nasa.gov/example/whitson_transcript.pdf",
        interview_date=date(2018, 5, 10),
        segment_index=3,
        speaker="astronaut",
        text=(
            "When I first floated through the hatch into the station, I was overwhelmed "
            "by the view of Earth. The cupola offers this incredible panoramic view, and "
            "I just sat there for about ten minutes staring. You never get used to it. "
            "Every orbit, the light changes and it's like seeing it for the first time again."
        ),
        word_count=60,
    )


@pytest.fixture
def sample_sentiment_result():
    """A minimal valid SentimentResult."""
    return SentimentResult(
        source_type="blog",
        source_id=1,
        label="neutral",
        positive_score=0.15,
        negative_score=0.05,
        neutral_score=0.80,
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )


@pytest.fixture
def sample_emotion_result():
    """A minimal valid EmotionResult."""
    return EmotionResult(
        source_type="oral_history",
        source_id=3,
        anger_score=0.02,
        disgust_score=0.01,
        fear_score=0.03,
        joy_score=0.65,
        neutral_score=0.20,
        sadness_score=0.04,
        surprise_score=0.05,
        dominant_emotion="joy",
        model_name="j-hartmann/emotion-english-distilroberta-base",
    )


@pytest.fixture
def sample_linguistic_features():
    """A minimal valid LinguisticFeatures."""
    return LinguisticFeatures(
        source_type="blog",
        source_id=1,
        flesch_reading_ease=45.2,
        avg_sentence_length=18.5,
        lexical_diversity=0.72,
        first_person_ratio=0.03,
        exclamation_count=0,
        question_count=1,
    )


# ── Mock HuggingFace Pipeline ───────────────────────────────────────


@pytest.fixture
def mock_tokenizer():
    """A fake tokenizer that splits on whitespace."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(
        side_effect=lambda text, add_special_tokens=False: text.split()
    )
    tokenizer.decode = MagicMock(
        side_effect=lambda ids, skip_special_tokens=True: " ".join(ids)
    )
    return tokenizer


@pytest.fixture
def mock_sentiment_pipeline():
    """A mock sentiment HuggingFace pipeline returning fixed scores."""

    def _pipeline(text, truncation=True, max_length=512):
        return [
            {"label": "positive", "score": 0.20},
            {"label": "neutral", "score": 0.65},
            {"label": "negative", "score": 0.15},
        ]

    pipe = MagicMock(side_effect=_pipeline)
    pipe.tokenizer = MagicMock()
    pipe.tokenizer.encode = MagicMock(
        side_effect=lambda text, add_special_tokens=False: text.split()
    )
    pipe.tokenizer.decode = MagicMock(
        side_effect=lambda ids, skip_special_tokens=True: " ".join(ids)
    )
    return pipe


@pytest.fixture
def mock_emotion_pipeline():
    """A mock emotion HuggingFace pipeline returning fixed scores."""

    def _pipeline(text, truncation=True, max_length=512):
        return [
            {"label": "joy", "score": 0.50},
            {"label": "neutral", "score": 0.25},
            {"label": "surprise", "score": 0.10},
            {"label": "sadness", "score": 0.05},
            {"label": "anger", "score": 0.04},
            {"label": "fear", "score": 0.03},
            {"label": "disgust", "score": 0.03},
        ]

    pipe = MagicMock(side_effect=_pipeline)
    pipe.tokenizer = MagicMock()
    pipe.tokenizer.encode = MagicMock(
        side_effect=lambda text, add_special_tokens=False: text.split()
    )
    pipe.tokenizer.decode = MagicMock(
        side_effect=lambda ids, skip_special_tokens=True: " ".join(ids)
    )
    return pipe


# ── HTML Fixtures for Scraper Tests ──────────────────────────────────


SAMPLE_LISTING_HTML = """
<html>
<body>
<div class="content">
  <article class="post">
    <h2 class="entry-title">
      <a href="https://blogs.nasa.gov/spacestation/2021/06/15/science-day/">Science Day on Station</a>
    </h2>
    <time class="entry-date" datetime="2021-06-15T10:30:00+00:00">June 15, 2021</time>
    <span class="byline"><a href="/author/garcia/">Mark Garcia</a></span>
  </article>
  <article class="post">
    <h2 class="entry-title">
      <a href="https://blogs.nasa.gov/spacestation/2021/06/14/spacewalk-prep/">Spacewalk Preparations</a>
    </h2>
    <time class="entry-date" datetime="2021-06-14T09:00:00+00:00">June 14, 2021</time>
    <span class="byline"><a href="/author/garcia/">Mark Garcia</a></span>
  </article>
</div>
<div class="pagination">
  <a class="page-numbers" href="/page/1/">1</a>
  <a class="page-numbers" href="/page/2/">2</a>
  <a class="page-numbers" href="/page/50/">50</a>
</div>
</body>
</html>
"""

SAMPLE_POST_HTML = """
<html>
<body>
<article>
  <h1 class="entry-title">Science Day on Station</h1>
  <time class="entry-date" datetime="2021-06-15T10:30:00+00:00">June 15, 2021</time>
  <span class="byline"><a href="/author/garcia/">Mark Garcia</a></span>
  <div class="entry-content">
    <p>The Expedition 65 crew focused on science operations today aboard the International Space Station.</p>
    <p>The astronauts conducted a series of microgravity experiments in the Destiny laboratory module. Commander Shannon Walker oversaw the operations while flight engineers worked on protein crystal growth experiments.</p>
    <p>In addition to the science work, the crew also prepared for an upcoming spacewalk scheduled for later this week. The spacewalk will focus on upgrading the station's power systems.</p>
  </div>
</article>
</body>
</html>
"""

SAMPLE_TRANSCRIPT_TEXT = """
NASA Johnson Space Center Oral History Project
Edited Oral History Transcript

WRIGHT: Today is May 10 2018. I am Rebecca Wright and we are speaking with Dr. Peggy Whitson at NASA Johnson Space Center.

WRIGHT: Thank you for joining us today. Can you tell us about your first experience arriving at the International Space Station?

WHITSON: When I first floated through the hatch into the station I was overwhelmed by the view of Earth. The cupola offers this incredible panoramic view and I just sat there for about ten minutes staring. You never get used to it. Every orbit the light changes and it is like seeing it for the first time again. I remember calling down to mission control and telling them that no photograph could ever capture what we were seeing.

WRIGHT: How did the daily routine work on the station?

WHITSON: We had a very structured day. We would wake up and check systems first thing. Then we had a conference with mission control about the day's activities. Most of the day was spent on experiments or maintenance. We always made time for exercise because in microgravity your bones and muscles start to weaken without it. I usually ran on the treadmill for about an hour.

WRIGHT: Thank you so much for sharing your experiences with us today.
"""
