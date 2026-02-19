"""Tests for the SQLite database layer."""

import json
import sqlite3
from datetime import date, datetime

import pytest

from src.data.db import (
    SCHEMA_SQL,
    get_connection,
    get_unanalyzed,
    init_db,
    insert_blog_post,
    insert_emotion,
    insert_expeditions,
    insert_linguistic,
    insert_oral_history,
    insert_sentiment,
    load_expeditions,
    map_date_to_expedition,
)
from src.data.models import (
    BlogPost,
    EmotionResult,
    Expedition,
    LinguisticFeatures,
    OralHistorySegment,
    SentimentResult,
)


# ── Schema Initialization ────────────────────────────────────────────


class TestSchemaInit:
    def test_init_creates_all_tables(self, db_conn):
        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = sorted(
            row["name"] for row in tables
            if not row["name"].startswith("sqlite_")  # exclude internal tables
        )
        expected = sorted([
            "blog_posts",
            "emotion_results",
            "expeditions",
            "linguistic_features",
            "oral_histories",
            "sentiment_results",
        ])
        assert table_names == expected

    def test_init_is_idempotent(self, db_conn):
        """Running init_db twice should not raise or duplicate tables."""
        init_db(db_conn)  # already initialized via fixture; call again
        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        user_tables = [t for t in tables if not t["name"].startswith("sqlite_")]
        assert len(user_tables) == 6

    def test_indexes_created(self, db_conn):
        indexes = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        index_names = {row["name"] for row in indexes}
        assert "idx_blog_posts_date" in index_names
        assert "idx_blog_posts_expedition" in index_names
        assert "idx_sentiment_source" in index_names

    def test_foreign_keys_enabled(self, db_conn):
        result = db_conn.execute("PRAGMA foreign_keys").fetchone()
        assert result[0] == 1


# ── get_connection ───────────────────────────────────────────────────


class TestGetConnection:
    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "deep" / "nested" / "test.db"
        conn = get_connection(db_path)
        assert db_path.parent.exists()
        conn.close()

    def test_row_factory_set(self, tmp_path):
        conn = get_connection(tmp_path / "test.db")
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_wal_mode(self, tmp_path):
        conn = get_connection(tmp_path / "test.db")
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()


# ── Expeditions ──────────────────────────────────────────────────────


class TestExpeditions:
    def test_insert_and_query(self, db_conn):
        expeditions = [
            Expedition(
                number=65,
                name="Expedition 65",
                start_date=date(2021, 4, 9),
                end_date=date(2021, 10, 17),
                crew=["Shannon Walker"],
            ),
        ]
        count = insert_expeditions(db_conn, expeditions)
        assert count == 1

        row = db_conn.execute(
            "SELECT * FROM expeditions WHERE number = 65"
        ).fetchone()
        assert row["name"] == "Expedition 65"
        assert json.loads(row["crew"]) == ["Shannon Walker"]

    def test_insert_ignore_duplicates(self, db_conn):
        exp = Expedition(
            number=1,
            name="Exp 1",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 6, 1),
            crew=["A"],
        )
        insert_expeditions(db_conn, [exp])
        insert_expeditions(db_conn, [exp])  # duplicate
        rows = db_conn.execute("SELECT COUNT(*) FROM expeditions").fetchone()[0]
        assert rows == 1

    def test_load_expeditions_from_json(self, tmp_path):
        data = [
            {
                "number": 99,
                "name": "Expedition 99",
                "start_date": "2030-01-01",
                "end_date": "2030-06-01",
                "crew": ["Alice", "Bob"],
            }
        ]
        json_path = tmp_path / "expeditions.json"
        json_path.write_text(json.dumps(data))

        result = load_expeditions(json_path)
        assert len(result) == 1
        assert result[0].number == 99
        assert result[0].crew == ["Alice", "Bob"]


# ── map_date_to_expedition ───────────────────────────────────────────


class TestMapDateToExpedition:
    def test_date_within_expedition(self, db_with_expeditions):
        _, expeditions = db_with_expeditions
        # 2021-06-15 falls within Expedition 65 (2021-04-09 to 2021-10-17)
        result = map_date_to_expedition(date(2021, 6, 15), expeditions)
        assert result == 65

    def test_date_during_handover_returns_later_expedition(self, db_with_expeditions):
        _, expeditions = db_with_expeditions
        # 2021-04-12 overlaps Exp 64 (ends 2021-04-17) and Exp 65 (starts 2021-04-09)
        result = map_date_to_expedition(date(2021, 4, 12), expeditions)
        assert result == 65  # higher number wins

    def test_date_outside_all_expeditions(self, db_with_expeditions):
        _, expeditions = db_with_expeditions
        result = map_date_to_expedition(date(2025, 1, 1), expeditions)
        assert result is None

    def test_date_on_start_boundary(self, db_with_expeditions):
        _, expeditions = db_with_expeditions
        result = map_date_to_expedition(date(2020, 10, 21), expeditions)
        assert result == 64

    def test_date_on_end_boundary(self, db_with_expeditions):
        _, expeditions = db_with_expeditions
        result = map_date_to_expedition(date(2022, 3, 30), expeditions)
        assert result == 66


# ── Blog Post Insert ─────────────────────────────────────────────────


class TestInsertBlogPost:
    @pytest.fixture(autouse=True)
    def _seed_expedition(self, db_conn):
        """Insert the expedition referenced by sample_blog_post (id=65)."""
        db_conn.execute(
            "INSERT OR IGNORE INTO expeditions (number, name, start_date, end_date, crew) "
            "VALUES (65, 'Expedition 65', '2021-04-09', '2021-10-17', '[\"A\"]')"
        )
        db_conn.commit()

    def test_insert_returns_positive_rowid(self, db_conn, sample_blog_post):
        row_id = insert_blog_post(db_conn, sample_blog_post)
        assert row_id > 0

    def test_insert_stores_all_fields(self, db_conn, sample_blog_post):
        insert_blog_post(db_conn, sample_blog_post)
        row = db_conn.execute("SELECT * FROM blog_posts WHERE id = 1").fetchone()
        assert row["title"] == sample_blog_post.title
        assert row["author"] == "Mark Garcia"
        assert row["word_count"] == 47
        assert row["expedition_id"] == 65

    def test_duplicate_url_returns_negative(self, db_conn, sample_blog_post):
        insert_blog_post(db_conn, sample_blog_post)
        result = insert_blog_post(db_conn, sample_blog_post)
        assert result == -1

    def test_different_urls_both_inserted(self, db_conn, sample_blog_post):
        insert_blog_post(db_conn, sample_blog_post)
        post2 = sample_blog_post.model_copy(
            update={"url": "https://blogs.nasa.gov/spacestation/2021/06/16/other/"}
        )
        row_id = insert_blog_post(db_conn, post2)
        assert row_id > 0
        count = db_conn.execute("SELECT COUNT(*) FROM blog_posts").fetchone()[0]
        assert count == 2


# ── Oral History Insert ──────────────────────────────────────────────


class TestInsertOralHistory:
    def test_insert_returns_rowid(self, db_conn, sample_oral_history):
        row_id = insert_oral_history(db_conn, sample_oral_history)
        assert row_id > 0

    def test_insert_stores_speaker(self, db_conn, sample_oral_history):
        insert_oral_history(db_conn, sample_oral_history)
        row = db_conn.execute("SELECT * FROM oral_histories WHERE id = 1").fetchone()
        assert row["speaker"] == "astronaut"
        assert row["astronaut_name"] == "Peggy A. Whitson"

    def test_multiple_segments_same_astronaut(self, db_conn, sample_oral_history):
        insert_oral_history(db_conn, sample_oral_history)
        seg2 = sample_oral_history.model_copy(update={"segment_index": 4})
        insert_oral_history(db_conn, seg2)
        count = db_conn.execute("SELECT COUNT(*) FROM oral_histories").fetchone()[0]
        assert count == 2


# ── Analysis Result Inserts ──────────────────────────────────────────


class TestInsertSentiment:
    def test_insert(self, db_conn, sample_sentiment_result):
        row_id = insert_sentiment(db_conn, sample_sentiment_result)
        assert row_id > 0

    def test_idempotent_insert_or_ignore(self, db_conn, sample_sentiment_result):
        insert_sentiment(db_conn, sample_sentiment_result)
        insert_sentiment(db_conn, sample_sentiment_result)  # duplicate
        count = db_conn.execute("SELECT COUNT(*) FROM sentiment_results").fetchone()[0]
        assert count == 1

    def test_different_models_both_stored(self, db_conn, sample_sentiment_result):
        insert_sentiment(db_conn, sample_sentiment_result)
        result2 = sample_sentiment_result.model_copy(
            update={"model_name": "other-model"}
        )
        insert_sentiment(db_conn, result2)
        count = db_conn.execute("SELECT COUNT(*) FROM sentiment_results").fetchone()[0]
        assert count == 2


class TestInsertEmotion:
    def test_insert(self, db_conn, sample_emotion_result):
        row_id = insert_emotion(db_conn, sample_emotion_result)
        assert row_id > 0

    def test_stores_dominant_emotion(self, db_conn, sample_emotion_result):
        insert_emotion(db_conn, sample_emotion_result)
        row = db_conn.execute("SELECT * FROM emotion_results WHERE id = 1").fetchone()
        assert row["dominant_emotion"] == "joy"


class TestInsertLinguistic:
    def test_insert(self, db_conn, sample_linguistic_features):
        row_id = insert_linguistic(db_conn, sample_linguistic_features)
        assert row_id > 0

    def test_idempotent(self, db_conn, sample_linguistic_features):
        insert_linguistic(db_conn, sample_linguistic_features)
        insert_linguistic(db_conn, sample_linguistic_features)
        count = db_conn.execute(
            "SELECT COUNT(*) FROM linguistic_features"
        ).fetchone()[0]
        assert count == 1


# ── get_unanalyzed ───────────────────────────────────────────────────


class TestGetUnanalyzed:
    @pytest.fixture(autouse=True)
    def _seed_expedition(self, db_conn):
        """Insert the expedition referenced by sample_blog_post (id=65)."""
        db_conn.execute(
            "INSERT OR IGNORE INTO expeditions (number, name, start_date, end_date, crew) "
            "VALUES (65, 'Expedition 65', '2021-04-09', '2021-10-17', '[\"A\"]')"
        )
        db_conn.commit()

    def test_returns_rows_without_analysis(self, db_conn, sample_blog_post):
        insert_blog_post(db_conn, sample_blog_post)
        rows = get_unanalyzed(
            db_conn,
            "blog_posts",
            "sentiment_results",
            "blog",
            "test-model",
        )
        assert len(rows) == 1
        assert rows[0]["id"] == 1

    def test_returns_empty_after_analysis(
        self, db_conn, sample_blog_post, sample_sentiment_result
    ):
        insert_blog_post(db_conn, sample_blog_post)
        insert_sentiment(db_conn, sample_sentiment_result)
        rows = get_unanalyzed(
            db_conn,
            "blog_posts",
            "sentiment_results",
            "blog",
            sample_sentiment_result.model_name,
        )
        assert len(rows) == 0

    def test_linguistic_no_model_name(
        self, db_conn, sample_blog_post, sample_linguistic_features
    ):
        insert_blog_post(db_conn, sample_blog_post)
        # Before inserting linguistic features
        rows = get_unanalyzed(
            db_conn, "blog_posts", "linguistic_features", "blog", model_name=None
        )
        assert len(rows) == 1

        # After inserting linguistic features
        insert_linguistic(db_conn, sample_linguistic_features)
        rows = get_unanalyzed(
            db_conn, "blog_posts", "linguistic_features", "blog", model_name=None
        )
        assert len(rows) == 0
