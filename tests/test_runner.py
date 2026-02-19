"""Tests for the analysis runner: chunking, aggregation, and orchestration."""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Mock transformers before importing runner (which imports sentiment/emotion)
sys.modules.setdefault("transformers", MagicMock())

from src.analysis.runner import (  # noqa: E402
    MIN_WORDS_FOR_ANALYSIS,
    aggregate_emotions,
    aggregate_sentiment,
    chunk_text,
)
from src.data.models import EmotionResult, SentimentResult  # noqa: E402


# ── chunk_text ───────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_returns_single_chunk(self, mock_tokenizer):
        """Text under max_tokens returns the original text as single chunk."""
        text = "Hello world this is short"
        chunks = chunk_text(text, mock_tokenizer, max_tokens=400, overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_returns_multiple_chunks(self, mock_tokenizer):
        """Text over max_tokens is split into overlapping windows."""
        # Create text with 20 words (tokens in our mock)
        words = [f"word{i}" for i in range(20)]
        text = " ".join(words)
        chunks = chunk_text(text, mock_tokenizer, max_tokens=10, overlap=3)
        # step = 10 - 3 = 7; windows: [0:10], [7:17], [14:20]
        assert len(chunks) == 3

    def test_chunks_respect_max_tokens(self, mock_tokenizer):
        words = [f"w{i}" for i in range(50)]
        text = " ".join(words)
        chunks = chunk_text(text, mock_tokenizer, max_tokens=15, overlap=5)
        # Each chunk, when tokenized, should have at most 15 tokens
        for chunk in chunks:
            token_count = len(mock_tokenizer.encode(chunk, add_special_tokens=False))
            assert token_count <= 15

    def test_overlap_preserved(self, mock_tokenizer):
        """Adjacent chunks share overlap tokens."""
        words = [f"w{i}" for i in range(30)]
        text = " ".join(words)
        chunks = chunk_text(text, mock_tokenizer, max_tokens=10, overlap=4)
        # step = 10 - 4 = 6; chunk 0: w0..w9, chunk 1: w6..w15
        tokens_0 = set(mock_tokenizer.encode(chunks[0], add_special_tokens=False))
        tokens_1 = set(mock_tokenizer.encode(chunks[1], add_special_tokens=False))
        overlap_tokens = tokens_0 & tokens_1
        assert len(overlap_tokens) >= 4

    def test_exact_boundary_no_extra_chunk(self, mock_tokenizer):
        """Text with exactly max_tokens words returns one chunk."""
        words = [f"w{i}" for i in range(10)]
        text = " ".join(words)
        chunks = chunk_text(text, mock_tokenizer, max_tokens=10, overlap=3)
        assert len(chunks) == 1

    def test_empty_text(self, mock_tokenizer):
        chunks = chunk_text("", mock_tokenizer, max_tokens=400, overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_single_word(self, mock_tokenizer):
        chunks = chunk_text("hello", mock_tokenizer, max_tokens=400, overlap=100)
        assert len(chunks) == 1


# ── aggregate_sentiment ──────────────────────────────────────────────


class TestAggregateSentiment:
    def test_averages_scores(self):
        results = [
            SentimentResult(
                source_type="blog", source_id=1, label="positive",
                positive_score=0.8, negative_score=0.1, neutral_score=0.1,
                model_name="m",
            ),
            SentimentResult(
                source_type="blog", source_id=1, label="neutral",
                positive_score=0.2, negative_score=0.1, neutral_score=0.7,
                model_name="m",
            ),
        ]
        agg = aggregate_sentiment(results, "blog", 1, "m")
        assert agg.positive_score == pytest.approx(0.5)
        assert agg.neutral_score == pytest.approx(0.4)
        assert agg.negative_score == pytest.approx(0.1)

    def test_label_matches_highest_average(self):
        results = [
            SentimentResult(
                source_type="blog", source_id=1, label="positive",
                positive_score=0.9, negative_score=0.05, neutral_score=0.05,
                model_name="m",
            ),
            SentimentResult(
                source_type="blog", source_id=1, label="positive",
                positive_score=0.7, negative_score=0.2, neutral_score=0.1,
                model_name="m",
            ),
        ]
        agg = aggregate_sentiment(results, "blog", 1, "m")
        assert agg.label == "positive"

    def test_single_result_passthrough(self):
        r = SentimentResult(
            source_type="blog", source_id=1, label="negative",
            positive_score=0.1, negative_score=0.8, neutral_score=0.1,
            model_name="m",
        )
        agg = aggregate_sentiment([r], "blog", 1, "m")
        assert agg.label == "negative"
        assert agg.negative_score == pytest.approx(0.8)

    def test_source_fields_propagated(self):
        r = SentimentResult(
            source_type="oral_history", source_id=42, label="neutral",
            positive_score=0.3, negative_score=0.3, neutral_score=0.4,
            model_name="test-model",
        )
        agg = aggregate_sentiment([r], "oral_history", 42, "test-model")
        assert agg.source_type == "oral_history"
        assert agg.source_id == 42
        assert agg.model_name == "test-model"


# ── aggregate_emotions ───────────────────────────────────────────────


class TestAggregateEmotions:
    def _make_emotion(self, **overrides):
        defaults = dict(
            source_type="blog", source_id=1,
            anger_score=0.0, disgust_score=0.0, fear_score=0.0,
            joy_score=0.0, neutral_score=0.0, sadness_score=0.0,
            surprise_score=0.0, dominant_emotion="neutral", model_name="m",
        )
        defaults.update(overrides)
        return EmotionResult(**defaults)

    def test_averages_all_seven_emotions(self):
        r1 = self._make_emotion(joy_score=0.8, neutral_score=0.2)
        r2 = self._make_emotion(joy_score=0.4, neutral_score=0.6)
        agg = aggregate_emotions([r1, r2], "blog", 1, "m")
        assert agg.joy_score == pytest.approx(0.6)
        assert agg.neutral_score == pytest.approx(0.4)

    def test_dominant_emotion_recalculated(self):
        r1 = self._make_emotion(sadness_score=0.9)
        r2 = self._make_emotion(sadness_score=0.7, joy_score=0.3)
        agg = aggregate_emotions([r1, r2], "blog", 1, "m")
        assert agg.dominant_emotion == "sadness"

    def test_single_result(self):
        r = self._make_emotion(surprise_score=0.95)
        agg = aggregate_emotions([r], "blog", 1, "m")
        assert agg.dominant_emotion == "surprise"
        assert agg.surprise_score == pytest.approx(0.95)

    def test_source_fields(self):
        r = self._make_emotion(source_type="oral_history", source_id=7)
        agg = aggregate_emotions([r], "oral_history", 7, "test")
        assert agg.source_type == "oral_history"
        assert agg.source_id == 7


# ── MIN_WORDS_FOR_ANALYSIS ───────────────────────────────────────────


class TestMinWordsConstant:
    def test_value(self):
        assert MIN_WORDS_FOR_ANALYSIS == 10
