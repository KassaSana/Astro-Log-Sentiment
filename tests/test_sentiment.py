"""Tests for the sentiment analysis module (mocked — no model download)."""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Mock transformers before importing the module under test
sys.modules.setdefault("transformers", MagicMock())

from src.analysis.sentiment import LABEL_MAP, SentimentAnalyzer  # noqa: E402
from src.data.models import SentimentResult  # noqa: E402


@pytest.fixture
def mock_analyzer(mock_sentiment_pipeline):
    """A SentimentAnalyzer with a mocked HuggingFace pipeline."""
    with patch.object(SentimentAnalyzer, "__init__", lambda self, **kw: None):
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        analyzer.pipe = mock_sentiment_pipeline
        analyzer.tokenizer = mock_sentiment_pipeline.tokenizer
    return analyzer


class TestLabelMap:
    def test_standard_labels(self):
        assert LABEL_MAP["negative"] == "negative"
        assert LABEL_MAP["neutral"] == "neutral"
        assert LABEL_MAP["positive"] == "positive"

    def test_fallback_labels(self):
        assert LABEL_MAP["LABEL_0"] == "negative"
        assert LABEL_MAP["LABEL_1"] == "neutral"
        assert LABEL_MAP["LABEL_2"] == "positive"


class TestSentimentAnalyzer:
    def test_analyze_returns_sentiment_result(self, mock_analyzer):
        result = mock_analyzer.analyze_text("A great day in space.", "blog", 1)
        assert isinstance(result, SentimentResult)

    def test_label_is_dominant_score(self, mock_analyzer):
        result = mock_analyzer.analyze_text("Neutral update from station.", "blog", 1)
        # Mock returns neutral=0.65 as highest
        assert result.label == "neutral"

    def test_scores_match_mock(self, mock_analyzer):
        result = mock_analyzer.analyze_text("Some text.", "blog", 1)
        assert result.positive_score == pytest.approx(0.20)
        assert result.neutral_score == pytest.approx(0.65)
        assert result.negative_score == pytest.approx(0.15)

    def test_source_fields_propagated(self, mock_analyzer):
        result = mock_analyzer.analyze_text("Text.", "oral_history", 42)
        assert result.source_type == "oral_history"
        assert result.source_id == 42

    def test_model_name_propagated(self, mock_analyzer):
        result = mock_analyzer.analyze_text("Text.", "blog", 1)
        assert result.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def test_analyzed_at_is_set(self, mock_analyzer):
        result = mock_analyzer.analyze_text("Text.", "blog", 1)
        assert isinstance(result.analyzed_at, datetime)

    def test_pipeline_called_with_truncation(self, mock_analyzer):
        mock_analyzer.analyze_text("Some text.", "blog", 1)
        mock_analyzer.pipe.assert_called_once_with(
            "Some text.", truncation=True, max_length=512
        )

    def test_handles_label_0_format(self):
        """Verify LABEL_0/1/2 format from some model versions is mapped correctly."""

        def alt_pipeline(text, truncation=True, max_length=512):
            return [
                {"label": "LABEL_2", "score": 0.80},
                {"label": "LABEL_1", "score": 0.15},
                {"label": "LABEL_0", "score": 0.05},
            ]

        with patch.object(SentimentAnalyzer, "__init__", lambda self, **kw: None):
            analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
            analyzer.model_name = "test-model"
            analyzer.pipe = MagicMock(side_effect=alt_pipeline)
            analyzer.tokenizer = MagicMock()

        result = analyzer.analyze_text("Great news!", "blog", 1)
        assert result.label == "positive"
        assert result.positive_score == pytest.approx(0.80)
        assert result.negative_score == pytest.approx(0.05)
