"""Tests for the emotion detection module (mocked — no model download)."""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Mock transformers before importing the module under test
sys.modules.setdefault("transformers", MagicMock())

from src.analysis.emotion import EMOTION_LABELS, EmotionAnalyzer  # noqa: E402
from src.data.models import EmotionResult  # noqa: E402


@pytest.fixture
def mock_analyzer(mock_emotion_pipeline):
    """An EmotionAnalyzer with a mocked HuggingFace pipeline."""
    with patch.object(EmotionAnalyzer, "__init__", lambda self, **kw: None):
        analyzer = EmotionAnalyzer.__new__(EmotionAnalyzer)
        analyzer.model_name = "j-hartmann/emotion-english-distilroberta-base"
        analyzer.pipe = mock_emotion_pipeline
        analyzer.tokenizer = mock_emotion_pipeline.tokenizer
    return analyzer


class TestEmotionLabels:
    def test_seven_emotions(self):
        assert len(EMOTION_LABELS) == 7
        assert set(EMOTION_LABELS) == {
            "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
        }


class TestEmotionAnalyzer:
    def test_returns_emotion_result(self, mock_analyzer):
        result = mock_analyzer.analyze_text("What a beautiful view!", "oral_history", 5)
        assert isinstance(result, EmotionResult)

    def test_dominant_emotion_is_highest_score(self, mock_analyzer):
        result = mock_analyzer.analyze_text("Text", "blog", 1)
        # Mock returns joy=0.50 as highest
        assert result.dominant_emotion == "joy"

    def test_all_scores_populated(self, mock_analyzer):
        result = mock_analyzer.analyze_text("Text", "blog", 1)
        assert result.joy_score == pytest.approx(0.50)
        assert result.neutral_score == pytest.approx(0.25)
        assert result.surprise_score == pytest.approx(0.10)
        assert result.sadness_score == pytest.approx(0.05)
        assert result.anger_score == pytest.approx(0.04)
        assert result.fear_score == pytest.approx(0.03)
        assert result.disgust_score == pytest.approx(0.03)

    def test_source_fields(self, mock_analyzer):
        result = mock_analyzer.analyze_text("Text", "oral_history", 99)
        assert result.source_type == "oral_history"
        assert result.source_id == 99

    def test_model_name_propagated(self, mock_analyzer):
        result = mock_analyzer.analyze_text("Text", "blog", 1)
        assert result.model_name == "j-hartmann/emotion-english-distilroberta-base"

    def test_analyzed_at_set(self, mock_analyzer):
        result = mock_analyzer.analyze_text("Text", "blog", 1)
        assert isinstance(result.analyzed_at, datetime)

    def test_pipeline_called_with_truncation(self, mock_analyzer):
        mock_analyzer.analyze_text("Hello world", "blog", 1)
        mock_analyzer.pipe.assert_called_once_with(
            "Hello world", truncation=True, max_length=512
        )

    def test_handles_unknown_label_gracefully(self):
        """Unknown labels should be ignored without crashing."""

        def extra_label_pipeline(text, truncation=True, max_length=512):
            return [
                {"label": "joy", "score": 0.60},
                {"label": "unknown_emotion", "score": 0.30},
                {"label": "anger", "score": 0.10},
            ]

        with patch.object(EmotionAnalyzer, "__init__", lambda self, **kw: None):
            analyzer = EmotionAnalyzer.__new__(EmotionAnalyzer)
            analyzer.model_name = "test-model"
            analyzer.pipe = MagicMock(side_effect=extra_label_pipeline)
            analyzer.tokenizer = MagicMock()

        result = analyzer.analyze_text("Surprising!", "blog", 1)
        assert result.dominant_emotion == "joy"
        assert result.joy_score == pytest.approx(0.60)
        # unknown_emotion is not in EMOTION_LABELS, should be skipped
        assert result.anger_score == pytest.approx(0.10)
