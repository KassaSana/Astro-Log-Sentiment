"""Tests for Pydantic data models."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.data.models import (
    BlogPost,
    EmotionResult,
    Expedition,
    LinguisticFeatures,
    OralHistorySegment,
    SentimentResult,
)


# ── BlogPost ─────────────────────────────────────────────────────────


class TestBlogPost:
    def test_valid_blog_post(self, sample_blog_post):
        assert sample_blog_post.url.startswith("https://")
        assert sample_blog_post.word_count == 47
        assert sample_blog_post.expedition_id == 65

    def test_defaults(self):
        post = BlogPost(
            url="https://example.com/post",
            title="Test",
            published_date=date(2021, 1, 1),
            text="Some text here.",
            word_count=3,
        )
        assert post.author is None
        assert post.expedition_id is None
        assert isinstance(post.scraped_at, datetime)

    def test_negative_word_count_rejected(self):
        with pytest.raises(ValidationError, match="word_count"):
            BlogPost(
                url="https://example.com",
                title="T",
                published_date=date(2021, 1, 1),
                text="text",
                word_count=-1,
            )

    def test_zero_word_count_accepted(self):
        post = BlogPost(
            url="https://example.com",
            title="T",
            published_date=date(2021, 1, 1),
            text="",
            word_count=0,
        )
        assert post.word_count == 0

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            BlogPost(url="https://example.com")


# ── OralHistorySegment ───────────────────────────────────────────────


class TestOralHistorySegment:
    def test_valid_segment(self, sample_oral_history):
        assert sample_oral_history.speaker == "astronaut"
        assert sample_oral_history.segment_index == 3
        assert sample_oral_history.word_count == 60

    def test_speaker_must_be_interviewer_or_astronaut(self):
        with pytest.raises(ValidationError, match="speaker"):
            OralHistorySegment(
                astronaut_name="Test",
                pdf_url="https://example.com/test.pdf",
                segment_index=0,
                speaker="unknown",
                text="Some text here for testing speaker validation.",
                word_count=8,
            )

    def test_interviewer_speaker_accepted(self):
        seg = OralHistorySegment(
            astronaut_name="Test",
            pdf_url="https://example.com/test.pdf",
            segment_index=0,
            speaker="interviewer",
            text="Some question text.",
            word_count=3,
        )
        assert seg.speaker == "interviewer"

    def test_negative_segment_index_rejected(self):
        with pytest.raises(ValidationError, match="segment_index"):
            OralHistorySegment(
                astronaut_name="Test",
                pdf_url="https://example.com/test.pdf",
                segment_index=-1,
                speaker="astronaut",
                text="text",
                word_count=1,
            )

    def test_optional_interview_date(self):
        seg = OralHistorySegment(
            astronaut_name="Test",
            pdf_url="https://example.com/test.pdf",
            segment_index=0,
            speaker="astronaut",
            text="Some text.",
            word_count=2,
        )
        assert seg.interview_date is None


# ── SentimentResult ──────────────────────────────────────────────────


class TestSentimentResult:
    def test_valid_result(self, sample_sentiment_result):
        assert sample_sentiment_result.label == "neutral"
        assert abs(
            sample_sentiment_result.positive_score
            + sample_sentiment_result.negative_score
            + sample_sentiment_result.neutral_score
            - 1.0
        ) < 0.01

    def test_invalid_source_type(self):
        with pytest.raises(ValidationError, match="source_type"):
            SentimentResult(
                source_type="twitter",
                source_id=1,
                label="positive",
                positive_score=0.9,
                negative_score=0.05,
                neutral_score=0.05,
                model_name="test-model",
            )

    def test_invalid_label(self):
        with pytest.raises(ValidationError, match="label"):
            SentimentResult(
                source_type="blog",
                source_id=1,
                label="very_positive",
                positive_score=0.9,
                negative_score=0.05,
                neutral_score=0.05,
                model_name="test-model",
            )

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            SentimentResult(
                source_type="blog",
                source_id=1,
                label="positive",
                positive_score=1.5,
                negative_score=0.0,
                neutral_score=0.0,
                model_name="test-model",
            )

    def test_negative_score_rejected(self):
        with pytest.raises(ValidationError):
            SentimentResult(
                source_type="blog",
                source_id=1,
                label="positive",
                positive_score=-0.1,
                negative_score=0.6,
                neutral_score=0.5,
                model_name="test-model",
            )

    def test_oral_history_source_type(self):
        result = SentimentResult(
            source_type="oral_history",
            source_id=5,
            label="positive",
            positive_score=0.8,
            negative_score=0.1,
            neutral_score=0.1,
            model_name="test-model",
        )
        assert result.source_type == "oral_history"


# ── EmotionResult ────────────────────────────────────────────────────


class TestEmotionResult:
    def test_valid_result(self, sample_emotion_result):
        assert sample_emotion_result.dominant_emotion == "joy"
        assert sample_emotion_result.joy_score == 0.65

    def test_all_seven_emotions_present(self, sample_emotion_result):
        emotions = [
            sample_emotion_result.anger_score,
            sample_emotion_result.disgust_score,
            sample_emotion_result.fear_score,
            sample_emotion_result.joy_score,
            sample_emotion_result.neutral_score,
            sample_emotion_result.sadness_score,
            sample_emotion_result.surprise_score,
        ]
        assert all(0.0 <= s <= 1.0 for s in emotions)

    def test_score_above_1_rejected(self):
        with pytest.raises(ValidationError):
            EmotionResult(
                source_type="blog",
                source_id=1,
                anger_score=1.5,
                disgust_score=0.0,
                fear_score=0.0,
                joy_score=0.0,
                neutral_score=0.0,
                sadness_score=0.0,
                surprise_score=0.0,
                dominant_emotion="anger",
                model_name="test-model",
            )


# ── LinguisticFeatures ──────────────────────────────────────────────


class TestLinguisticFeatures:
    def test_valid_features(self, sample_linguistic_features):
        assert sample_linguistic_features.flesch_reading_ease == 45.2
        assert 0.0 <= sample_linguistic_features.lexical_diversity <= 1.0

    def test_lexical_diversity_out_of_range(self):
        with pytest.raises(ValidationError):
            LinguisticFeatures(
                source_type="blog",
                source_id=1,
                flesch_reading_ease=50.0,
                avg_sentence_length=15.0,
                lexical_diversity=1.5,
                first_person_ratio=0.0,
                exclamation_count=0,
                question_count=0,
            )

    def test_negative_exclamation_count_rejected(self):
        with pytest.raises(ValidationError):
            LinguisticFeatures(
                source_type="blog",
                source_id=1,
                flesch_reading_ease=50.0,
                avg_sentence_length=15.0,
                lexical_diversity=0.5,
                first_person_ratio=0.0,
                exclamation_count=-1,
                question_count=0,
            )

    def test_first_person_ratio_bounds(self):
        # Valid at boundaries
        feat = LinguisticFeatures(
            source_type="oral_history",
            source_id=1,
            flesch_reading_ease=60.0,
            avg_sentence_length=12.0,
            lexical_diversity=0.0,
            first_person_ratio=1.0,
            exclamation_count=0,
            question_count=0,
        )
        assert feat.first_person_ratio == 1.0
        assert feat.lexical_diversity == 0.0


# ── Expedition ───────────────────────────────────────────────────────


class TestExpedition:
    def test_valid_expedition(self):
        exp = Expedition(
            number=65,
            name="Expedition 65",
            start_date=date(2021, 4, 9),
            end_date=date(2021, 10, 17),
            crew=["Shannon Walker", "Michael Hopkins"],
        )
        assert exp.number == 65
        assert len(exp.crew) == 2
        assert exp.patch_url is None

    def test_empty_crew_list(self):
        exp = Expedition(
            number=1,
            name="Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 6, 1),
            crew=[],
        )
        assert exp.crew == []
