"""Tests for the linguistic feature extraction module."""

from datetime import datetime

import pytest

from src.analysis.linguistic import FIRST_PERSON_PRONOUNS, LinguisticAnalyzer
from src.data.models import LinguisticFeatures


@pytest.fixture
def analyzer():
    return LinguisticAnalyzer()


class TestLinguisticAnalyzer:
    def test_returns_linguistic_features(self, analyzer):
        text = (
            "The astronauts conducted experiments in the laboratory today. "
            "They focused on protein crystal growth and fluid dynamics. "
            "The results were transmitted to mission control for review."
        )
        result = analyzer.analyze(text, "blog", 1)
        assert isinstance(result, LinguisticFeatures)
        assert result.source_type == "blog"
        assert result.source_id == 1

    def test_flesch_reading_ease_range(self, analyzer):
        # Simple text should be easy to read (high score)
        easy_text = "The cat sat on the mat. The dog ran in the park. It was fun."
        result = analyzer.analyze(easy_text, "blog", 1)
        assert result.flesch_reading_ease > 50

    def test_avg_sentence_length(self, analyzer):
        text = "First sentence has five words. Second also has five words."
        result = analyzer.analyze(text, "blog", 1)
        # 10 words / 2 sentences = 5.0
        assert result.avg_sentence_length == pytest.approx(5.0, abs=1.0)

    def test_lexical_diversity(self, analyzer):
        # All unique words → diversity close to 1
        unique_text = "every single word here is completely different and totally unique stuff"
        result_unique = analyzer.analyze(unique_text, "blog", 1)

        # Repeated words → lower diversity
        repeat_text = "the the the the the the the the the the the the"
        result_repeat = analyzer.analyze(repeat_text, "blog", 2)

        assert result_unique.lexical_diversity > result_repeat.lexical_diversity

    def test_first_person_ratio(self, analyzer):
        # Heavy first-person usage
        fp_text = "I went to space. My dream was to explore. I told myself we could do it. Our mission succeeded."
        result = analyzer.analyze(fp_text, "oral_history", 1)
        assert result.first_person_ratio > 0.15

        # No first-person pronouns
        no_fp_text = "The crew performed experiments. The station orbited Earth. Science data was collected."
        result_no = analyzer.analyze(no_fp_text, "blog", 2)
        assert result_no.first_person_ratio == 0.0

    def test_exclamation_count(self, analyzer):
        text = "Amazing! What a view! The Earth is beautiful! Incredible sight from up here."
        result = analyzer.analyze(text, "oral_history", 1)
        assert result.exclamation_count == 3

    def test_question_count(self, analyzer):
        text = "How was the spacewalk? Did you see the aurora? The view was great."
        result = analyzer.analyze(text, "oral_history", 1)
        assert result.question_count == 2

    def test_short_text_returns_zeros(self, analyzer):
        """Text with fewer than 5 words returns all-zero features."""
        short_text = "Hello world."
        result = analyzer.analyze(short_text, "blog", 1)
        assert result.flesch_reading_ease == 0.0
        assert result.avg_sentence_length == 0.0
        assert result.lexical_diversity == 0.0
        assert result.first_person_ratio == 0.0

    def test_empty_text_returns_zeros(self, analyzer):
        result = analyzer.analyze("", "blog", 1)
        assert result.flesch_reading_ease == 0.0

    def test_analyzed_at_is_set(self, analyzer):
        text = "The astronauts will perform a spacewalk tomorrow morning at dawn."
        result = analyzer.analyze(text, "blog", 1)
        assert isinstance(result.analyzed_at, datetime)


class TestFirstPersonPronouns:
    def test_contains_expected_pronouns(self):
        expected = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
        assert FIRST_PERSON_PRONOUNS == expected

    def test_is_frozenset(self):
        assert isinstance(FIRST_PERSON_PRONOUNS, frozenset)
