"""Custom linguistic feature extraction (rule-based, no ML model)."""

import re
from datetime import datetime

import textstat

from src.data.models import LinguisticFeatures

FIRST_PERSON_PRONOUNS = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
)


class LinguisticAnalyzer:
    def analyze(
        self,
        text: str,
        source_type: str,
        source_id: int,
    ) -> LinguisticFeatures:
        """Compute linguistic features for a text.

        Features:
        - Flesch reading ease (0-100+, higher = easier to read)
        - Average sentence length in words
        - Lexical diversity (type-token ratio, 0-1)
        - First-person pronoun ratio (0-1)
        - Exclamation mark count
        - Question mark count
        """
        words = text.lower().split()
        total_words = len(words)

        # Guard against empty/trivial text
        if total_words < 5:
            return LinguisticFeatures(
                source_type=source_type,
                source_id=source_id,
                flesch_reading_ease=0.0,
                avg_sentence_length=0.0,
                lexical_diversity=0.0,
                first_person_ratio=0.0,
                exclamation_count=0,
                question_count=0,
                analyzed_at=datetime.utcnow(),
            )

        unique_words = len(set(words))

        # Split into sentences on terminal punctuation
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)

        # Count first-person pronouns
        fp_count = sum(1 for w in words if w in FIRST_PERSON_PRONOUNS)

        return LinguisticFeatures(
            source_type=source_type,
            source_id=source_id,
            flesch_reading_ease=textstat.flesch_reading_ease(text),
            avg_sentence_length=total_words / num_sentences,
            lexical_diversity=unique_words / total_words,
            first_person_ratio=fp_count / total_words,
            exclamation_count=text.count("!"),
            question_count=text.count("?"),
            analyzed_at=datetime.utcnow(),
        )
