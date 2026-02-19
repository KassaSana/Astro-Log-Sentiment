"""Emotion detection using j-hartmann/emotion-english-distilroberta-base."""

import logging
from datetime import datetime

from transformers import pipeline

from src.data.models import EmotionResult

logger = logging.getLogger(__name__)

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class EmotionAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        logger.info(f"Loading emotion model: {model_name}")
        self.model_name = model_name
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
            top_k=None,  # Return all 7 emotion scores
        )
        self.tokenizer = self.pipe.tokenizer
        logger.info("Emotion model loaded.")

    def analyze_text(
        self,
        text: str,
        source_type: str,
        source_id: int,
    ) -> EmotionResult:
        """Analyze a single text chunk (must be ≤ 512 tokens).

        Returns an EmotionResult with scores for all 7 emotions.
        """
        results = self.pipe(text, truncation=True, max_length=512)

        scores = {e: 0.0 for e in EMOTION_LABELS}
        for item in results:
            label = item["label"].lower()
            if label in scores:
                scores[label] = item["score"]

        dominant = max(scores, key=scores.get)

        return EmotionResult(
            source_type=source_type,
            source_id=source_id,
            anger_score=scores["anger"],
            disgust_score=scores["disgust"],
            fear_score=scores["fear"],
            joy_score=scores["joy"],
            neutral_score=scores["neutral"],
            sadness_score=scores["sadness"],
            surprise_score=scores["surprise"],
            dominant_emotion=dominant,
            model_name=self.model_name,
            analyzed_at=datetime.utcnow(),
        )
