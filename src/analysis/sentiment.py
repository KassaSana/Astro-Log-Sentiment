"""Sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment-latest."""

import logging
from datetime import datetime

from transformers import pipeline

from src.data.models import SentimentResult

logger = logging.getLogger(__name__)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Map model output labels to our schema
LABEL_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    # Some model versions use different label formats
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


class SentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        logger.info(f"Loading sentiment model: {model_name}")
        self.model_name = model_name
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device,
            top_k=None,  # Return all class scores
        )
        self.tokenizer = self.pipe.tokenizer
        logger.info("Sentiment model loaded.")

    def analyze_text(
        self,
        text: str,
        source_type: str,
        source_id: int,
    ) -> SentimentResult:
        """Analyze a single text chunk (must be ≤ 512 tokens).

        Returns a SentimentResult with scores for all 3 classes.
        """
        results = self.pipe(text, truncation=True, max_length=512)
        # results is a list of dicts: [{"label": "positive", "score": 0.95}, ...]

        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for item in results:
            mapped_label = LABEL_MAP.get(item["label"], item["label"].lower())
            if mapped_label in scores:
                scores[mapped_label] = item["score"]

        label = max(scores, key=scores.get)

        return SentimentResult(
            source_type=source_type,
            source_id=source_id,
            label=label,
            positive_score=scores["positive"],
            negative_score=scores["negative"],
            neutral_score=scores["neutral"],
            model_name=self.model_name,
            analyzed_at=datetime.utcnow(),
        )
