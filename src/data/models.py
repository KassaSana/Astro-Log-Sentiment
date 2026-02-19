"""Pydantic data models for the Astronaut Log Sentiment Analyzer."""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class BlogPost(BaseModel):
    url: str
    title: str
    author: Optional[str] = None
    published_date: date
    text: str
    word_count: int = Field(ge=0)
    expedition_id: Optional[int] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class OralHistorySegment(BaseModel):
    astronaut_name: str
    pdf_url: str
    interview_date: Optional[date] = None
    segment_index: int = Field(ge=0)
    speaker: str = Field(pattern=r"^(interviewer|astronaut)$")
    text: str
    word_count: int = Field(ge=0)
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class SentimentResult(BaseModel):
    source_type: str = Field(pattern=r"^(blog|oral_history)$")
    source_id: int
    label: str = Field(pattern=r"^(positive|negative|neutral)$")
    positive_score: float = Field(ge=0.0, le=1.0)
    negative_score: float = Field(ge=0.0, le=1.0)
    neutral_score: float = Field(ge=0.0, le=1.0)
    model_name: str
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class EmotionResult(BaseModel):
    source_type: str = Field(pattern=r"^(blog|oral_history)$")
    source_id: int
    anger_score: float = Field(ge=0.0, le=1.0)
    disgust_score: float = Field(ge=0.0, le=1.0)
    fear_score: float = Field(ge=0.0, le=1.0)
    joy_score: float = Field(ge=0.0, le=1.0)
    neutral_score: float = Field(ge=0.0, le=1.0)
    sadness_score: float = Field(ge=0.0, le=1.0)
    surprise_score: float = Field(ge=0.0, le=1.0)
    dominant_emotion: str
    model_name: str
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class LinguisticFeatures(BaseModel):
    source_type: str = Field(pattern=r"^(blog|oral_history)$")
    source_id: int
    flesch_reading_ease: float
    avg_sentence_length: float = Field(ge=0.0)
    lexical_diversity: float = Field(ge=0.0, le=1.0)
    first_person_ratio: float = Field(ge=0.0, le=1.0)
    exclamation_count: int = Field(ge=0)
    question_count: int = Field(ge=0)
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class Expedition(BaseModel):
    number: int
    name: str
    start_date: date
    end_date: date
    crew: list[str]
    patch_url: Optional[str] = None
