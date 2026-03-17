"""Data models for Kala."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Tense(str, Enum):
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    AMBIGUOUS = "ambiguous"


class TemporalBiasType(str, Enum):
    RECENCY = "recency"
    ANCHORING = "anchoring"
    TELESCOPING = "telescoping"
    NONE = "none"


class TemporalProbe(BaseModel):
    """A probe question testing temporal understanding."""
    question: str
    expected_tense: Optional[Tense] = None
    expected_ordering: Optional[list[str]] = None
    expected_duration: Optional[str] = None


class TemporalResponse(BaseModel):
    """An LLM response to a temporal probe."""
    probe: TemporalProbe
    response_text: str
    detected_tense: Optional[Tense] = None
    is_correct: Optional[bool] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class OrderingResult(BaseModel):
    """Result of an event ordering test."""
    events: list[str]
    expected_order: list[str]
    predicted_order: list[str]
    kendall_tau: float = Field(default=0.0, ge=-1.0, le=1.0)
    is_correct: bool = False


class DurationResult(BaseModel):
    """Result of a duration estimation test."""
    event: str
    expected_duration_seconds: float
    estimated_duration_seconds: float
    absolute_error: float = 0.0
    log_ratio_error: float = 0.0


class ConsistencyResult(BaseModel):
    """Result of temporal consistency checking."""
    text: str
    contradictions: list[str] = Field(default_factory=list)
    is_consistent: bool = True


class BiasResult(BaseModel):
    """Result of temporal bias detection."""
    text: str
    bias_type: TemporalBiasType
    evidence: str = ""
    severity: float = Field(default=0.0, ge=0.0, le=1.0)


class TemporalReport(BaseModel):
    """Full temporal analysis report."""
    perception_results: list[TemporalResponse] = Field(default_factory=list)
    ordering_results: list[OrderingResult] = Field(default_factory=list)
    duration_results: list[DurationResult] = Field(default_factory=list)
    consistency_results: list[ConsistencyResult] = Field(default_factory=list)
    bias_results: list[BiasResult] = Field(default_factory=list)
