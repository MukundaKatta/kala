"""Duration estimation testing for LLMs."""

from __future__ import annotations

import math
import re

from kala.models import DurationResult


# Common durations in seconds for reference
_REFERENCE_DURATIONS = {
    "blinking": 0.3,
    "deep breath": 4.0,
    "boiling an egg": 600.0,
    "flight new york to london": 25200.0,
    "human pregnancy": 23328000.0,
    "earth orbit around sun": 31557600.0,
    "bachelor degree": 126230400.0,
}

_TIME_UNIT_SECONDS = {
    "millisecond": 0.001,
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
    "month": 2592000.0,
    "year": 31557600.0,
}


class DurationEstimator:
    """Tests LLM understanding of time durations.

    Probes whether models can estimate how long events take and
    evaluates accuracy against known reference durations.
    """

    def __init__(self) -> None:
        self.references = dict(_REFERENCE_DURATIONS)

    def parse_duration_from_text(self, text: str) -> float | None:
        """Extract a duration in seconds from free-text LLM response."""
        pattern = r"(\d+(?:\.\d+)?)\s*(millisecond|second|minute|hour|day|week|month|year)s?"
        matches = re.findall(pattern, text.lower())
        if not matches:
            return None

        total = 0.0
        for value_str, unit in matches:
            value = float(value_str)
            total += value * _TIME_UNIT_SECONDS.get(unit, 1.0)
        return total

    def evaluate(
        self, event: str, expected_seconds: float, estimated_seconds: float
    ) -> DurationResult:
        """Evaluate a duration estimate against expected value."""
        absolute_error = abs(estimated_seconds - expected_seconds)
        if expected_seconds > 0 and estimated_seconds > 0:
            log_ratio = abs(math.log10(estimated_seconds / expected_seconds))
        else:
            log_ratio = float("inf") if estimated_seconds != expected_seconds else 0.0

        return DurationResult(
            event=event,
            expected_duration_seconds=expected_seconds,
            estimated_duration_seconds=estimated_seconds,
            absolute_error=absolute_error,
            log_ratio_error=round(log_ratio, 4),
        )

    def generate_probes(self) -> list[tuple[str, float]]:
        """Generate standard duration estimation probes."""
        return [(event, duration) for event, duration in self.references.items()]
