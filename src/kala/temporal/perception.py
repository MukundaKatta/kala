"""Temporal perception testing for LLMs."""

from __future__ import annotations

import re

from kala.models import Tense, TemporalProbe, TemporalResponse


_PAST_PATTERNS = [
    r"\b(was|were|had|did|went|came|said|made|found|gave|told)\b",
    r"\b(ago|yesterday|last (week|month|year)|previously|formerly)\b",
    r"\b(in \d{4})\b",
    r"\b\w+ed\b",
]
_PRESENT_PATTERNS = [
    r"\b(is|are|am|has|have|do|does|goes|comes)\b",
    r"\b(now|currently|today|at present|right now)\b",
]
_FUTURE_PATTERNS = [
    r"\b(will|shall|going to|gonna)\b",
    r"\b(tomorrow|next (week|month|year)|soon|eventually)\b",
    r"\b(in the future|upcoming)\b",
]


class TemporalPerceptionTester:
    """Probes how LLMs represent past, present, and future.

    Analyzes text for temporal markers and classifies the dominant
    tense orientation of LLM responses.
    """

    def __init__(self) -> None:
        self._past_re = [re.compile(p, re.IGNORECASE) for p in _PAST_PATTERNS]
        self._present_re = [re.compile(p, re.IGNORECASE) for p in _PRESENT_PATTERNS]
        self._future_re = [re.compile(p, re.IGNORECASE) for p in _FUTURE_PATTERNS]

    def detect_tense(self, text: str) -> tuple[Tense, float]:
        """Detect the dominant tense in a text."""
        past = sum(len(p.findall(text)) for p in self._past_re)
        present = sum(len(p.findall(text)) for p in self._present_re)
        future = sum(len(p.findall(text)) for p in self._future_re)

        total = past + present + future
        if total == 0:
            return Tense.AMBIGUOUS, 0.33

        scores = {Tense.PAST: past, Tense.PRESENT: present, Tense.FUTURE: future}
        best = max(scores, key=lambda k: scores[k])
        confidence = scores[best] / total
        return best, round(confidence, 3)

    def test(self, probe: TemporalProbe, response_text: str) -> TemporalResponse:
        """Test an LLM response against a temporal probe."""
        tense, confidence = self.detect_tense(response_text)
        is_correct = tense == probe.expected_tense if probe.expected_tense else None

        return TemporalResponse(
            probe=probe,
            response_text=response_text,
            detected_tense=tense,
            is_correct=is_correct,
            confidence=confidence,
        )

    def generate_probes(self) -> list[TemporalProbe]:
        """Generate a standard set of temporal perception probes."""
        return [
            TemporalProbe(
                question="Describe what happened during the Renaissance.",
                expected_tense=Tense.PAST,
            ),
            TemporalProbe(
                question="What is the current state of AI research?",
                expected_tense=Tense.PRESENT,
            ),
            TemporalProbe(
                question="What will space travel look like in 2100?",
                expected_tense=Tense.FUTURE,
            ),
            TemporalProbe(
                question="Describe the concept of entropy.",
                expected_tense=Tense.PRESENT,
            ),
        ]
