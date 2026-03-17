"""Temporal consistency checking for LLM outputs."""

from __future__ import annotations

import re

from kala.models import ConsistencyResult


_TEMPORAL_CONTRADICTION_PATTERNS = [
    (r"before .+ after .+ before", "Circular temporal reference"),
    (r"in (\d{4}).+in (\d{4})", None),  # Check year ordering
]


class TemporalConsistencyChecker:
    """Finds contradictions in temporal references within text.

    Detects when an LLM makes inconsistent claims about when events
    occurred relative to each other.
    """

    def check(self, text: str) -> ConsistencyResult:
        """Check a text for temporal contradictions."""
        contradictions: list[str] = []

        # Check for year-based contradictions
        contradictions.extend(self._check_year_ordering(text))

        # Check for before/after contradictions
        contradictions.extend(self._check_before_after(text))

        # Check for tense inconsistencies
        contradictions.extend(self._check_tense_consistency(text))

        return ConsistencyResult(
            text=text,
            contradictions=contradictions,
            is_consistent=len(contradictions) == 0,
        )

    def _check_year_ordering(self, text: str) -> list[str]:
        """Check if years mentioned are used consistently."""
        contradictions = []
        sentences = re.split(r"[.!?]", text)

        year_claims: dict[str, list[int]] = {}
        for sent in sentences:
            # Find patterns like "X happened in YYYY"
            matches = re.findall(r"(\w[\w\s]{2,30}?)\s+(?:in|around|circa)\s+(\d{4})", sent)
            for event, year_str in matches:
                event_key = event.strip().lower()
                year = int(year_str)
                if event_key in year_claims:
                    for prev_year in year_claims[event_key]:
                        if abs(prev_year - year) > 5:
                            contradictions.append(
                                f"Inconsistent dating: '{event_key}' dated to both "
                                f"{prev_year} and {year}"
                            )
                year_claims.setdefault(event_key, []).append(year)

        return contradictions

    def _check_before_after(self, text: str) -> list[str]:
        """Check for contradictory before/after relationships."""
        contradictions = []
        before_pattern = re.compile(r"(\w[\w\s]+?)\s+(?:before|prior to)\s+(\w[\w\s]+?)(?:[.,;]|$)")
        after_pattern = re.compile(r"(\w[\w\s]+?)\s+(?:after|following)\s+(\w[\w\s]+?)(?:[.,;]|$)")

        befores = set()
        for m in before_pattern.finditer(text.lower()):
            befores.add((m.group(1).strip(), m.group(2).strip()))

        for m in after_pattern.finditer(text.lower()):
            a, b = m.group(1).strip(), m.group(2).strip()
            # "A after B" means B before A; contradiction if also "B after A" or "A before B" reversed
            if (b, a) in befores:
                contradictions.append(
                    f"Contradictory ordering: '{a}' is stated as both before and after '{b}'"
                )

        return contradictions

    def _check_tense_consistency(self, text: str) -> list[str]:
        """Check for jarring tense switches within the same sentence."""
        contradictions = []
        sentences = re.split(r"[.!?]", text)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            has_past = bool(re.search(r"\b(was|were|had been|did)\b", sent))
            has_future = bool(re.search(r"\b(will be|shall|going to)\b", sent))
            if has_past and has_future:
                contradictions.append(
                    f"Mixed tenses in one sentence: '{sent[:60]}...'"
                )

        return contradictions
