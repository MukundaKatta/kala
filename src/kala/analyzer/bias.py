"""Temporal bias detection for LLM outputs."""

from __future__ import annotations

import re
from collections import Counter

from kala.models import BiasResult, TemporalBiasType


class TemporalBiasDetector:
    """Detects recency bias, anchoring bias, and telescoping in LLM outputs.

    - Recency bias: disproportionate focus on recent events
    - Anchoring bias: over-reliance on first temporal reference
    - Telescoping: perceiving distant events as closer than they are
    """

    def detect(self, text: str) -> list[BiasResult]:
        """Detect all temporal biases in a text."""
        biases: list[BiasResult] = []

        recency = self._detect_recency_bias(text)
        if recency:
            biases.append(recency)

        anchoring = self._detect_anchoring_bias(text)
        if anchoring:
            biases.append(anchoring)

        telescoping = self._detect_telescoping(text)
        if telescoping:
            biases.append(telescoping)

        if not biases:
            biases.append(BiasResult(
                text=text, bias_type=TemporalBiasType.NONE,
                evidence="No temporal biases detected.", severity=0.0,
            ))

        return biases

    def _detect_recency_bias(self, text: str) -> BiasResult | None:
        """Detect if text disproportionately references recent times."""
        years = [int(y) for y in re.findall(r"\b(1\d{3}|20\d{2})\b", text)]
        if len(years) < 3:
            return None

        recent = sum(1 for y in years if y >= 2000)
        ratio = recent / len(years)
        if ratio > 0.75:
            return BiasResult(
                text=text,
                bias_type=TemporalBiasType.RECENCY,
                evidence=f"{recent}/{len(years)} year references are post-2000",
                severity=min(ratio, 1.0),
            )
        return None

    def _detect_anchoring_bias(self, text: str) -> BiasResult | None:
        """Detect if text anchors heavily on the first temporal reference."""
        years = re.findall(r"\b(1\d{3}|20\d{2})\b", text)
        if len(years) < 3:
            return None

        anchor = int(years[0])
        diffs = [abs(int(y) - anchor) for y in years[1:]]
        avg_diff = sum(diffs) / len(diffs)

        if avg_diff < 20:
            return BiasResult(
                text=text,
                bias_type=TemporalBiasType.ANCHORING,
                evidence=f"Anchored around {anchor}, avg deviation: {avg_diff:.0f} years",
                severity=min(1.0 - avg_diff / 100, 1.0),
            )
        return None

    def _detect_telescoping(self, text: str) -> BiasResult | None:
        """Detect telescoping: recent language for distant events."""
        distant_refs = re.findall(
            r"\b(recently|just|lately|the other day)\b.*\b(ancient|centuries|medieval|1[0-7]\d{2})\b",
            text, re.IGNORECASE,
        )
        if distant_refs:
            return BiasResult(
                text=text,
                bias_type=TemporalBiasType.TELESCOPING,
                evidence=f"Proximity language used for distant events: {distant_refs[0]}",
                severity=0.7,
            )
        return None
