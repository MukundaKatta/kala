"""Event ordering tests for LLM temporal reasoning."""

from __future__ import annotations

import re

from kala.models import OrderingResult


class EventOrderingTester:
    """Tests temporal reasoning accuracy by event ordering tasks.

    Presents events and evaluates whether the LLM can correctly
    order them chronologically.
    """

    def evaluate_ordering(
        self, expected: list[str], predicted: list[str]
    ) -> OrderingResult:
        """Evaluate a predicted event ordering against the expected ordering."""
        tau = self._kendall_tau(expected, predicted)
        is_correct = expected == predicted

        return OrderingResult(
            events=expected,
            expected_order=expected,
            predicted_order=predicted,
            kendall_tau=tau,
            is_correct=is_correct,
        )

    def extract_ordering_from_text(
        self, text: str, events: list[str]
    ) -> list[str]:
        """Extract event ordering from free-text LLM response."""
        positions: dict[str, int] = {}
        text_lower = text.lower()
        for event in events:
            pos = text_lower.find(event.lower())
            if pos >= 0:
                positions[event] = pos
            else:
                # Try partial match
                words = event.lower().split()
                for w in words:
                    idx = text_lower.find(w)
                    if idx >= 0:
                        positions[event] = idx
                        break
                else:
                    positions[event] = len(text)  # not found, put last

        return sorted(events, key=lambda e: positions.get(e, len(text)))

    def generate_test_cases(self) -> list[tuple[str, list[str]]]:
        """Generate standard event ordering test cases."""
        return [
            (
                "Order these historical events chronologically.",
                ["Invention of printing press", "Moon landing", "World Wide Web", "iPhone release"],
            ),
            (
                "Order these scientific discoveries.",
                ["Gravity (Newton)", "Evolution (Darwin)", "Relativity (Einstein)", "DNA structure"],
            ),
            (
                "Order these from earliest to latest.",
                ["Ancient Rome", "Medieval period", "Renaissance", "Industrial Revolution"],
            ),
        ]

    @staticmethod
    def _kendall_tau(list_a: list[str], list_b: list[str]) -> float:
        """Compute Kendall tau correlation between two orderings."""
        if not list_a or not list_b:
            return 0.0
        n = len(list_a)
        if n <= 1:
            return 1.0

        rank_b = {item: i for i, item in enumerate(list_b)}
        b_ranks = []
        for item in list_a:
            if item in rank_b:
                b_ranks.append(rank_b[item])
            else:
                b_ranks.append(n)

        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                if b_ranks[i] < b_ranks[j]:
                    concordant += 1
                elif b_ranks[i] > b_ranks[j]:
                    discordant += 1

        pairs = n * (n - 1) / 2
        return (concordant - discordant) / pairs if pairs > 0 else 0.0
