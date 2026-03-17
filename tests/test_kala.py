"""Tests for Kala."""

import math

from kala.analyzer.bias import TemporalBiasDetector
from kala.analyzer.consistency import TemporalConsistencyChecker
from kala.models import Tense, TemporalBiasType, TemporalProbe
from kala.temporal.duration import DurationEstimator
from kala.temporal.ordering import EventOrderingTester
from kala.temporal.perception import TemporalPerceptionTester


class TestTemporalPerception:
    def setup_method(self):
        self.tester = TemporalPerceptionTester()

    def test_past_tense_detection(self):
        tense, conf = self.tester.detect_tense(
            "The Romans built the Colosseum. They conquered many lands and established an empire."
        )
        assert tense == Tense.PAST

    def test_present_tense_detection(self):
        tense, conf = self.tester.detect_tense(
            "The sun is shining. Birds are singing. Life goes on right now."
        )
        assert tense == Tense.PRESENT

    def test_future_tense_detection(self):
        tense, conf = self.tester.detect_tense(
            "Humans will colonize Mars. Space travel will become common. We shall explore."
        )
        assert tense == Tense.FUTURE

    def test_probe_evaluation(self):
        probe = TemporalProbe(
            question="What happened in ancient Egypt?",
            expected_tense=Tense.PAST,
        )
        result = self.tester.test(probe, "The Egyptians built pyramids and worshipped gods.")
        assert result.detected_tense == Tense.PAST
        assert result.is_correct is True

    def test_generate_probes(self):
        probes = self.tester.generate_probes()
        assert len(probes) >= 3


class TestEventOrdering:
    def setup_method(self):
        self.tester = EventOrderingTester()

    def test_correct_ordering(self):
        expected = ["A", "B", "C"]
        result = self.tester.evaluate_ordering(expected, ["A", "B", "C"])
        assert result.is_correct is True
        assert result.kendall_tau == 1.0

    def test_reversed_ordering(self):
        result = self.tester.evaluate_ordering(["A", "B", "C"], ["C", "B", "A"])
        assert result.is_correct is False
        assert result.kendall_tau == -1.0

    def test_extract_ordering(self):
        text = "First came Rome, then the Medieval period, finally the Renaissance."
        events = ["Renaissance", "Medieval period", "Rome"]
        ordered = self.tester.extract_ordering_from_text(text, events)
        assert ordered[0] == "Rome"

    def test_generate_cases(self):
        cases = self.tester.generate_test_cases()
        assert len(cases) >= 2


class TestDurationEstimator:
    def setup_method(self):
        self.estimator = DurationEstimator()

    def test_parse_duration(self):
        result = self.estimator.parse_duration_from_text("It takes about 3 hours and 30 minutes.")
        assert result is not None
        assert abs(result - 12600.0) < 1

    def test_parse_no_duration(self):
        assert self.estimator.parse_duration_from_text("No time info here.") is None

    def test_evaluate(self):
        result = self.estimator.evaluate("test event", 3600.0, 3500.0)
        assert result.absolute_error == 100.0
        assert result.log_ratio_error < 0.1

    def test_generate_probes(self):
        probes = self.estimator.generate_probes()
        assert len(probes) >= 3


class TestConsistencyChecker:
    def setup_method(self):
        self.checker = TemporalConsistencyChecker()

    def test_consistent_text(self):
        result = self.checker.check("Rome fell in 476 AD. The Renaissance began in the 14th century.")
        assert result.is_consistent is True

    def test_year_contradiction(self):
        result = self.checker.check(
            "The discovery happened in 1905. Later the discovery happened in 1750."
        )
        assert len(result.contradictions) > 0

    def test_tense_mix(self):
        result = self.checker.check("The event was completed and will be finished tomorrow.")
        assert not result.is_consistent


class TestBiasDetector:
    def setup_method(self):
        self.detector = TemporalBiasDetector()

    def test_recency_bias(self):
        text = (
            "Key events: 2010 launch, 2015 expansion, 2018 IPO, "
            "2020 pandemic, 2022 recovery, 2023 growth."
        )
        results = self.detector.detect(text)
        types = [r.bias_type for r in results]
        assert TemporalBiasType.RECENCY in types

    def test_no_bias(self):
        results = self.detector.detect("Hello world, this is a simple text.")
        assert results[0].bias_type == TemporalBiasType.NONE

    def test_telescoping(self):
        text = "Recently, ancient civilizations in medieval times built monuments."
        results = self.detector.detect(text)
        types = [r.bias_type for r in results]
        assert TemporalBiasType.TELESCOPING in types
