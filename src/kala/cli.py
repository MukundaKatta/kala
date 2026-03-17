"""CLI interface for Kala."""

from __future__ import annotations

import click
from rich.console import Console

from kala.analyzer.bias import TemporalBiasDetector
from kala.analyzer.consistency import TemporalConsistencyChecker
from kala.models import TemporalReport
from kala.report import render_report
from kala.temporal.perception import TemporalPerceptionTester

console = Console()


@click.group()
def cli() -> None:
    """Kala: Explore how AI systems experience and reason about time."""


@cli.command()
@click.argument("text")
def check_consistency(text: str) -> None:
    """Check temporal consistency of a text."""
    checker = TemporalConsistencyChecker()
    result = checker.check(text)
    report = TemporalReport(consistency_results=[result])
    render_report(report, console)


@cli.command()
@click.argument("text")
def detect_bias(text: str) -> None:
    """Detect temporal biases in a text."""
    detector = TemporalBiasDetector()
    results = detector.detect(text)
    report = TemporalReport(bias_results=results)
    render_report(report, console)


@cli.command()
@click.argument("text")
def detect_tense(text: str) -> None:
    """Detect dominant tense in a text."""
    tester = TemporalPerceptionTester()
    tense, conf = tester.detect_tense(text)
    console.print(f"Detected tense: [bold]{tense.value}[/bold] (confidence: {conf:.1%})")


if __name__ == "__main__":
    cli()
