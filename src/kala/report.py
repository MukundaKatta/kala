"""Report generation for Kala temporal analysis."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kala.models import TemporalReport


def render_report(report: TemporalReport, console: Console | None = None) -> None:
    """Render a rich report of temporal analysis results."""
    console = console or Console()
    console.print(Panel("[bold cyan]Kala Temporal Analysis Report[/bold cyan]", expand=False))

    # Perception results
    if report.perception_results:
        t = Table(title="Temporal Perception")
        t.add_column("Probe", max_width=40)
        t.add_column("Detected Tense", style="magenta")
        t.add_column("Correct", style="green")
        t.add_column("Confidence", justify="right")
        for r in report.perception_results:
            t.add_row(
                r.probe.question[:40],
                r.detected_tense.value if r.detected_tense else "?",
                str(r.is_correct) if r.is_correct is not None else "N/A",
                f"{r.confidence:.1%}",
            )
        console.print(t)

    # Ordering results
    if report.ordering_results:
        t = Table(title="Event Ordering")
        t.add_column("Events", max_width=50)
        t.add_column("Kendall Tau", justify="right", style="cyan")
        t.add_column("Correct", style="green")
        for r in report.ordering_results:
            t.add_row(
                " > ".join(r.predicted_order[:3]) + "...",
                f"{r.kendall_tau:.3f}",
                str(r.is_correct),
            )
        console.print(t)

    # Consistency results
    if report.consistency_results:
        t = Table(title="Temporal Consistency")
        t.add_column("Text (truncated)", max_width=40)
        t.add_column("Consistent", style="green")
        t.add_column("Contradictions", style="red")
        for r in report.consistency_results:
            t.add_row(
                r.text[:40] + "...",
                str(r.is_consistent),
                str(len(r.contradictions)),
            )
        console.print(t)

    # Bias results
    if report.bias_results:
        t = Table(title="Temporal Biases")
        t.add_column("Bias Type", style="yellow")
        t.add_column("Severity", justify="right")
        t.add_column("Evidence", max_width=50)
        for r in report.bias_results:
            t.add_row(r.bias_type.value, f"{r.severity:.1%}", r.evidence[:50])
        console.print(t)
