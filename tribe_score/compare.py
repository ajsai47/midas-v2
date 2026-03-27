"""Multi-variant comparison with rich terminal output."""

from __future__ import annotations

from pathlib import Path

from .scorer import NeuralEngagementScorer, NeuralScoreResult


def format_comparison_table(
    results: list[tuple[str, NeuralScoreResult]],
) -> str:
    """Format comparison results as an ASCII table.

    Parameters
    ----------
    results : list of (label, NeuralScoreResult)
        Labeled results to compare.

    Returns
    -------
    str
        Formatted comparison table.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        return _rich_table(results)
    except ImportError:
        return _ascii_table(results)


def _rich_table(results: list[tuple[str, NeuralScoreResult]]) -> str:
    """Render comparison using rich."""
    from io import StringIO

    from rich.console import Console
    from rich.table import Table

    table = Table(title="Neural Engagement Comparison", show_lines=True)
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Content", style="cyan")
    table.add_column("NES", justify="right", style="bold green")
    table.add_column("Tier", style="yellow")
    table.add_column("Emotional", justify="right")
    table.add_column("Reward", justify="right")
    table.add_column("Attention", justify="right")
    table.add_column("Social", justify="right")
    table.add_column("Memory", justify="right")

    for i, (label, r) in enumerate(results, 1):
        table.add_row(
            str(i),
            label,
            f"{r.nes:.1f}",
            r.tier,
            f"{r.dimensions.get('emotional_arousal', 0):.1f}",
            f"{r.dimensions.get('reward_motivation', 0):.1f}",
            f"{r.dimensions.get('attention_capture', 0):.1f}",
            f"{r.dimensions.get('social_cognition', 0):.1f}",
            f"{r.dimensions.get('memory_encoding', 0):.1f}",
        )

    buf = StringIO()
    console = Console(file=buf, force_terminal=True)
    console.print(table)
    return buf.getvalue()


def _ascii_table(results: list[tuple[str, NeuralScoreResult]]) -> str:
    """Fallback ASCII table when rich is not installed."""
    lines = [
        f"{'Rank':<5} {'Content':<30} {'NES':>6} {'Tier':<20} "
        f"{'Emot':>6} {'Reward':>6} {'Attn':>6} {'Social':>6} {'Memory':>6}"
    ]
    lines.append("-" * len(lines[0]))
    for i, (label, r) in enumerate(results, 1):
        lines.append(
            f"{i:<5} {label[:30]:<30} {r.nes:>6.1f} {r.tier:<20} "
            f"{r.dimensions.get('emotional_arousal', 0):>6.1f} "
            f"{r.dimensions.get('reward_motivation', 0):>6.1f} "
            f"{r.dimensions.get('attention_capture', 0):>6.1f} "
            f"{r.dimensions.get('social_cognition', 0):>6.1f} "
            f"{r.dimensions.get('memory_encoding', 0):>6.1f}"
        )
    return "\n".join(lines)


def compare_texts(
    texts: list[str],
    labels: list[str] | None = None,
    scorer: NeuralEngagementScorer | None = None,
    **scorer_kwargs,
) -> str:
    """Compare multiple text variants and return a formatted table.

    Parameters
    ----------
    texts : list of str
        Text content variants to compare.
    labels : list of str, optional
        Labels for each variant. Defaults to "Variant 1", "Variant 2", etc.
    scorer : NeuralEngagementScorer, optional
        Pre-initialized scorer. Created with scorer_kwargs if not provided.
    """
    if scorer is None:
        scorer = NeuralEngagementScorer(**scorer_kwargs)

    if labels is None:
        labels = [f"Variant {i}" for i in range(1, len(texts) + 1)]

    results = [(label, scorer.score_text(text)) for label, text in zip(labels, texts)]
    results.sort(key=lambda x: x[1].nes, reverse=True)
    return format_comparison_table(results)


def compare_files(
    paths: list[str],
    scorer: NeuralEngagementScorer | None = None,
    **scorer_kwargs,
) -> str:
    """Compare multiple content files and return a formatted table."""
    if scorer is None:
        scorer = NeuralEngagementScorer(**scorer_kwargs)

    results = [(Path(p).name, scorer.score(p)) for p in paths]
    results.sort(key=lambda x: x[1].nes, reverse=True)
    return format_comparison_table(results)
