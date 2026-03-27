"""CLI for tribe-score — neural engagement scoring."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import click


def _parse_weights(ctx, param, value) -> dict[str, float] | None:
    """Parse --weights emotional=0.4,reward=0.3 into a dict."""
    if not value:
        return None
    # Map short names to full dimension names
    aliases = {
        "emotional": "emotional_arousal",
        "reward": "reward_motivation",
        "attention": "attention_capture",
        "social": "social_cognition",
        "memory": "memory_encoding",
    }
    weights = {}
    for pair in value.split(","):
        key, val = pair.strip().split("=")
        key = aliases.get(key.strip(), key.strip())
        weights[key] = float(val)
    return weights


@click.group()
@click.option("--model", default="facebook/tribev2", help="Model ID or local path.")
@click.option("--device", default="auto", help="Device: auto, cpu, cuda.")
@click.option("--cache", default=None, help="Cache folder for features.")
@click.pass_context
def cli(ctx, model, device, cache):
    """tribe-score: Neural content engagement scoring powered by brain predictions."""
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["device"] = device
    ctx.obj["cache"] = cache


@cli.command()
@click.argument("content", required=False)
@click.option("--file", "-f", "file_path", help="Path to content file.")
@click.option("--text", "-t", "text_input", help="Inline text to score.")
@click.option(
    "--weights", "-w", callback=_parse_weights, default=None,
    help="Override weights: emotional=0.4,reward=0.3,..."
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.pass_context
def score(ctx, content, file_path, text_input, weights, as_json):
    """Score content for neural engagement.

    CONTENT can be inline text or a file path. Use --file or --text to be explicit.
    """
    from .scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer(
        model_id=ctx.obj["model"],
        weights=weights,
        device=ctx.obj["device"],
        cache_folder=ctx.obj["cache"],
    )

    # Determine what to score
    if text_input:
        result = scorer.score_text(text_input)
        label = "inline text"
    elif file_path:
        result = scorer.score(file_path)
        label = file_path
    elif content:
        path = Path(content)
        if path.is_file():
            result = scorer.score(str(path))
            label = content
        else:
            # Treat as inline text
            result = scorer.score_text(content)
            label = "inline text"
    else:
        click.echo("Error: provide content as argument, --file, or --text.", err=True)
        sys.exit(1)

    if as_json:
        import json

        click.echo(json.dumps({
            "label": label,
            "nes": round(result.nes, 1),
            "tier": result.tier,
            "dimensions": {k: round(v, 1) for k, v in result.dimensions.items()},
            "top_regions": result.top_regions[:10],
        }, indent=2))
    else:
        _print_score(label, result)


@cli.command()
@click.argument("files", nargs=-1)
@click.option("--text", "-t", "texts", multiple=True, help="Inline text variants.")
@click.pass_context
def compare(ctx, files, texts):
    """Compare multiple content variants for neural engagement.

    Pass file paths as arguments, or use --text for inline text variants.
    """
    from .compare import compare_files, compare_texts
    from .scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer(
        model_id=ctx.obj["model"],
        device=ctx.obj["device"],
        cache_folder=ctx.obj["cache"],
    )

    if texts:
        output = compare_texts(list(texts), scorer=scorer)
    elif files:
        output = compare_files(list(files), scorer=scorer)
    else:
        click.echo("Error: provide files or --text variants.", err=True)
        sys.exit(1)

    click.echo(output)


@cli.command()
@click.argument("content", required=False)
@click.option("--file", "-f", "file_path", help="Path to content file.")
@click.option("--text", "-t", "text_input", help="Inline text to explain.")
@click.pass_context
def explain(ctx, content, file_path, text_input):
    """Explain which brain regions activated and why.

    Shows per-dimension breakdown with neuroscience context.
    """
    from .regions import ENGAGEMENT_REGIONS
    from .scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer(
        model_id=ctx.obj["model"],
        device=ctx.obj["device"],
        cache_folder=ctx.obj["cache"],
    )

    if text_input:
        result = scorer.score_text(text_input)
    elif file_path:
        result = scorer.score(file_path)
    elif content:
        path = Path(content)
        if path.is_file():
            result = scorer.score(str(path))
        else:
            result = scorer.score_text(content)
    else:
        click.echo("Error: provide content.", err=True)
        sys.exit(1)

    click.echo(f"\nNeural Engagement Score: {result.nes:.1f} — {result.tier}\n")

    explanations = {
        "emotional_arousal": (
            "Emotional Arousal",
            "Anterior Insula + ACC + Temporal Pole",
            "Drives sharing and commenting behavior",
        ),
        "reward_motivation": (
            "Reward / Motivation",
            "OFC + vmPFC",
            "Creates 'I want more' response — saves and follows",
        ),
        "attention_capture": (
            "Attention Capture",
            "FEF + DLPFC + Visual Cortex + IPS",
            "Must grab attention for any engagement",
        ),
        "social_cognition": (
            "Social Cognition",
            "STS + TPJ + Fusiform",
            "Social content processing — drives virality",
        ),
        "memory_encoding": (
            "Memory Encoding",
            "Parahippocampal + Retrosplenial",
            "Memorable content gets shared",
        ),
    }

    for dim, score_val in sorted(result.dimensions.items(), key=lambda x: -x[1]):
        title, regions, desc = explanations.get(dim, (dim, "—", "—"))
        bar = "#" * int(score_val / 5)
        click.echo(f"  {title}")
        click.echo(f"    Score:   {score_val:.1f}  {bar}")
        click.echo(f"    Regions: {regions}")
        click.echo(f"    Why:     {desc}")
        click.echo(f"    HCP ROIs: {', '.join(ENGAGEMENT_REGIONS.get(dim, []))}")
        click.echo()

    click.echo(f"  Top 10 activated regions: {', '.join(result.top_regions[:10])}")


@cli.command()
@click.argument("content", required=False)
@click.option("--file", "-f", "file_path", help="Path to content file.")
@click.option("--text", "-t", "text_input", help="Inline text.")
@click.option("--output", "-o", default="brain_heatmap.png", help="Output image path.")
@click.option(
    "--views", default="left,right",
    help="Comma-separated views: left, right, dorsal, ventral, etc.",
)
@click.pass_context
def heatmap(ctx, content, file_path, text_input, output, views):
    """Generate a brain activation heatmap.

    Requires the plotting optional dependencies (nilearn, matplotlib, etc).
    """
    from .scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer(
        model_id=ctx.obj["model"],
        device=ctx.obj["device"],
        cache_folder=ctx.obj["cache"],
    )

    if text_input:
        result = scorer.score_text(text_input)
    elif file_path:
        result = scorer.score(file_path)
    elif content:
        path = Path(content)
        if path.is_file():
            result = scorer.score(str(path))
        else:
            result = scorer.score_text(content)
    else:
        click.echo("Error: provide content.", err=True)
        sys.exit(1)

    try:
        import matplotlib.pyplot as plt
        from tribev2.plotting.cortical import PlotBrainNilearn
    except ImportError:
        click.echo(
            "Error: plotting dependencies required. "
            'Install with: pip install -e ".[plotting]"',
            err=True,
        )
        sys.exit(1)

    view_list = [v.strip() for v in views.split(",")]

    plotter = PlotBrainNilearn()
    fig, axes = plotter.get_fig_axes(view_list)
    plotter.plot_surf(
        result.raw_activation,
        views=view_list,
        axes=axes,
        cmap="hot",
        colorbar=True,
        colorbar_title="Activation",
    )
    fig.suptitle(f"NES: {result.nes:.1f} — {result.tier}", fontsize=10)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Saved brain heatmap to {output}")


def _print_score(label: str, result):
    """Pretty-print a single score result."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Dimension table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Dimension", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Bar", min_width=20)

        for dim, val in sorted(result.dimensions.items(), key=lambda x: -x[1]):
            bar_len = int(val / 5)
            color = "green" if val >= 60 else "yellow" if val >= 40 else "red"
            bar = f"[{color}]{'#' * bar_len}[/{color}]"
            table.add_row(dim.replace("_", " ").title(), f"{val:.1f}", bar)

        console.print(Panel(
            f"[bold green]{result.nes:.1f}[/bold green] — [yellow]{result.tier}[/yellow]\n"
            f"{result.tier_description}\n\n"
            f"Top regions: {', '.join(result.top_regions[:5])}",
            title=f"Neural Engagement Score — {label}",
            border_style="blue",
        ))
        console.print(table)

    except ImportError:
        # Fallback: plain text
        click.echo(str(result))


def main():
    cli()


if __name__ == "__main__":
    main()
