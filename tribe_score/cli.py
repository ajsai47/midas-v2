"""CLI for tribe-score — neural engagement scoring."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import click


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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.pass_context
def score(ctx, content, file_path, text_input, as_json):
    """Score content for neural engagement.

    CONTENT can be inline text or a file path. Use --file or --text to be explicit.
    """
    from .scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer(
        model_id=ctx.obj["model"],
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
            "raw_score": round(result.raw_score, 4),
            "group_scores": {k: round(v, 4) for k, v in result.group_scores.items()},
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

    Shows per-group breakdown with neuroscience context for the 10 empirical
    regions that correlate with LinkedIn engagement (p<0.01).
    """
    from .regions import EMPIRICAL_REGIONS, REGION_GROUPS
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

    group_info = {
        "Reward": (
            "Posterior OFC + OFC",
            "Value judgment and reward anticipation — drives saves and follows",
        ),
        "Memory": (
            "Hippocampus",
            "Memory encoding — memorable content gets shared",
        ),
        "Social": (
            "Temporal Pole (dorsal + ventral)",
            "Social/emotional processing — drives virality through social cognition",
        ),
        "Auditory (suppressed)": (
            "TA2, A4, PBelt, MBelt, A5",
            "Auditory cortex SUPPRESSION in viral content — visual/conceptual > auditory",
        ),
    }

    for group_name, group_score in sorted(result.group_scores.items(), key=lambda x: -x[1]):
        info = group_info.get(group_name)
        if info:
            regions_str, desc = info
        else:
            regions_str, desc = "—", "Brain focus signal"

        bar = "+" * max(0, int(group_score * 2)) if group_score > 0 else "-" * max(0, int(-group_score * 2))
        click.echo(f"  {group_name}")
        click.echo(f"    Score:   {group_score:+.2f}  {bar}")
        click.echo(f"    Regions: {regions_str}")
        click.echo(f"    Why:     {desc}")

        # Show individual region z-scores for this group
        if group_name in REGION_GROUPS:
            region_details = []
            for r in REGION_GROUPS[group_name]:
                if r in result.region_zscores:
                    rho = EMPIRICAL_REGIONS[r]
                    z = result.region_zscores[r]
                    region_details.append(f"{r}: z={z:+.2f} (rho={rho:+.4f})")
            if region_details:
                click.echo(f"    Detail:  {', '.join(region_details)}")
        click.echo()

    click.echo(f"  Brain focus: {result.variability_zscore:+.2f}\u03c3 (lower variability = more focused)")
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

        # Group scores table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Region Group", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Bar", min_width=20)

        for group, val in sorted(result.group_scores.items(), key=lambda x: -x[1]):
            if val >= 0:
                bar_len = min(20, int(val * 2))
                color = "green" if val >= 1.0 else "yellow"
                bar = f"[{color}]{'+'* bar_len}[/{color}]"
            else:
                bar_len = min(20, int(-val * 2))
                bar = f"[red]{'-' * bar_len}[/red]"
            table.add_row(group, f"{val:+.2f}", bar)

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
