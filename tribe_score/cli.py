"""CLI for Midas v2 — predict content virality before you post."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import click


def _brain_available() -> bool:
    """Check if brain model dependencies are installed."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_text(content, file_path, text_input):
    """Resolve text from the various input options. Returns None if no input."""
    if text_input:
        return text_input
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    if content:
        path = Path(content)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return content
    return None


def _require_brain(command_name: str):
    """Exit with a clean message if brain deps are missing."""
    if not _brain_available():
        click.echo(
            f"'{command_name}' requires the brain model.\n"
            "Install with: pip install -e \".[brain]\"",
            err=True,
        )
        sys.exit(1)


class MidasGroup(click.Group):
    """Custom group that routes bare text to evaluate."""

    def parse_args(self, ctx, args):
        # No args at all → show welcome screen
        if not args:
            return super().parse_args(ctx, args)
        # If first arg isn't a known command and doesn't start with '-',
        # treat it as text to evaluate (structural + agents, no brain)
        if args[0] not in self.commands and not args[0].startswith("-"):
            args = ["evaluate", "--no-brain"] + args
        return super().parse_args(ctx, args)

    def invoke(self, ctx):
        # If no subcommand was invoked, show welcome
        if ctx.invoked_subcommand is None:
            _show_welcome()
            return
        return super().invoke(ctx)


def _show_welcome():
    """Show the welcome screen when midas is called with no args."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel(
            '[bold]midas "your post text"[/bold]          Score your post (fast)\n'
            '[bold]midas optimize "your post"[/bold]      Auto-improve until ship-ready\n'
            '[bold]midas setup[/bold]                     Guided install\n'
            '[bold]midas --help[/bold]                    All commands',
            title="Midas v2",
            subtitle="Predict content virality before you post.",
            border_style="blue",
        ))
    except ImportError:
        click.echo("Midas v2 — Predict content virality before you post.\n")
        click.echo('  midas "your post text"          Score your post (fast)')
        click.echo('  midas optimize "your post"      Auto-improve until ship-ready')
        click.echo("  midas setup                     Guided install")
        click.echo("  midas --help                    All commands")


@click.group(cls=MidasGroup, invoke_without_command=True)
@click.option("--model", default="facebook/tribev2", help="Model ID or local path.")
@click.option("--device", default="auto", help="Device: auto, cpu, cuda.")
@click.option("--cache", default=None, help="Cache folder for features.")
@click.pass_context
def cli(ctx, model, device, cache):
    """Predict content virality before you post."""
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["device"] = device
    ctx.obj["cache"] = cache


@cli.command()
def setup():
    """Guided install — choose your experience."""
    click.echo("\nWelcome to Midas v2 — predict content virality before you post.\n")
    click.echo("What experience do you want?\n")
    click.echo("  1. Quick scoring (structural patterns)          — pip install -e .")
    click.echo('  2. Score + auto-improve with AI agents          — pip install -e ".[agents]"')
    click.echo('  3. Full brain predictions + SVG renders + agents — pip install -e ".[all]"\n')

    choice = click.prompt("Your choice", type=click.Choice(["1", "2", "3"]), default="1")

    install_map = {
        "1": [sys.executable, "-m", "pip", "install", "-e", "."],
        "2": [sys.executable, "-m", "pip", "install", "-e", ".[agents]"],
        "3": [sys.executable, "-m", "pip", "install", "-e", ".[all]"],
    }

    click.echo(f"\nInstalling...\n")
    result = subprocess.run(install_map[choice], cwd=str(Path(__file__).resolve().parents[1]))

    if result.returncode != 0:
        click.echo("\nInstall failed. Check the output above.", err=True)
        sys.exit(1)

    click.echo("\nInstalled!")

    if choice in ("2", "3"):
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            click.echo(
                "\nFor AI agent simulation, set your API key:\n"
                "  export ANTHROPIC_API_KEY=your-key-here"
            )

    click.echo('\nTry it: midas "I just got fired from Google. Best thing that ever happened."')


@cli.command()
@click.argument("content", required=False)
@click.option("--file", "-f", "file_path", help="Path to content file.")
@click.option("--text", "-t", "text_input", help="Inline text to score.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.pass_context
def score(ctx, content, file_path, text_input, as_json):
    """Score content for neural engagement (brain model only).

    Requires brain dependencies: pip install -e ".[brain]"
    """
    _require_brain("score")

    from .scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer(
        model_id=ctx.obj["model"],
        device=ctx.obj["device"],
        cache_folder=ctx.obj["cache"],
    )

    text = _resolve_text(content, file_path, text_input)
    if text is None:
        click.echo("Error: provide content as argument, --file, or --text.", err=True)
        sys.exit(1)

    path = Path(content) if content and Path(content).is_file() else None
    if path:
        result = scorer.score(str(path))
        label = content
    else:
        result = scorer.score_text(text)
        label = "inline text"

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

    Requires brain dependencies: pip install -e ".[brain]"
    """
    _require_brain("compare")

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

    Requires brain dependencies: pip install -e ".[brain]"
    """
    _require_brain("explain")

    from .regions import EMPIRICAL_REGIONS, REGION_GROUPS
    from .scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer(
        model_id=ctx.obj["model"],
        device=ctx.obj["device"],
        cache_folder=ctx.obj["cache"],
    )

    text = _resolve_text(content, file_path, text_input)
    if text is None:
        click.echo("Error: provide content.", err=True)
        sys.exit(1)

    path = Path(content) if content and Path(content).is_file() else None
    if path:
        result = scorer.score(str(path))
    else:
        result = scorer.score_text(text)

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
    """Generate a brain activation heatmap image.

    Requires brain + plotting dependencies: pip install -e ".[all]"
    """
    _require_brain("heatmap")

    from .scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer(
        model_id=ctx.obj["model"],
        device=ctx.obj["device"],
        cache_folder=ctx.obj["cache"],
    )

    text = _resolve_text(content, file_path, text_input)
    if text is None:
        click.echo("Error: provide content.", err=True)
        sys.exit(1)

    path = Path(content) if content and Path(content).is_file() else None
    if path:
        result = scorer.score(str(path))
    else:
        result = scorer.score_text(text)

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


@cli.command()
@click.argument("content", required=False)
@click.option("--file", "-f", "file_path", help="Path to content file.")
@click.option("--text", "-t", "text_input", help="Inline text to render.")
@click.option("--output", "-o", default="midas_brain.svg", help="Output SVG path.")
@click.option(
    "--views", default="left,right",
    help="Comma-separated views: left, right, dorsal, ventral, etc.",
)
@click.pass_context
def render(ctx, content, file_path, text_input, output, views):
    """Generate an SVG brain activation map.

    Requires brain + plotting dependencies: pip install -e ".[all]"
    """
    _require_brain("render")

    from .brain_svg import render_brain_svg
    from .scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer(
        model_id=ctx.obj["model"],
        device=ctx.obj["device"],
        cache_folder=ctx.obj["cache"],
    )

    text = _resolve_text(content, file_path, text_input)
    if text is None:
        click.echo("Error: provide content.", err=True)
        sys.exit(1)

    path = Path(content) if content and Path(content).is_file() else None
    if path:
        result = scorer.score(str(path))
    else:
        result = scorer.score_text(text)

    view_list = [v.strip() for v in views.split(",")]
    render_brain_svg(
        activation=result.raw_activation,
        nes=result.nes,
        tier=result.tier,
        output=output,
        views=view_list,
    )
    click.echo(f"Brain SVG saved to {output}")


@cli.command()
@click.argument("content", required=False)
@click.option("--file", "-f", "file_path", help="Path to content file.")
@click.option("--text", "-t", "text_input", help="Inline text to optimize.")
@click.option("--loops", "-n", default=5, help="Max rewrite iterations (default 5).")
@click.option("--config", "-c", "config_path", help="Path to structural scoring YAML config.")
@click.option("--personas", "-p", "personas_path", help="Path to custom personas YAML file.")
@click.option("--no-brain", is_flag=True, help="Skip brain model (fast mode).")
@click.option("--no-agents", is_flag=True, help="Skip agent simulation.")
@click.option("--rewrite-model", default="claude-sonnet-4-6", help="Model for rewrites.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--svg", "svg_path", default=None, help="Save brain SVG of final version.")
@click.pass_context
def optimize(ctx, content, file_path, text_input, loops, config_path, personas_path, no_brain, no_agents, rewrite_model, as_json, svg_path):
    """Iteratively rewrite content to maximize scores.

    Runs evaluate -> rewrite cycles until the post hits SHIP or plateaus.
    Requires ANTHROPIC_API_KEY for rewrites (otherwise report-only mode).
    """
    from .optimizer import Optimizer

    text = _resolve_text(content, file_path, text_input)
    if text is None:
        click.echo("Error: provide content as argument, --file, or --text.", err=True)
        sys.exit(1)

    # Auto-skip brain if not installed
    if not no_brain and not _brain_available():
        click.echo("Brain model not installed (pip install -e \".[brain]\" to enable). Running structural + agents.")
        no_brain = True

    optimizer = Optimizer(
        rewrite_model=rewrite_model,
        model_id=ctx.obj["model"],
        device=ctx.obj["device"],
        cache_folder=ctx.obj["cache"],
    )

    def on_step(step):
        if as_json:
            return
        ev = step.evaluation
        parts = [f"Iteration {step.iteration}: structural {ev.structural.score:.0f} ({ev.structural.tier})"]
        if ev.agents is not None:
            parts.append(f"agents {ev.agents.share_rate:.0%} share")
        if ev.brain is not None:
            parts.append(f"brain NES {ev.brain.nes:.1f}")
        parts.append(f"verdict: {ev.verdict.upper()}")
        click.echo(" | ".join(parts))
        if step.iteration > 0 and step.changes and step.changes != "initial":
            click.echo(f"  -> Changes: {step.changes}")

    result = optimizer.optimize(
        text,
        max_loops=loops,
        skip_brain=no_brain,
        skip_agents=no_agents,
        config_path=config_path,
        personas_path=personas_path,
        on_step=on_step,
    )

    if as_json:
        import json

        data = {
            "total_iterations": result.total_iterations,
            "improved": result.improved,
            "final_verdict": result.final_evaluation.verdict,
            "initial_structural": round(result.steps[0].evaluation.structural.score, 1),
            "final_structural": round(result.final_evaluation.structural.score, 1),
            "final_text": result.final_text,
            "steps": [
                {
                    "iteration": s.iteration,
                    "structural_score": round(s.evaluation.structural.score, 1),
                    "structural_tier": s.evaluation.structural.tier,
                    "verdict": s.evaluation.verdict,
                    "changes": s.changes,
                    "agents_share_rate": round(s.evaluation.agents.share_rate, 2) if s.evaluation.agents else None,
                    "brain_nes": round(s.evaluation.brain.nes, 1) if s.evaluation.brain else None,
                }
                for s in result.steps
            ],
        }
        click.echo(json.dumps(data, indent=2))
    else:
        # Print summary
        initial = result.steps[0].evaluation.structural.score
        final = result.final_evaluation.structural.score
        click.echo(f"\nDone! {result.total_iterations} iterations, "
                    f"structural {initial:.0f} -> {final:.0f}, "
                    f"verdict: {result.final_evaluation.verdict.upper()}")
        click.echo(f"\n{'='*60}\nFinal post:\n{'='*60}\n")
        click.echo(result.final_text)

    # SVG render of final version
    if svg_path:
        if not _brain_available():
            click.echo("Skipping SVG render (brain model not installed).", err=True)
        else:
            try:
                from .brain_svg import render_brain_svg
                from .scorer import NeuralEngagementScorer

                scorer = NeuralEngagementScorer(
                    model_id=ctx.obj["model"],
                    device=ctx.obj["device"],
                    cache_folder=ctx.obj["cache"],
                )
                brain_result = scorer.score_text(result.final_text)
                render_brain_svg(
                    activation=brain_result.raw_activation,
                    nes=brain_result.nes,
                    tier=brain_result.tier,
                    output=svg_path,
                )
                click.echo(f"Brain SVG saved to {svg_path}")
            except Exception as e:
                click.echo(f"SVG render failed: {e}", err=True)


@cli.command()
@click.argument("content", required=False)
@click.option("--file", "-f", "file_path", help="Path to content file.")
@click.option("--text", "-t", "text_input", help="Inline text to evaluate.")
@click.option("--config", "-c", "config_path", help="Path to structural scoring YAML config.")
@click.option("--personas", "-p", "personas_path", help="Path to custom personas YAML file.")
@click.option("--no-brain", is_flag=True, help="Skip brain model (fast mode: structural + agents only).")
@click.option("--no-agents", is_flag=True, help="Skip agent simulation.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--svg", "svg_path", default=None, help="Save brain SVG alongside evaluation.")
@click.pass_context
def evaluate(ctx, content, file_path, text_input, config_path, personas_path, no_brain, no_agents, as_json, svg_path):
    """Evaluate content with all three mechanisms and get a unified verdict.

    Runs the brain model, agent simulation, and structural scoring, then
    synthesizes a ship/revise/kill verdict.
    """
    from .evaluator import Evaluator

    text = _resolve_text(content, file_path, text_input)
    if text is None:
        click.echo("Error: provide content as argument, --file, or --text.", err=True)
        sys.exit(1)

    # Auto-skip brain if not installed
    if not no_brain and not _brain_available():
        click.echo("Brain model not installed (pip install -e \".[brain]\" to enable). Running structural + agents.")
        no_brain = True

    evaluator = Evaluator(
        model_id=ctx.obj["model"],
        device=ctx.obj["device"],
        cache_folder=ctx.obj["cache"],
    )

    result = evaluator.evaluate(
        text,
        config_path=config_path,
        personas_path=personas_path,
        skip_brain=no_brain,
        skip_agents=no_agents,
    )

    if as_json:
        import json

        data = {
            "verdict": result.verdict,
            "confidence": round(result.confidence, 2),
            "explanation": result.explanation,
            "structural": {
                "score": round(result.structural.score, 1),
                "tier": result.structural.tier,
                "signals": result.structural.signals,
                "penalties": result.structural.penalties,
                "suggestions": result.structural.suggestions,
            },
        }
        if result.brain is not None:
            data["brain"] = {
                "nes": round(result.brain.nes, 1),
                "tier": result.brain.tier,
                "group_scores": {k: round(v, 4) for k, v in result.brain.group_scores.items()},
            }
        if result.agents is not None:
            data["agents"] = {
                "share_rate": round(result.agents.share_rate, 2),
                "engagement_rate": round(result.agents.engagement_rate, 2),
                "avg_confidence": round(result.agents.avg_confidence, 2),
                "reactions": [
                    {
                        "persona": r.persona_name,
                        "action": r.action,
                        "confidence": round(r.confidence, 2),
                        "reason": r.reason,
                    }
                    for r in result.agents.reactions
                ],
            }
        click.echo(json.dumps(data, indent=2))
    else:
        _print_evaluation(result)

    # SVG render
    if svg_path and result.brain is not None:
        try:
            from .brain_svg import render_brain_svg
            render_brain_svg(
                activation=result.brain.raw_activation,
                nes=result.brain.nes,
                tier=result.brain.tier,
                output=svg_path,
            )
            click.echo(f"Brain SVG saved to {svg_path}")
        except Exception as e:
            click.echo(f"SVG render failed: {e}", err=True)


def _print_evaluation(result):
    """Pretty-print a unified evaluation result."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Verdict panel
        verdict_color = {"ship": "bold green", "revise": "bold yellow", "kill": "bold red"}.get(
            result.verdict, "bold white"
        )
        verdict_icon = {"ship": "GO", "revise": "REVISE", "kill": "KILL"}.get(
            result.verdict, result.verdict.upper()
        )

        console.print(Panel(
            f"[{verdict_color}]{verdict_icon}[/{verdict_color}] — "
            f"confidence {result.confidence:.0%}\n\n"
            f"{result.explanation}",
            title="Midas v2 Evaluation",
            border_style="blue",
        ))

        # Mechanism breakdown table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Mechanism", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Tier / Rate", style="yellow")
        table.add_column("Signal", min_width=30)

        if result.brain is not None:
            table.add_row(
                "Brain Model",
                f"{result.brain.nes:.1f}",
                result.brain.tier,
                f"Top regions: {', '.join(result.brain.top_regions[:3])}",
            )

        table.add_row(
            "Structural",
            f"{result.structural.score:.0f}",
            result.structural.tier,
            f"{len(result.structural.signals)} signals, {len(result.structural.penalties)} penalties",
        )

        if result.agents is not None:
            sharers = [r.persona_name.split()[0] for r in result.agents.reactions if r.action == "share"]
            table.add_row(
                "Agent Sim",
                f"{result.agents.engagement_rate:.0%}",
                f"{result.agents.share_rate:.0%} share",
                f"Sharers: {', '.join(sharers) if sharers else 'none'}",
            )

        console.print(table)

        # Show structural suggestions if verdict is revise/kill
        if result.verdict != "ship" and result.structural.suggestions:
            console.print("\n[bold]Quick wins:[/bold]")
            for s in result.structural.suggestions[:5]:
                console.print(f"  -> {s}")

        # Show agent reactions if available
        if result.agents is not None:
            console.print("\n[bold]Persona reactions:[/bold]")
            for r in result.agents.reactions:
                icon = {"share": "[green]+[/green]", "comment": "[yellow]~[/yellow]",
                        "save": "[blue]*[/blue]", "scroll_past": "[red]-[/red]"}.get(r.action, "?")
                console.print(
                    f"  {icon} {r.persona_name}: {r.action} ({r.confidence:.0%}) — {r.reason}"
                )

    except ImportError:
        # Fallback: plain text
        click.echo(str(result))


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
