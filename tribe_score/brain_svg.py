"""SVG brain activation map renderer.

Uses PlotBrainNilearn from tribev2.plotting to generate publication-quality
SVG brain renders with activation overlays and region annotations.

Requires: pip install -e ".[all]" (brain + plotting dependencies)
"""

from __future__ import annotations

from .regions import EMPIRICAL_REGIONS


# The 10 empirical regions to annotate on the brain surface
_ANNOTATE_ROIS = list(EMPIRICAL_REGIONS.keys())


def render_brain_svg(
    activation,
    nes: float,
    tier: str,
    output: str = "midas_brain.svg",
    views: list[str] | None = None,
) -> str:
    """Render a brain activation map as SVG.

    Parameters
    ----------
    activation : np.ndarray
        Raw vertex activation array from NeuralScoreResult.raw_activation.
    nes : float
        Neural Engagement Score (0-100).
    tier : str
        Score tier label.
    output : str
        Output file path (default: midas_brain.svg).
    views : list[str], optional
        Brain views to render (default: ["left", "right"]).

    Returns
    -------
    str
        The output file path.
    """
    try:
        import matplotlib.pyplot as plt
        from tribev2.plotting.cortical import PlotBrainNilearn
    except ImportError:
        raise ImportError(
            "SVG brain render requires plotting dependencies. "
            "Install with: pip install -e '.[all]'"
        ) from None

    if views is None:
        views = ["left", "right"]

    plotter = PlotBrainNilearn()
    fig, axes = plotter.get_fig_axes(views)
    plotter.plot_surf(
        activation,
        views=views,
        axes=axes,
        cmap="hot",
        colorbar=True,
        colorbar_title="Activation",
        annotated_rois=_ANNOTATE_ROIS,
    )

    fig.suptitle(f"NES: {nes:.1f} — {tier}", fontsize=12, fontweight="bold")
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)

    return output
