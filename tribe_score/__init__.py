"""tribe_score — Three mechanisms to predict content virality before you post."""

__version__ = "0.3.0"

from .structural.config import MidasConfig
from .structural.scorer import ScoreResult, score as structural_score


def __getattr__(name):
    if name in ("NeuralEngagementScorer", "NeuralScoreResult"):
        from .scorer import NeuralEngagementScorer, NeuralScoreResult
        return locals()[name]
    if name in ("Evaluator", "EvaluationResult"):
        from .evaluator import Evaluator, EvaluationResult
        return locals()[name]
    if name in ("Optimizer", "OptimizationResult", "OptimizationStep"):
        from .optimizer import Optimizer, OptimizationResult, OptimizationStep
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
