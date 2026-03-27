"""tribe_score — Neural engagement scoring powered by TRIBE v2 brain predictions."""

__version__ = "0.1.0"

from .regions import DEFAULT_WEIGHTS, ENGAGEMENT_REGIONS
from .scorer import NeuralEngagementScorer, NeuralScoreResult
