"""tribe_score — Neural engagement scoring powered by TRIBE v2 brain predictions."""

__version__ = "0.2.0"

from .regions import EMPIRICAL_REGIONS, REGION_GROUPS
from .scorer import NeuralEngagementScorer, NeuralScoreResult
