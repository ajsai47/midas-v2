"""Unified evaluator — runs all three mechanisms and synthesizes a verdict.

Mechanism 1: Brain Model (TRIBE v2 neural engagement scoring)
Mechanism 2: Agent Simulation (LLM persona reactions)
Mechanism 3: Structural Scoring (config-driven pattern matching)
"""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from .scorer import NeuralScoreResult

from .structural.config import MidasConfig
from .structural.scorer import ScoreResult
from .structural.scorer import score as structural_score

logger = logging.getLogger(__name__)

# Structural tiers ordered best-to-worst for verdict logic
_STRUCTURAL_TOP_TIERS = {"VIRAL CANDIDATE", "HIGH PERFORMER"}
_STRUCTURAL_BOTTOM_TIERS = {"BELOW AVERAGE", "AVERAGE"}


@dataclass
class EvaluationResult:
    """Combined result from all three scoring mechanisms."""

    brain: NeuralScoreResult | None  # Mechanism 1
    structural: ScoreResult  # Mechanism 3
    agents: "AgentSimResult | None"  # Mechanism 2 (None if no API key)
    verdict: str  # "ship" / "revise" / "kill"
    confidence: float  # 0.0-1.0
    explanation: str

    def __str__(self) -> str:
        parts = [f"Verdict: {self.verdict.upper()} (confidence: {self.confidence:.0%})"]
        parts.append(f"  {self.explanation}")
        parts.append("")

        if self.brain is not None:
            parts.append(f"  Brain Model:  NES {self.brain.nes:.1f} — {self.brain.tier}")
        else:
            parts.append("  Brain Model:  skipped")

        parts.append(f"  Structural:   {self.structural.score:.0f} — {self.structural.tier}")

        if self.agents is not None:
            parts.append(
                f"  Agent Sim:    {self.agents.engagement_rate:.0%} engagement, "
                f"{self.agents.share_rate:.0%} share rate"
            )
        else:
            parts.append("  Agent Sim:    skipped (no ANTHROPIC_API_KEY)")

        return "\n".join(parts)


def _compute_verdict(
    brain: NeuralScoreResult | None,
    structural: ScoreResult,
    agents: "AgentSimResult | None",
) -> tuple[str, float, str]:
    """Determine verdict from all three mechanisms.

    Returns (verdict, confidence, explanation).
    """
    reasons_ship: list[str] = []
    reasons_kill: list[str] = []
    reasons_revise: list[str] = []

    # Brain model signals
    if brain is not None:
        if brain.nes >= 60:
            reasons_ship.append(f"brain NES {brain.nes:.0f} (strong neural activation)")
        elif brain.nes < 30:
            reasons_kill.append(f"brain NES {brain.nes:.0f} (weak neural signal)")
        else:
            reasons_revise.append(f"brain NES {brain.nes:.0f} (moderate — mixed signal)")

    # Structural signals
    if structural.tier in _STRUCTURAL_TOP_TIERS:
        reasons_ship.append(f"structural tier {structural.tier}")
    elif structural.tier in _STRUCTURAL_BOTTOM_TIERS:
        reasons_kill.append(f"structural tier {structural.tier}")
    else:
        reasons_revise.append(f"structural tier {structural.tier}")

    # Agent simulation signals
    if agents is not None:
        if agents.share_rate >= 0.4:
            reasons_ship.append(f"agent share rate {agents.share_rate:.0%}")
        elif agents.share_rate < 0.2:
            reasons_kill.append(f"agent share rate {agents.share_rate:.0%}")
        else:
            reasons_revise.append(f"agent share rate {agents.share_rate:.0%}")

    # Determine verdict
    # Count how many active mechanisms we have
    active = 1  # structural is always active
    if brain is not None:
        active += 1
    if agents is not None:
        active += 1

    ship_count = len(reasons_ship)
    kill_count = len(reasons_kill)

    if ship_count == active:
        verdict = "ship"
        confidence = 0.9
        explanation = "All mechanisms agree: " + ", ".join(reasons_ship)
    elif kill_count == active:
        verdict = "kill"
        confidence = 0.9
        explanation = "All mechanisms agree: " + ", ".join(reasons_kill)
    elif ship_count > kill_count and kill_count == 0:
        verdict = "ship"
        confidence = 0.7
        explanation = "Majority positive: " + ", ".join(reasons_ship + reasons_revise)
    elif kill_count > ship_count and ship_count == 0:
        verdict = "kill"
        confidence = 0.7
        explanation = "Majority negative: " + ", ".join(reasons_kill + reasons_revise)
    else:
        verdict = "revise"
        all_reasons = reasons_ship + reasons_revise + reasons_kill
        confidence = 0.5
        explanation = "Mixed signals: " + ", ".join(all_reasons)

    return verdict, confidence, explanation


class Evaluator:
    """Orchestrator that runs all three mechanisms and returns a unified verdict."""

    def __init__(
        self,
        model_id: str = "facebook/tribev2",
        device: str = "auto",
        cache_folder: str | None = None,
        anthropic_api_key: str | None = None,
        agent_model: str = "claude-haiku-4-5-20251001",
    ):
        self.model_id = model_id
        self.device = device
        self.cache_folder = cache_folder
        self.anthropic_api_key = anthropic_api_key
        self.agent_model = agent_model

    def evaluate(
        self,
        text: str,
        config: MidasConfig | None = None,
        config_path: str | None = None,
        personas_path: str | None = None,
        skip_brain: bool = False,
        skip_agents: bool = False,
    ) -> EvaluationResult:
        """Run all three mechanisms and return unified verdict.

        Parameters
        ----------
        text : str
            The content to evaluate.
        config : MidasConfig, optional
            Structural scoring config. If None, uses default.
        config_path : str, optional
            Path to a YAML config file for structural scoring.
        personas_path : str, optional
            Path to a YAML file with custom persona definitions.
        skip_brain : bool
            Skip the brain model (for fast structural+agent only).
        skip_agents : bool
            Skip agent simulation.
        """
        # Load structural config if path provided
        if config is None and config_path is not None:
            from .structural.config import load_config
            config = load_config(config_path)

        # Mechanism 1: Brain Model
        brain_result: NeuralScoreResult | None = None
        if not skip_brain:
            try:
                from .scorer import NeuralEngagementScorer
                scorer = NeuralEngagementScorer(
                    model_id=self.model_id,
                    device=self.device,
                    cache_folder=self.cache_folder,
                )
                brain_result = scorer.score_text(text)
                logger.info("Brain model: NES %.1f — %s", brain_result.nes, brain_result.tier)
            except Exception as e:
                logger.warning("Brain model failed (skipping): %s", e)

        # Mechanism 3: Structural Scoring
        struct_result = structural_score(text, config)
        logger.info("Structural: %.0f — %s", struct_result.score, struct_result.tier)

        # Mechanism 2: Agent Simulation
        agent_result = None
        if not skip_agents:
            try:
                from .agents.simulator import AgentSimResult, AgentSimulator
                sim = AgentSimulator(
                    api_key=self.anthropic_api_key,
                    model=self.agent_model,
                    personas_path=personas_path,
                )
                agent_result = sim.simulate(text)
                logger.info(
                    "Agent sim: %.0f%% engagement, %.0f%% share rate",
                    agent_result.engagement_rate * 100,
                    agent_result.share_rate * 100,
                )
            except RuntimeError as e:
                logger.info("Agent simulation skipped: %s", e)
            except Exception as e:
                logger.warning("Agent simulation failed: %s", e)

        # Synthesize verdict
        verdict, confidence, explanation = _compute_verdict(
            brain_result, struct_result, agent_result
        )

        return EvaluationResult(
            brain=brain_result,
            structural=struct_result,
            agents=agent_result,
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
        )
