"""Neural engagement scoring engine built on TRIBE v2 brain predictions.

Scoring formula derived from empirical correlation analysis (n=20 LinkedIn posts).
Uses 10 brain regions at p<0.01 significance + brain variability signal.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
except ImportError:
    raise ImportError(
        "Brain model requires additional dependencies. "
        "Install with: pip install -e '.[brain]'"
    ) from None

from .regions import (
    CALIBRATION,
    CALIBRATION_RAW_RANGE,
    EMPIRICAL_REGIONS,
    REGION_GROUPS,
    TIERS,
    VARIABILITY_RHO,
    get_tier,
)

logger = logging.getLogger(__name__)

# File extension -> modality kwarg for TribeModel.get_events_dataframe()
_EXT_TO_MODALITY: dict[str, str] = {
    ".txt": "text_path",
    ".wav": "audio_path",
    ".mp3": "audio_path",
    ".flac": "audio_path",
    ".ogg": "audio_path",
    ".mp4": "video_path",
    ".avi": "video_path",
    ".mkv": "video_path",
    ".mov": "video_path",
    ".webm": "video_path",
}


@dataclass
class NeuralScoreResult:
    """Detailed neural engagement scoring breakdown."""

    nes: float  # 0-100 composite score
    tier: str
    tier_description: str

    # Per-region z-scores (how far above/below calibration mean)
    region_zscores: dict[str, float]

    # Grouped scores for display
    group_scores: dict[str, float]

    # Top activated regions (from full 180-region atlas)
    top_regions: list[str]

    # Temporal dynamics
    temporal_profile: np.ndarray  # (n_segments,) mean activation over time

    # Raw data for heatmaps/debugging
    raw_activation: np.ndarray  # (n_vertices,) mean vertex activations
    brain_magnitude: float = 0.0
    brain_variability: float = 0.0
    variability_zscore: float = 0.0
    raw_score: float = 0.0  # pre-normalization score

    def __str__(self) -> str:
        parts = [f"NES: {self.nes:.1f}/100 — {self.tier}"]
        parts.append(f"  {self.tier_description}")
        parts.append("")
        parts.append("  Region Group Scores:")
        for group, score in sorted(self.group_scores.items(), key=lambda x: -x[1]):
            bar = "+" * max(0, int(score / 5)) if score > 0 else "-" * max(0, int(-score / 5))
            parts.append(f"    {group:30s} {score:+6.2f}  {bar}")
        parts.append(f"\n  Brain focus: {self.variability_zscore:+.2f}σ (lower variability = more focused)")
        parts.append(f"  Magnitude: {self.brain_magnitude:.4f}  Variability: {self.brain_variability:.4f}")
        parts.append(f"  Top regions: {', '.join(self.top_regions[:5])}")
        return "\n".join(parts)


class NeuralEngagementScorer:
    """Score content using empirically-validated brain region correlations.

    The scoring formula uses 10 HCP brain regions that showed statistically
    significant (p<0.01) correlation with LinkedIn post engagement in a
    20-post calibration study. Each region's activation is z-scored against
    calibration data, then weighted by its Spearman correlation coefficient.

    Positive-rho regions (reward, memory, social): higher activation = higher score
    Negative-rho regions (auditory cortex): lower activation = higher score
    Brain variability: lower = more focused = higher score
    """

    def __init__(
        self,
        model_id: str = "facebook/tribev2",
        device: str = "auto",
        cache_folder: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.cache_folder = cache_folder or str(
            Path.home() / ".cache" / "tribe_score"
        )
        self._model = None

    def _ensure_model(self):
        """Lazy-load the TRIBE model on first use."""
        if self._model is not None:
            return
        from tribev2.demo_utils import TribeModel

        logger.info("Loading TRIBE model from %s...", self.model_id)
        import torch as _torch

        config_update = {}
        if not _torch.cuda.is_available():
            config_update["data.text_feature.device"] = "cpu"
            config_update["data.audio_feature.device"] = "cpu"
            config_update["data.video_feature.image.device"] = "cpu"

        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B")
        except (OSError, Exception):
            logger.info("meta-llama/Llama-3.2-3B not accessible, using unsloth/Llama-3.2-3B")
            config_update["data.text_feature.model_name"] = "unsloth/Llama-3.2-3B"

        self._model = TribeModel.from_pretrained(
            self.model_id,
            cache_folder=self.cache_folder,
            device=self.device,
            config_update=config_update or None,
        )
        logger.info("Model loaded on %s", self.device)

    def _detect_modality(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext not in _EXT_TO_MODALITY:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {sorted(_EXT_TO_MODALITY.keys())}"
            )
        return _EXT_TO_MODALITY[ext]

    def _predict(self, path: str, modality: str | None = None) -> np.ndarray:
        """Run TRIBE prediction and return (n_segments, n_vertices) array."""
        self._ensure_model()
        if modality is None:
            modality = self._detect_modality(path)
        events = self._model.get_events_dataframe(**{modality: path})
        preds, _ = self._model.predict(events, verbose=False)
        return preds

    def _compute_empirical_score(
        self, mean_activation: np.ndarray, brain_variability: float
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Compute engagement score using empirically-validated regions.

        Returns (raw_score, region_zscores, group_scores).
        """
        from tribev2.utils import summarize_by_roi, get_hcp_labels

        # Get per-ROI activations
        roi_activations = summarize_by_roi(mean_activation)
        region_labels = get_hcp_labels()

        # Build label -> activation lookup
        label_to_activation: dict[str, float] = {}
        for j, label in enumerate(region_labels):
            if label != "?":
                label_to_activation[label] = float(roi_activations[j])

        # Z-score each empirical region and weight by correlation
        raw_score = 0.0
        region_zscores: dict[str, float] = {}

        for region, rho in EMPIRICAL_REGIONS.items():
            if region not in label_to_activation:
                logger.warning("Region %s not found in atlas", region)
                continue
            if region not in CALIBRATION:
                continue

            activation = label_to_activation[region]
            cal_mean, cal_std = CALIBRATION[region]
            if cal_std < 1e-8:
                cal_std = 1.0

            z = (activation - cal_mean) / cal_std
            region_zscores[region] = z

            # rho * z: positive rho + positive z = good
            # negative rho + negative z = good (double negative = positive)
            raw_score += rho * z

        # Add brain variability signal
        var_mean, var_std = CALIBRATION["_variability"]
        if var_std < 1e-8:
            var_std = 1.0
        var_z = (brain_variability - var_mean) / var_std
        raw_score += VARIABILITY_RHO * var_z

        # Compute group scores for display
        group_scores: dict[str, float] = {}
        for group_name, regions in REGION_GROUPS.items():
            group_total = 0.0
            for r in regions:
                if r in region_zscores:
                    group_total += EMPIRICAL_REGIONS[r] * region_zscores[r]
            group_scores[group_name] = group_total

        group_scores["Focus (low variability)"] = VARIABILITY_RHO * var_z

        return raw_score, region_zscores, group_scores

    def _normalize_score(self, raw_score: float) -> float:
        """Normalize raw score to 0-100 using calibration range."""
        lo, hi = CALIBRATION_RAW_RANGE
        if hi <= lo:
            return 50.0
        normalized = (raw_score - lo) / (hi - lo) * 100.0
        return float(np.clip(normalized, 0, 100))

    def score(self, path: str, modality: str | None = None) -> NeuralScoreResult:
        """Score a content file and return detailed neural engagement breakdown."""
        from tribev2.utils import get_topk_rois

        preds = self._predict(path, modality)

        temporal_profile = preds.mean(axis=1)
        mean_activation = preds.mean(axis=0)

        brain_magnitude = float(mean_activation.mean())
        brain_variability = float(mean_activation.std())

        raw_score, region_zscores, group_scores = self._compute_empirical_score(
            mean_activation, brain_variability
        )

        nes = self._normalize_score(raw_score)
        top_regions = list(get_topk_rois(mean_activation, k=10))
        tier, tier_desc = get_tier(nes)

        var_mean, var_std = CALIBRATION["_variability"]
        variability_zscore = (brain_variability - var_mean) / max(var_std, 1e-8)

        return NeuralScoreResult(
            nes=nes,
            tier=tier,
            tier_description=tier_desc,
            region_zscores=region_zscores,
            group_scores=group_scores,
            top_regions=top_regions,
            temporal_profile=temporal_profile,
            raw_activation=mean_activation,
            brain_magnitude=brain_magnitude,
            brain_variability=brain_variability,
            variability_zscore=variability_zscore,
            raw_score=raw_score,
        )

    def score_text(self, text: str) -> NeuralScoreResult:
        """Score inline text."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            f.flush()
            return self.score(f.name, modality="text_path")

    def compare(self, *paths: str) -> list[NeuralScoreResult]:
        """Score multiple content files and return results sorted by NES."""
        results = [self.score(p) for p in paths]
        results.sort(key=lambda r: r.nes, reverse=True)
        return results
