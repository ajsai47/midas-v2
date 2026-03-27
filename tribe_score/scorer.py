"""Neural engagement scoring engine built on TRIBE v2 brain predictions."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .regions import DEFAULT_WEIGHTS, ENGAGEMENT_REGIONS, get_tier

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
    dimensions: dict[str, float]  # per-dimension 0-100 scores
    top_regions: list[str]  # top-k most activated brain regions
    temporal_profile: np.ndarray  # (n_segments,) mean activation over time
    raw_activation: np.ndarray  # (n_vertices,) mean vertex activations

    def __str__(self) -> str:
        parts = [f"NES: {self.nes:.1f} — {self.tier}"]
        parts.append(f"  {self.tier_description}")
        parts.append("  Dimensions:")
        for name, val in sorted(self.dimensions.items(), key=lambda x: -x[1]):
            bar = "#" * int(val / 5)
            parts.append(f"    {name:25s} {val:5.1f}  {bar}")
        parts.append(f"  Top regions: {', '.join(self.top_regions[:5])}")
        return "\n".join(parts)


class NeuralEngagementScorer:
    """Score content by predicting brain activation with TRIBE v2.

    Parameters
    ----------
    model_id : str
        HuggingFace model id or local checkpoint path.
    weights : dict
        Engagement dimension weights (must sum to ~1.0).
    device : str
        Torch device. "auto" picks CUDA if available.
    cache_folder : str
        Cache directory for TRIBE feature extraction.
    """

    def __init__(
        self,
        model_id: str = "facebook/tribev2",
        weights: dict[str, float] | None = None,
        device: str = "auto",
        cache_folder: str | None = None,
    ):
        self.model_id = model_id
        self.weights = weights or DEFAULT_WEIGHTS.copy()
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
        # Build config overrides for local environment
        import torch as _torch
        config_update = {}

        # Override device for feature extractors if no CUDA
        if not _torch.cuda.is_available():
            config_update["data.text_feature.device"] = "cpu"
            config_update["data.audio_feature.device"] = "cpu"
            config_update["data.video_feature.image.device"] = "cpu"

        # Use ungated copy of Llama 3.2-3B if the original is inaccessible
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
        """Return the get_events_dataframe kwarg name for a file path."""
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

    def _compute_dimension_scores(
        self, mean_activation: np.ndarray
    ) -> dict[str, float]:
        """Compute per-dimension scores from mean vertex activation."""
        from tribev2.utils import get_hcp_roi_indices

        dimension_scores = {}
        for dim_name, regions in ENGAGEMENT_REGIONS.items():
            try:
                indices = get_hcp_roi_indices(regions)
                dim_activation = mean_activation[indices].mean()
                dimension_scores[dim_name] = float(dim_activation)
            except (ValueError, IndexError) as e:
                logger.warning("Skipping dimension %s: %s", dim_name, e)
                dimension_scores[dim_name] = 0.0
        return dimension_scores

    def _normalize_dimensions(
        self, raw_dims: dict[str, float]
    ) -> dict[str, float]:
        """Normalize raw dimension activations to 0-100 scale.

        Uses min-max normalization across dimensions with a sigmoid stretch
        to spread values across the range.
        """
        values = np.array(list(raw_dims.values()))
        if values.max() == values.min():
            return {k: 50.0 for k in raw_dims}

        # Sigmoid-based normalization: center on mean, stretch by std
        mean, std = values.mean(), values.std()
        if std < 1e-8:
            std = 1.0
        z_scores = (values - mean) / std
        # Sigmoid maps z-scores to (0, 1), then scale to 0-100
        normalized = 100.0 / (1.0 + np.exp(-z_scores))
        return dict(zip(raw_dims.keys(), normalized.tolist()))

    def score(self, path: str, modality: str | None = None) -> NeuralScoreResult:
        """Score a content file and return detailed neural engagement breakdown.

        Parameters
        ----------
        path : str
            Path to content file (.txt, .mp3, .mp4, etc.)
        modality : str, optional
            Override modality detection. One of "text_path", "audio_path", "video_path".
        """
        from tribev2.utils import get_topk_rois

        preds = self._predict(path, modality)

        # Temporal profile: mean activation per segment
        temporal_profile = preds.mean(axis=1)

        # Mean activation across all time segments
        mean_activation = preds.mean(axis=0)

        # Per-dimension raw activations
        raw_dims = self._compute_dimension_scores(mean_activation)

        # Normalize to 0-100
        norm_dims = self._normalize_dimensions(raw_dims)

        # Weighted composite score
        nes = sum(
            norm_dims[dim] * self.weights.get(dim, 0.0)
            for dim in norm_dims
        )

        # Top activated regions
        top_regions = list(get_topk_rois(mean_activation, k=10))

        tier, tier_desc = get_tier(nes)

        return NeuralScoreResult(
            nes=nes,
            tier=tier,
            tier_description=tier_desc,
            dimensions=norm_dims,
            top_regions=top_regions,
            temporal_profile=temporal_profile,
            raw_activation=mean_activation,
        )

    def score_text(self, text: str) -> NeuralScoreResult:
        """Convenience method to score inline text.

        Writes text to a temp file and scores it as text modality.
        """
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
