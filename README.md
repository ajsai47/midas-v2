<div align="center">

# TRIBE v2

**A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

📄 [Paper](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) ▶️ [Demo](https://aidemos.atmeta.com/tribev2/) | 🤗 [Weights](https://huggingface.co/facebook/tribev2)

</div>

TRIBE v2 is a deep multimodal brain encoding model that predicts fMRI brain responses to naturalistic stimuli (video, audio, text). It combines state-of-the-art feature extractors — [**LLaMA 3.2**](https://huggingface.co/meta-llama/Llama-3.2-3B) (text), [**V-JEPA2**](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256) (video), and [**Wav2Vec-BERT**](https://huggingface.co/facebook/w2v-bert-2.0) (audio) — into a unified Transformer architecture that maps multimodal representations onto the cortical surface.

## Quick start

Load a pretrained model from HuggingFace and predict brain responses to a video:

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

df = model.get_events_dataframe(video_path="path/to/video.mp4")
preds, segments = model.predict(events=df)
print(preds.shape)  # (n_timesteps, n_vertices)
```

Predictions are for the "average" subject (see paper for details) and live on the **fsaverage5** cortical mesh (~20k vertices). You can also pass `text_path` or `audio_path` to `model.get_events_dataframe` — text is automatically converted to speech and transcribed to obtain word-level timings.

For a full walkthrough with brain visualizations, see the [Colab demo notebook](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb).

## Installation

**Basic** (inference only):
```bash
pip install -e .
```

**With brain visualization**:
```bash
pip install -e ".[plotting]"
```

**With training dependencies** (PyTorch Lightning, W&B, etc.):
```bash
pip install -e ".[training]"
```

## Training a model from scratch

### 1. Set environment variables

Configure data/output paths and Slurm partition (or edit `tribev2/grids/defaults.py` directly):

```bash
export DATAPATH="/path/to/studies"
export SAVEPATH="/path/to/output"
export SLURM_PARTITION="your_partition"
```

### 2. Authenticate with HuggingFace

The text encoder requires access to the gated [LLaMA 3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) model:

```bash
huggingface-cli login
```

Create a `read` [access token](https://huggingface.co/settings/tokens) and paste it when prompted.

### 3. Run training

**Local test run:**
```bash
python -m tribev2.grids.test_run
```

**Grid search on Slurm:**
```bash
python -m tribev2.grids.run_cortical
python -m tribev2.grids.run_subcortical
```

## Project structure

```
tribev2/
├── main.py              # Experiment pipeline: Data, TribeExperiment
├── model.py             # FmriEncoder: Transformer-based multimodal→fMRI model
├── pl_module.py         # PyTorch Lightning training module
├── demo_utils.py        # TribeModel and helpers for inference from text/audio/video
├── eventstransforms.py  # Custom event transforms (word extraction, chunking, …)
├── utils.py             # Multi-study loading, splitting, subject weighting
├── utils_fmri.py        # Surface projection (MNI / fsaverage) and ROI analysis
├── grids/
│   ├── defaults.py      # Full default experiment configuration
│   └── test_run.py      # Quick local test entry point
├── plotting/            # Brain visualization (PyVista & Nilearn backends)
└── studies/             # Dataset definitions (Algonauts2025, Lahner2024, …)
```

---

## tribe_score — Neural Engagement Scoring

`tribe_score` is a scoring engine built on top of TRIBE v2 that predicts how engaging content will be on social media (LinkedIn). It uses 10 brain regions empirically correlated with post engagement (p<0.01, n=20 posts) to compute a **Neural Engagement Score (NES)** from 0–100.

**Validation**: Spearman rho=0.7522 (p=0.000131) with actual engagement. Viral posts average NES=81.9 vs low-engagement posts at NES=32.6.

### Python API

```python
from tribe_score import NeuralEngagementScorer

scorer = NeuralEngagementScorer()

# Score inline text
result = scorer.score_text("I just got fired from Google. Best thing that ever happened.")
print(result.nes)            # 0-100 composite score
print(result.tier)           # e.g. "NEURAL VIRAL"
print(result.group_scores)   # per-group breakdown
print(result.top_regions)    # top activated HCP regions

# Score a file
result = scorer.score("post.txt")

# Compare variants
results = scorer.compare("version_a.txt", "version_b.txt")
```

### CLI

```bash
# Score text
tribe-score score --text "Your post content here"
tribe-score score --text "Your post" --json

# Score a file
tribe-score score post.txt

# Compare variants
tribe-score compare --text "Version A" --text "Version B"

# Explain per-region breakdown
tribe-score explain --text "Your post content here"

# Generate brain heatmap (requires plotting extras)
tribe-score heatmap --text "Your post" --output brain.png
```

### How Scoring Works

The scorer extracts activation from 10 HCP brain regions that showed statistically significant correlation with LinkedIn engagement:

| Group | Regions | Signal |
|-------|---------|--------|
| **Reward** | pOFC, OFC | Higher activation → higher engagement |
| **Memory** | Hippocampus | Higher activation → higher engagement |
| **Social** | TGd, TGv (temporal pole) | Higher activation → higher engagement |
| **Auditory** | TA2, A4, PBelt, MBelt, A5 | **Suppressed** in viral content |

Each region's activation is z-scored against calibration data, then weighted by its Spearman correlation coefficient. Brain variability (std across vertices) is also factored in — more focused activation patterns score higher.

### Scoring Tiers

| NES Range | Tier | Meaning |
|-----------|------|---------|
| 80–100 | NEURAL VIRAL | Strong reward + memory + social, suppressed auditory |
| 60–79 | HIGH ACTIVATION | Good engagement pattern |
| 40–59 | MODERATE | Mixed signal |
| 20–39 | LOW ACTIVATION | Weak engagement pattern |
| 0–19 | MINIMAL | No engagement signal |

---

## Contributing to open science

If you use this software, please share your results with the broader research community using the following citation:

```bibtex
@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brookes, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
  year={2026}
}
```

## License

This project is licensed under CC-BY-NC-4.0. See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.
