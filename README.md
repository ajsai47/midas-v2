<div align="center">

# tribe_score

**Predict social media virality using brain activation patterns.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

*What if you could run content through a brain model and know if it'll go viral — before you post it?*

</div>

## The idea

Meta built [TRIBE v2](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) — a foundation model that predicts fMRI brain responses to any content (text, audio, video). It maps input through LLaMA 3.2, V-JEPA2, and Wav2Vec-BERT onto a cortical surface model of ~20k brain vertices.

We asked a different question: **do the brain regions that light up for viral content differ from non-viral content?**

Yes. Dramatically.

We ran 20 LinkedIn posts with known engagement data through TRIBE v2, analyzed activation across all 180 HCP brain regions, and found **10 regions correlated with real-world virality at p<0.01**. The pattern is clear:

- **Reward circuits fire** (orbitofrontal cortex) — the "I want more" response
- **Memory encodes** (hippocampus) — memorable content gets shared
- **Social cognition activates** (temporal pole) — social processing drives viral spread
- **Auditory cortex suppresses** — viral content is visual/conceptual, not auditory

We turned this into a scoring formula. Feed any text into `tribe_score`, and it tells you how viral-ready it is based on the neurological response pattern.

## Validation

| Metric | Value |
|--------|-------|
| Spearman correlation with real engagement | **rho = 0.7522** (p = 0.000131) |
| Viral post average NES | 81.9 |
| Low-engagement post average NES | 32.6 |
| Separation delta | +49.3 points |

The brain knows what's going to spread before the algorithm does.

## Quick start

```bash
git clone https://github.com/ajsai47/tribev2.git
cd tribev2
pip install -e .
```

### Score content

```python
from tribe_score import NeuralEngagementScorer

scorer = NeuralEngagementScorer()

result = scorer.score_text("I just got fired from Google. Best thing that ever happened.")
print(result.nes)            # 0-100 Neural Engagement Score
print(result.tier)           # NEURAL VIRAL / HIGH / MODERATE / LOW / MINIMAL
print(result.group_scores)   # reward, memory, social, auditory breakdown
```

### CLI

```bash
tribe-score score --text "Your post content here"
tribe-score score --text "Your post" --json
tribe-score compare --text "Version A" --text "Version B"
tribe-score explain --text "Why does this work?"
tribe-score heatmap --text "Your post" --output brain.png
```

### Compare variants

```bash
tribe-score compare \
  --text "I just raised $5M for my AI startup" \
  --text "Lessons from raising a seed round in 2026"
```

Returns a ranked table with NES scores and per-region breakdowns — pick the version that activates the right brain regions.

## The scoring formula

10 brain regions, empirically selected. Each region's activation is z-scored against calibration data, then weighted by its correlation with engagement.

| Group | Brain Regions | Correlation | Signal |
|-------|--------------|-------------|--------|
| **Reward** | pOFC (rho=+0.66), OFC (+0.60) | Positive | Higher reward activation → more engagement |
| **Memory** | Hippocampus (+0.63) | Positive | Stronger encoding → more shares |
| **Social** | Temporal pole dorsal (+0.61), ventral (+0.57) | Positive | Social processing → viral spread |
| **Auditory** | TA2 (-0.63), A4 (-0.59), PBelt (-0.58), MBelt (-0.58), A5 (-0.57) | **Negative** | Suppressed auditory cortex → higher engagement |
| **Focus** | Brain variability (-0.60) | **Negative** | Focused activation > diffuse |

### Tiers

| NES | Tier | Pattern |
|-----|------|---------|
| 80–100 | **NEURAL VIRAL** | Strong reward + memory + social, suppressed auditory |
| 60–79 | HIGH ACTIVATION | Good engagement region activation |
| 40–59 | MODERATE | Mixed signal |
| 20–39 | LOW ACTIVATION | Weak engagement pattern |
| 0–19 | MINIMAL | No engagement signal |

## What we built

~800 lines on top of Meta's TRIBE v2:

```
tribe_score/              # Our scoring engine
├── scorer.py             # Brain activation → engagement score
├── cli.py                # tribe-score CLI
├── compare.py            # A/B comparison tables
├── regions.py            # Empirical weights + calibration data
└── __init__.py

tribev2/                  # Meta's TRIBE v2 (upstream, 2 small CPU patches)
├── demo_utils.py         # TribeModel — predicts brain response to content
├── model.py              # FmriEncoder Transformer
├── utils.py              # HCP atlas, ROI analysis
└── ...
```

## Upstream credit

Built on **TRIBE v2** by Meta FAIR — a foundation model for in-silico neuroscience.

📄 [Paper](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) | 🤗 [Weights](https://huggingface.co/facebook/tribev2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)

```bibtex
@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brookes, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
  year={2026}
}
```

## License

CC-BY-NC-4.0. See [LICENSE](LICENSE).
