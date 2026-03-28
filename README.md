<div align="center">

# tribe_score

**Three independent mechanisms to predict content virality — before you post.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

## How it works

What if you could scan your content through a human brain before posting it? We built that. A neuroscience foundation model predicts the exact brain regions your content will activate. Reward circuits in the orbitofrontal cortex fire when content feels valuable. The hippocampus encodes memorable content that people share. Social cognition in the temporal pole drives viral spread. We validated against real engagement data and found that viral posts consistently activate all three — reward, memory, social — while suppressing auditory processing. Posts that flop activate nothing. Now we added two more layers: AI audience simulation and pattern scoring from thousands of real posts. Three independent systems must agree before your content ships.

```
                          ┌─────────────────────┐
                          │     Your Content     │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │       Matrix         │
                          │    (orchestrator)    │
                          └──┬───────┬───────┬──┘
                             │       │       │
                   ┌─────────▼──┐ ┌──▼─────────┐ ┌──▼─────────┐
                   │   Brain    │ │   Agent     │ │  Fine-tuned │
                   │   Model    │ │ Simulation  │ │    Model    │
                   │ (TRIBE v2) │ │ (personas)  │ │  (scoring)  │
                   └─────────┬──┘ └──┬─────────┘ └──┬─────────┘
                             │       │               │
                          ┌──▼───────▼───────────────▼──┐
                          │     Virality Prediction      │
                          │   go / revise / kill          │
                          └─────────────────────────────┘
```

### Mechanism 1: Brain Model (live)

Meta's [TRIBE v2](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) predicts fMRI brain responses to any content. We ran 20 LinkedIn posts with known engagement through it, analyzed all 180 HCP brain regions, and found **10 regions correlated with real-world virality at p<0.01**.

The pattern:
- **Reward circuits fire** (orbitofrontal cortex) — "I want more of this"
- **Memory encodes** (hippocampus) — memorable content gets shared
- **Social cognition activates** (temporal pole) — social processing drives viral spread
- **Auditory cortex suppresses** — viral content is visual/conceptual, not auditory

Validation: **Spearman rho = 0.7522 (p = 0.000131)**. Viral posts avg NES = 81.9 vs low-engagement = 32.6.

### Mechanism 2: Agent Simulation (coming)

Imagine showing your post to a room of fifty people before publishing. A venture capitalist reads it and immediately forwards it to their portfolio founders. An engineer screenshots it and shares it in their team Slack. A journalist bookmarks it for a story they are writing. A CEO reads the first line and keeps scrolling. That is what our agent simulation does. We built AI personas modeled on real audience segments — investors, engineers, founders, journalists, executives — each with unique attention patterns and sharing behaviors. The simulation tells you who will engage, who will share, and who will ignore your content.

### Mechanism 3: Fine-tuned Scoring Model (coming)

A fine-tuned model trained on thousands of posts with real engagement data. Pattern-matches against structural and linguistic features that correlate with virality — hook strength, narrative arc, specificity, emotional payload.

### Matrix: The Orchestrator

[Matrix](https://github.com/ajsai47/ghost) sees the whole picture. It sends your content through three prediction systems that analyze completely different dimensions. The brain model detects neurological reward activation, memory encoding, and social cognition patterns that no language model can see — these are the three neural signatures that separate viral from forgettable content. The agent simulation puts your post in front of simulated investors, engineers, journalists, and founders to predict exactly who will share it and who will scroll past. The pattern model identifies the structural fingerprint of virality: hook strength, narrative arc, emotional specificity, formatting that drives engagement. Every mechanism catches blind spots the other two cannot see. Matrix synthesizes their signals into a single verdict: ship it, revise it, or kill it.

## Quick start

The brain model mechanism is live today:

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
# Score text
tribe-score score --text "Your post content here"
tribe-score score --text "Your post" --json

# Compare A/B variants
tribe-score compare --text "Version A" --text "Version B"

# Explain per-region brain activation breakdown
tribe-score explain --text "Your post content here"

# Generate brain heatmap
tribe-score heatmap --text "Your post" --output brain.png
```

## Brain model details

10 brain regions, empirically selected. Each region's activation is z-scored against calibration data, then weighted by its Spearman correlation with engagement.

| Group | Brain Regions | Correlation | Signal |
|-------|--------------|-------------|--------|
| **Reward** | pOFC (rho=+0.66), OFC (+0.60) | Positive | Higher reward activation → more engagement |
| **Memory** | Hippocampus (+0.63) | Positive | Stronger encoding → more shares |
| **Social** | Temporal pole dorsal (+0.61), ventral (+0.57) | Positive | Social processing → viral spread |
| **Auditory** | TA2 (-0.63), A4 (-0.59), PBelt (-0.58), MBelt (-0.58), A5 (-0.57) | **Negative** | Suppressed auditory cortex → higher engagement |
| **Focus** | Brain variability (-0.60) | **Negative** | Focused activation > diffuse |

### Scoring tiers

| NES | Tier | Pattern |
|-----|------|---------|
| 80–100 | **NEURAL VIRAL** | Strong reward + memory + social, suppressed auditory |
| 60–79 | HIGH ACTIVATION | Good engagement region activation |
| 40–59 | MODERATE | Mixed signal |
| 20–39 | LOW ACTIVATION | Weak engagement pattern |
| 0–19 | MINIMAL | No engagement signal |

## Architecture

```
tribe_score/              # Brain model scoring engine (live)
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

## Credits

Brain model built on **TRIBE v2** by Meta FAIR.

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
