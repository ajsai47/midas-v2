<div align="center">

# Midas v2

**Predict content virality before you post.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**Score.** Three independent prediction systems analyze your content.
**Improve.** The optimizer rewrites your post until it's ship-ready.
**Ship.** Or kill it before it flops.

</div>

## Demo

```bash
$ midas "I just got fired from Google. Best thing that ever happened."

╭─────────── Midas v2 Evaluation ───────────╮
│ GO — confidence 70%                        │
│                                            │
│ Majority positive: structural tier HIGH    │
│ PERFORMER, agent share rate 50%            │
╰────────────────────────────────────────────╯

$ midas optimize "I just got fired. Here's what I learned."

Iteration 0: structural 62 (HIGH PERFORMER) | agents 33% share | verdict: REVISE
Iteration 1: structural 78 (VIRAL CANDIDATE) | agents 67% share | verdict: SHIP
  → Changes: Added personal hook, power line break after opener, specific details

Done! 2 iterations, structural 62 → 78, verdict: SHIP
```

## Install

Three tiers — start light, add power when you need it:

```bash
git clone https://github.com/ajsai47/midas-v2.git
cd midas-v2

# Quick: structural scoring only (~5 seconds)
pip install -e .

# Pro: + AI agent simulation + optimizer rewrites
pip install -e ".[agents]"
export ANTHROPIC_API_KEY=your-key-here

# Full: + brain predictions + SVG renders + everything
pip install -e ".[all]"
```

Or use the guided installer:

```bash
midas setup
```

## Commands

```bash
# Score a post (structural + agents, fast)
midas "I just got fired from Google. Best thing that ever happened."

# Full evaluation with all mechanisms
midas evaluate "Your post text"

# Auto-improve until ship-ready
midas optimize "Your draft post" --loops 5

# Generate SVG brain activation map
midas render "Your post" -o brain.svg

# Score with brain model only
midas score "Your post"

# Compare A/B variants (brain model)
midas compare --text "Version A" --text "Version B"

# Explain brain region activation breakdown
midas explain "Your post"

# Generate brain heatmap PNG
midas heatmap "Your post" --output brain.png

# Guided install
midas setup
```

### Flags

| Flag | Commands | What it does |
|------|----------|-------------|
| `--no-brain` | evaluate, optimize | Skip brain model (fast mode) |
| `--no-agents` | evaluate, optimize | Skip agent simulation |
| `--svg brain.svg` | evaluate, optimize | Save SVG brain render alongside results |
| `--json` | evaluate, optimize, score | Machine-readable JSON output |
| `--config my.yaml` | evaluate, optimize | Custom structural scoring config |
| `--personas my.yaml` | evaluate, optimize | Custom agent persona definitions |
| `--loops N` | optimize | Max rewrite iterations (default 5) |

## How it works

Three independent systems must agree before your content ships:

```
                     ┌─────────────────────┐
                     │     Your Content     │
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
    ┌─────────▼──┐   ┌─────────▼──┐   ┌──────────▼──┐
    │   Brain    │   │   Agent     │   │  Structural  │
    │   Model    │   │ Simulation  │   │   Scoring    │
    │ (TRIBE v2) │   │ (personas)  │   │  (patterns)  │
    └─────────┬──┘   └─────────┬──┘   └──────────┬──┘
              │                 │                  │
              └─────────────────┼──────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │  ship / revise / kill │
                     └─────────────────────┘
```

**Brain Model** — Meta's TRIBE v2 predicts fMRI brain responses. 10 regions correlated with real engagement at p<0.01: reward circuits (OFC), memory encoding (hippocampus), social cognition (temporal pole), auditory suppression. Validation: Spearman rho = 0.75, p = 0.0001.

**Agent Simulation** — Six AI personas (VC partner, staff engineer, startup founder, tech journalist, corporate VP, content creator) predict share/comment/save/scroll_past with confidence scores. Built on TinyTroupe and action-guided generation research.

**Structural Scoring** — Config-driven pattern matching: hook type, formatting, narrative elements, CTAs. Ships with defaults; run `midas analyze` on your LinkedIn data for personalized weights.

**Optimizer** — Runs evaluate → rewrite → evaluate cycles using Claude until the post hits SHIP or plateaus. The best feature — turns any draft into a polished post.

## Python API

```python
# Quick structural scoring (no heavy deps)
from tribe_score import structural_score
result = structural_score("Your post text")
print(result.score, result.tier)  # 72, "HIGH PERFORMER"

# Full evaluation
from tribe_score import Evaluator
evaluator = Evaluator()
result = evaluator.evaluate("Your post text", skip_brain=True)
print(result.verdict)  # "ship" / "revise" / "kill"

# Optimizer loop
from tribe_score import Optimizer
optimizer = Optimizer()
result = optimizer.optimize("Your draft", max_loops=5, skip_brain=True)
print(result.final_text)
print(result.final_evaluation.verdict)
```

### Brain model (requires `[brain]` or `[all]`)

```python
from tribe_score import NeuralEngagementScorer
scorer = NeuralEngagementScorer()
result = scorer.score_text("Your post text")
print(result.nes, result.tier)  # 81.9, "NEURAL VIRAL"
print(result.group_scores)      # reward, memory, social, auditory
```

### SVG brain render (requires `[all]`)

```python
from tribe_score.brain_svg import render_brain_svg
from tribe_score import NeuralEngagementScorer

scorer = NeuralEngagementScorer()
result = scorer.score_text("Your post text")
render_brain_svg(result.raw_activation, result.nes, result.tier, "brain.svg")
```

## Configuration

### Custom structural config

Create a YAML file with your own signal weights:

```bash
midas evaluate "Your post" --config my_config.yaml
```

### Custom personas

Define your own audience personas for agent simulation:

```bash
midas evaluate "Your post" --personas my_personas.yaml
```

### Personalized config from your data

```bash
midas analyze  # generates config from your LinkedIn engagement data
```

## Brain Model

Meta's [TRIBE v2](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) predicts fMRI brain responses to any content. We ran 20 LinkedIn posts with known engagement through it and found **10 regions correlated with real-world virality at p<0.01**.

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
| 80-100 | **NEURAL VIRAL** | Strong reward + memory + social, suppressed auditory |
| 60-79 | HIGH ACTIVATION | Good engagement region activation |
| 40-59 | MODERATE | Mixed signal |
| 20-39 | LOW ACTIVATION | Weak engagement pattern |
| 0-19 | MINIMAL | No engagement signal |

### SVG brain renders

With `[all]` installed, generate shareable brain activation maps:

```bash
midas render "Your post" -o brain.svg
midas evaluate "Your post" --svg brain.svg
midas optimize "Your post" --svg optimized_brain.svg
```

The SVG shows activation overlays on the cortical surface with the 10 empirical regions annotated.

## Architecture

```
tribe_score/              # Midas v2 scoring engine
├── cli.py                # CLI with smart defaults + MidasGroup routing
├── evaluator.py          # Unified evaluator — orchestrates all 3 mechanisms
├── optimizer.py          # Evaluate → rewrite → evaluate loop
├── scorer.py             # Mechanism 1: Brain activation → engagement score
├── brain_svg.py          # SVG brain render using PlotBrainNilearn
├── compare.py            # A/B comparison tables
├── regions.py            # Empirical brain region weights + calibration
├── structural/           # Mechanism 3: Config-driven pattern scoring
│   ├── scorer.py         # Pattern matching engine
│   ├── config.py         # YAML config parser
│   └── default_config.yaml
├── agents/               # Mechanism 2: LLM persona simulation
│   ├── simulator.py      # Two-stage evaluation engine
│   └── personas.py       # 6 LinkedIn audience personas
└── __init__.py           # Lazy imports — no crash without brain deps

tribev2/                  # Meta's TRIBE v2 (upstream, 2 small CPU patches)
├── demo_utils.py         # TribeModel — predicts brain response to content
├── plotting/             # Brain visualization (nilearn + pyvista)
├── model.py              # FmriEncoder Transformer
├── utils.py              # HCP atlas, ROI analysis
└── ...
```

## Credits

Brain model built on **TRIBE v2** by Meta FAIR. Agent simulation inspired by [TinyTroupe](https://github.com/microsoft/TinyTroupe) and [action-guided generation research](https://arxiv.org/abs/2502.12073).

[Paper](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) | [Weights](https://huggingface.co/facebook/tribev2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)

```bibtex
@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brookes, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
  year={2026}
}
```

## License

CC-BY-NC-4.0. See [LICENSE](LICENSE).
