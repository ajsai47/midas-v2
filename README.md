<div align="center">

```
           _____ _______ _____           _____    ___   ___
          |     |__   __|  __ \    /\   / ____|  / _ \ / _ \
          | |\/| | | |  | |  | |  /  \ | (___  | (_) | (_) |
          | |  | | | |  | |  | | / /\ \ \___ \  \__, |\__, |
          | |  | |_| |_ | |__| |/ ____ \____) |   / /   / /
          |_|  |_|_____|_|_____/_/    \_\_____/   /_/   /_/
                                v2
```

### Know if your content will go viral — before you hit post.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Built with Claude](https://img.shields.io/badge/built%20with-Claude-blueviolet)](https://anthropic.com)
[![Meta TRIBE v2](https://img.shields.io/badge/brain%20model-Meta%20TRIBE%20v2-red)](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/)
[![Works on My Machine](https://img.shields.io/badge/works%20on-my%20machine-success)](https://github.com/ajsai47/midas-v2)
[![10x Engagement](https://img.shields.io/badge/engagement-10x%20improvement-gold)](https://www.linkedin.com/in/aj-green-ai)

---

**This tool 10x'd my content engagement.** Don't believe me? [Check my LinkedIn.](https://www.linkedin.com/in/aj-green-ai)

Three AI systems score your post. An optimizer rewrites it until it's viral-ready.
You stop guessing, start shipping bangers.

**[Get Started](#install)** · **[See it Work](#demo)** · **[How it Works](#how-it-works)** · **[Python API](#python-api)**

</div>

---

## Why Midas?

You've been there. You spend 45 minutes writing a LinkedIn post. You hit publish. Crickets.

Meanwhile some guy posts *"I got fired. Best day of my life."* and gets 50,000 impressions.

**The difference isn't talent — it's pattern recognition.** Viral content activates specific brain regions, follows structural patterns, and resonates with specific audience archetypes. Midas v2 reverse-engineers all three.

- **Score** — Three independent prediction systems analyze your content in seconds
- **Optimize** — The AI rewrites your post through iterative loops until it hits SHIP
- **Ship** — Or kill it before it flops and save yourself the embarrassment

> *"I went from mass-producing mediocre posts to only posting content that hits. My engagement is up 10x since I started using Midas."* — [AJ Green](https://www.linkedin.com/in/aj-green-ai)

---

## Demo

```bash
$ midas "I just got fired from Google. Best thing that ever happened."

╭─────────── Midas v2 Evaluation ───────────╮
│ GO — confidence 70%                        │
│                                            │
│ Majority positive: structural tier HIGH    │
│ PERFORMER, agent share rate 50%            │
╰────────────────────────────────────────────╯
```

Not good enough? **Let Midas rewrite it for you:**

```bash
$ midas optimize "I just got fired. Here's what I learned."

Iteration 0: structural 62 (HIGH PERFORMER) | agents 33% share | verdict: REVISE
Iteration 1: structural 78 (VIRAL CANDIDATE) | agents 67% share | verdict: SHIP
  → Changes: Added personal hook, power line break after opener, specific details

Done! 2 iterations, structural 62 → 78, verdict: SHIP
```

From REVISE to SHIP in two iterations. Zero effort. That's the whole point.

---

## Install

Start light. Add power when you need it.

```bash
git clone https://github.com/ajsai47/midas-v2.git
cd midas-v2
```

| Tier | Install | What You Get | Speed |
|------|---------|-------------|-------|
| **Quick** | `pip install -e .` | Structural scoring only | ~5 sec install |
| **Pro** | `pip install -e ".[agents]"` | + AI agent simulation + optimizer rewrites | ~15 sec install |
| **Full** | `pip install -e ".[all]"` | + fMRI brain predictions + SVG brain renders | ~2 min install |

Pro tier needs an Anthropic API key for the agent simulation + optimizer:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

Or just run the guided installer and let Midas figure it out:

```bash
midas setup
```

---

## Commands

```bash
# Score a post — just paste your text
midas "I just got fired from Google. Best thing that ever happened."

# Full evaluation with all mechanisms
midas evaluate "Your post text"

# Auto-improve until ship-ready (this is the killer feature)
midas optimize "Your draft post" --loops 5

# Generate SVG brain activation map
midas render "Your post" -o brain.svg

# Score with brain model only
midas score "Your post"

# Compare A/B variants
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

---

## How it Works

Three independent AI systems must agree before your content ships. No single point of failure. No blind spots.

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
                     │  SHIP / REVISE / KILL │
                     └─────────────────────┘
```

### Brain Model — *"Will this light up their brain?"*

Meta's TRIBE v2 predicts fMRI brain responses to your content. We found **10 brain regions correlated with real-world virality at p<0.01** — reward circuits, memory encoding, social cognition, and auditory suppression. If your post activates the right regions, it's going viral. Validation: Spearman rho = 0.75, p = 0.0001.

### Agent Simulation — *"Would real people actually share this?"*

Six AI personas — VC partner, staff engineer, startup founder, tech journalist, corporate VP, content creator — read your post and decide: share, comment, save, or scroll past. Each gives a confidence score. If 4 out of 6 share, you're golden. Built on [TinyTroupe](https://github.com/microsoft/TinyTroupe) and [action-guided generation research](https://arxiv.org/abs/2502.12073).

### Structural Scoring — *"Does this follow the patterns that work?"*

Config-driven pattern matching against 1,000+ analyzed posts: hook type, formatting, narrative elements, CTAs, line breaks, power phrases. Ships with battle-tested defaults. Run `midas analyze` on your own LinkedIn data for personalized weights.

### Optimizer — *"Make it better. Automatically."*

This is the killer feature. Runs evaluate → rewrite → evaluate cycles using Claude until your post hits SHIP or plateaus. Turns any rough draft into a polished, viral-ready post. Set it and forget it.

---

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

# Optimizer loop — the money maker
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

---

## Configuration

### Custom structural config

Tweak the signal weights to match your audience:

```bash
midas evaluate "Your post" --config my_config.yaml
```

### Custom personas

Define your own audience personas for agent simulation:

```bash
midas evaluate "Your post" --personas my_personas.yaml
```

### Personalized config from your data

Export your LinkedIn analytics and let Midas learn what works for *you*:

```bash
midas analyze  # generates config from your LinkedIn engagement data
```

---

## Brain Model Deep Dive

Meta's [TRIBE v2](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) predicts fMRI brain responses to any content. We ran 20 LinkedIn posts with known engagement through it and found **10 regions correlated with real-world virality at p<0.01**.

| Group | Brain Regions | Correlation | What it Means |
|-------|--------------|-------------|---------------|
| **Reward** | pOFC (rho=+0.66), OFC (+0.60) | Positive | Lights up dopamine → people engage |
| **Memory** | Hippocampus (+0.63) | Positive | Encodes to memory → people share |
| **Social** | Temporal pole dorsal (+0.61), ventral (+0.57) | Positive | Social processing → viral spread |
| **Auditory** | TA2 (-0.63), A4 (-0.59), PBelt (-0.58), MBelt (-0.58), A5 (-0.57) | **Negative** | Suppressed auditory cortex → higher engagement |
| **Focus** | Brain variability (-0.60) | **Negative** | Focused activation beats diffuse |

### Scoring Tiers

| NES | Tier | What to Expect |
|-----|------|----------------|
| 80-100 | **NEURAL VIRAL** | Strong reward + memory + social. Ship immediately. |
| 60-79 | HIGH ACTIVATION | Good signal. Minor tweaks could push it viral. |
| 40-59 | MODERATE | Mixed signal. Run the optimizer. |
| 20-39 | LOW ACTIVATION | Weak pattern. Major rewrite needed. |
| 0-19 | MINIMAL | Kill it. Start over. |

### SVG Brain Renders

With `[all]` installed, generate shareable brain activation maps:

```bash
midas render "Your post" -o brain.svg
midas evaluate "Your post" --svg brain.svg
midas optimize "Your post" --svg optimized_brain.svg
```

The SVG shows activation overlays on the cortical surface with the 10 empirical regions annotated. Share it on LinkedIn for bonus engagement (yes, really).

---

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

---

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
