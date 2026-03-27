"""Brain region -> engagement dimension mapping using HCP MMP1 atlas names.

Region names/patterns are compatible with tribev2.utils.get_hcp_roi_indices(),
which supports exact names and prefix wildcards (e.g. "STS*").
"""

# Each dimension maps to a list of HCP MMP1 region names/patterns.
# tribev2's get_hcp_roi_indices() resolves these to vertex indices.
ENGAGEMENT_REGIONS: dict[str, list[str]] = {
    # Emotional salience -> shares/comments
    # Anterior Insula, ACC subregions, temporal pole
    "emotional_arousal": [
        "AAIC",      # Anterior Agranular Insular Complex
        "p24",       # Posterior cingulate area 24
        "33pr",      # Area 33 prime
        "a24",       # Anterior cingulate area 24
        "TGd",       # Temporal pole (dorsal)
        "TGv",       # Temporal pole (ventral)
    ],
    # "I want more" -> saves/follows
    # Orbitofrontal cortex, ventromedial prefrontal cortex
    "reward_motivation": [
        "OFC",       # Orbitofrontal cortex
        "pOFC",      # Posterior orbitofrontal cortex
        "10r",       # Area 10r (frontopolar)
        "10v",       # Area 10v (ventral frontopolar)
        "25",        # Subgenual area 25
        "s32",       # Subgenual area 32
    ],
    # Must grab attention to get any engagement
    # Frontal eye fields, DLPFC, visual cortex, intraparietal sulcus
    "attention_capture": [
        "FEF",       # Frontal eye fields
        "8C",        # Area 8C (DLPFC)
        "46",        # Area 46 (DLPFC)
        "V1",        # Primary visual cortex
        "V2",        # Visual area 2
        "V3",        # Visual area 3
        "V4",        # Visual area 4
        "IP0",       # Intraparietal area 0
        "IP1",       # Intraparietal area 1
        "IP2",       # Intraparietal area 2
    ],
    # Social content goes viral
    # Superior temporal sulcus, temporo-parietal junction, fusiform
    "social_cognition": [
        "STS*",      # Superior temporal sulcus (all subregions)
        "TPOJ1",     # Temporo-parietal-occipital junction 1
        "TPOJ2",     # Temporo-parietal-occipital junction 2
        "TPOJ3",     # Temporo-parietal-occipital junction 3
        "FFC",       # Fusiform face complex
    ],
    # Memorable = shareable
    # Parahippocampal, retrosplenial cortex (cortical memory correlates)
    "memory_encoding": [
        "PreS",      # Presubiculum
        "PHA1",      # Parahippocampal area 1
        "PHA2",      # Parahippocampal area 2
        "PHA3",      # Parahippocampal area 3
        "RSC",       # Retrosplenial cortex
    ],
}

# Default dimension weights (sum to 1.0).
# These are initial estimates; calibrate against real engagement data.
DEFAULT_WEIGHTS: dict[str, float] = {
    "emotional_arousal": 0.25,
    "reward_motivation": 0.25,
    "attention_capture": 0.20,
    "social_cognition": 0.20,
    "memory_encoding": 0.10,
}

# NES tier thresholds and labels
TIERS: list[tuple[float, str, str]] = [
    (85, "NEURAL VIRAL", "Extreme activation across engagement dimensions"),
    (70, "HIGH ACTIVATION", "Strong neural engagement signal"),
    (55, "MODERATE ACTIVATION", "Decent neural response, room to optimize"),
    (40, "LOW ACTIVATION", "Weak neural engagement predicted"),
    (0, "MINIMAL ACTIVATION", "Very low predicted brain response"),
]


def get_tier(score: float) -> tuple[str, str]:
    """Return (tier_name, tier_description) for a given NES score."""
    for threshold, name, desc in TIERS:
        if score >= threshold:
            return name, desc
    return TIERS[-1][1], TIERS[-1][2]
