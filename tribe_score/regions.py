"""Empirically-validated brain region weights for LinkedIn engagement prediction.

Derived from region hunt analysis (n=20 posts, Spearman correlation).
Only regions with p<0.01 are used — strongest signal, least overfitting risk.

Region names are HCP MMP1 atlas names compatible with tribev2.utils functions.
"""

# Regions empirically correlated with LinkedIn engagement (p < 0.01).
# Key: HCP region name, Value: Spearman rho with engagement.
# Positive rho = higher activation → higher engagement.
# Negative rho = lower activation → higher engagement.
EMPIRICAL_REGIONS: dict[str, float] = {
    # Reward / value judgment
    "pOFC": +0.6559,   # Posterior orbitofrontal cortex (p=0.0017)
    "OFC":  +0.6010,   # Orbitofrontal cortex (p=0.0051)
    # Memory
    "H":    +0.6266,   # Hippocampus (p=0.0031)
    # Social / emotional processing
    "TGd":  +0.6100,   # Temporal pole dorsal (p=0.0043)
    "TGv":  +0.5724,   # Temporal pole ventral (p=0.0084)
    # Auditory cortex (SUPPRESSED in viral content)
    "TA2":  -0.6303,   # Auditory association (p=0.0029)
    "A4":   -0.5867,   # Primary auditory (p=0.0065)
    "PBelt": -0.5792,  # Parabelt (p=0.0075)
    "MBelt": -0.5777,  # Medial belt (p=0.0076)
    "A5":   -0.5739,   # Auditory area 5 (p=0.0081)
}

# Brain variability is negatively correlated: focused > diffuse
VARIABILITY_RHO: float = -0.5995  # rho=-0.60, p=0.0052

# Calibration stats from 20-post region hunt (mean, std per region).
# Used to z-score new activations before applying weights.
CALIBRATION: dict[str, tuple[float, float]] = {
    "pOFC":  (-0.011540, 0.013402),
    "TA2":   ( 0.223245, 0.078142),
    "H":     (-0.041178, 0.017342),
    "TGd":   (-0.086515, 0.042735),
    "OFC":   (-0.006283, 0.005880),
    "A4":    ( 0.294178, 0.114398),
    "PBelt": ( 0.280630, 0.107057),
    "MBelt": ( 0.169161, 0.060954),
    "A5":    ( 0.294577, 0.133120),
    "TGv":   (-0.079115, 0.031443),
    "_variability": (0.111894, 0.030000),  # brain_variability mean/std estimate
}

# Raw score range from calibration data (computed below, updated at import).
# Used for linear normalization to 0-100.
CALIBRATION_RAW_RANGE: tuple[float, float] = (-9.0, 9.0)  # from 20-post calibration

# Human-readable grouping for display purposes
REGION_GROUPS: dict[str, list[str]] = {
    "Reward":    ["pOFC", "OFC"],
    "Memory":    ["H"],
    "Social":    ["TGd", "TGv"],
    "Auditory (suppressed)": ["TA2", "A4", "PBelt", "MBelt", "A5"],
}

# NES tier thresholds — recalibrated for empirical scoring
TIERS: list[tuple[float, str, str]] = [
    (80, "NEURAL VIRAL", "Strong reward + memory + social activation, suppressed auditory"),
    (60, "HIGH ACTIVATION", "Good engagement region activation pattern"),
    (40, "MODERATE", "Mixed signal across engagement regions"),
    (20, "LOW ACTIVATION", "Weak engagement pattern"),
    (0,  "MINIMAL", "No engagement signal detected"),
]


def get_tier(score: float) -> tuple[str, str]:
    """Return (tier_name, tier_description) for a given NES score."""
    for threshold, name, desc in TIERS:
        if score >= threshold:
            return name, desc
    return TIERS[-1][1], TIERS[-1][2]
