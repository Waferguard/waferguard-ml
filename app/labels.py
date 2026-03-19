"""Defect pattern labels and descriptions."""

ID_TO_PATTERN: dict[int, str] = {
    0: "Normal",
    1: "Random",
    2: "Scratch",
    3: "Near_Full",
    4: "Loc",
    5: "Loc+Scratch",
    6: "Edge_Ring",
    7: "Edge_Ring+Scratch",
    8: "Edge_Ring+Loc",
    9: "Edge_Ring+Loc+Scratch",
    10: "Edge_Loc",
    11: "Edge_Loc+Scratch",
    12: "Edge_Loc+Loc",
    13: "Edge_Loc+Loc+Scratch",
    14: "Donut",
    15: "Donut+Scratch",
    16: "Donut+Loc",
    17: "Donut+Loc+Scratch",
    18: "Donut+Edge_Ring",
    19: "Donut+Edge_Ring+Scratch",
    20: "Donut+Edge_Ring+Loc",
    21: "Donut+Edge_Ring+Loc+Scratch",
    22: "Donut+Edge_Loc",
    23: "Donut+Edge_Loc+Scratch",
    24: "Donut+Edge_Loc+Loc",
    25: "Donut+Edge_Loc+Loc+Scratch",
    26: "Center",
    27: "Center+Scratch",
    28: "Center+Loc",
    29: "Center+Loc+Scratch",
    30: "Center+Edge_Ring",
    31: "Center+Edge_Ring+Scratch",
    32: "Center+Edge_Ring+Loc",
    33: "Center+Edge_Ring+Loc+Scratch",
    34: "Center+Edge_Loc",
    35: "Center+Edge_Loc+Scratch",
    36: "Center+Edge_Loc+Loc",
    37: "Center+Edge_Loc+Loc+Scratch",
}

PATTERN_TO_ID: dict[str, int] = {v: k for k, v in ID_TO_PATTERN.items()}

BASE_DEFECT_DESCRIPTIONS: dict[str, str] = {
    "Normal": "No defect pattern detected. The wafer passed quality checks.",
    "Center": (
        "Defects concentrated in the center of the wafer, often caused by "
        "process uniformity issues in CMP or etch steps."
    ),
    "Donut": (
        "Ring-shaped defect pattern surrounding the center, typically linked "
        "to gas flow or temperature gradients during deposition."
    ),
    "Edge_Loc": (
        "Defects localized at the wafer edge, commonly caused by edge bead removal issues or handling damage."
    ),
    "Edge_Ring": (
        "Defects forming a ring along the wafer periphery, often from "
        "edge-specific process variations like spin coating."
    ),
    "Loc": (
        "Localized cluster of defects in a specific region, potentially "
        "caused by particle contamination or tool-specific issues."
    ),
    "Near_Full": ("Defects spread across nearly the entire wafer surface, indicating systemic process failure."),
    "Scratch": (
        "Linear scratch patterns across the wafer, typically from mechanical "
        "handling, CMP pad issues, or robotic arm contact."
    ),
    "Random": ("Randomly distributed defects with no spatial pattern, often from airborne particle contamination."),
}


def get_description(pattern_name: str) -> str:
    """Get a description for a defect pattern, including combinations."""
    if pattern_name in BASE_DEFECT_DESCRIPTIONS:
        return BASE_DEFECT_DESCRIPTIONS[pattern_name]

    parts = pattern_name.split("+")
    descriptions = []
    for part in parts:
        if part in BASE_DEFECT_DESCRIPTIONS:
            descriptions.append(f"**{part}**: {BASE_DEFECT_DESCRIPTIONS[part]}")
    return "\n\n".join(descriptions) if descriptions else f"Combined defect pattern: {pattern_name}"
