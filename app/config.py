"""App configuration constants."""

from pathlib import Path

# Dataset path
DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "mixedtype-wafer-defect-datasets" / "Wafer_Map_Datasets.npz"
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model paths
MODEL_DIR = PROJECT_ROOT / "wafer_images" / "phase2_models"

MODEL_REGISTRY = {
    "Custom CNN": {
        "path": MODEL_DIR / "model_p2_cnn.keras",
        "size": "4.3 MB",
        "params": "428K",
        "macro_f1": 0.9736,
        "weighted_f1": 0.9735,
        "description": "Lightweight 4-block CNN trained from scratch on wafer data.",
    },
    "MobileNetV2": {
        "path": MODEL_DIR / "model_p2_tl.keras",
        "size": "28 MB",
        "params": "2.4M",
        "macro_f1": 0.9667,
        "weighted_f1": 0.9658,
        "description": "MobileNetV2 backbone with transfer learning from ImageNet.",
    },
}

# Input
INPUT_SHAPE = (52, 52, 3)
NUM_CLASSES = 38

# Wafer map colors (from data_exploration.ipynb convert_to_rgb)
WAFER_RGB = {
    0: (0, 63, 92),  # blank — dark teal
    1: (255, 166, 0),  # normal die — amber
    2: (188, 80, 144),  # broken die — magenta
}

# Normalized for matplotlib
WAFER_COLORS = {k: (r / 255, g / 255, b / 255) for k, (r, g, b) in WAFER_RGB.items()}

# App defaults
TOP_N_DEFAULT = 5
MAX_BATCH_SIZE = 100

# Financial analysis defaults (leadership decision support)
FINANCIAL_CONFIG_DEFAULTS = {
    "WPH": 100,
    "VALUE_PER_WAFER": 5_000,
    "REPAIR_HOURS": 8,
    "PLANNING_HORIZON": 30,
    "CONFIDENCE_THRESHOLD": 0.70,
}

# Scenario defaults used for base-pattern-dominant demo analysis
BASE_PATTERN_SCENARIO_DEFAULTS = {
    "NORMAL_SHARE": 0.80,
    "TARGET_SHARE": 0.16,
    "RANDOM_SEED": 42,
}

# Backward-compatible alias for older references.
DONUT_SCENARIO_DEFAULTS = BASE_PATTERN_SCENARIO_DEFAULTS

# Dark theme background (matches .streamlit/config.toml)
BG_COLOR = "#0E1117"
