"""Visualization: wafer maps and confidence charts."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from app.config import BG_COLOR, WAFER_COLORS
from app.labels import ID_TO_PATTERN

# Colormap for wafer map rendering
_CMAP = mcolors.ListedColormap([WAFER_COLORS[0], WAFER_COLORS[1], WAFER_COLORS[2]])
_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _CMAP.N)


def render_wafer_map(raw_array: np.ndarray) -> plt.Figure:
    """Render a 52x52 wafer map with color-coded pixel states.

    Args:
        raw_array: shape (52, 52) int array with values in {0, 1, 2}

    """
    fig, ax = plt.subplots(figsize=(4, 4), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.imshow(raw_array, cmap=_CMAP, norm=_NORM, interpolation="nearest")
    ax.axis("off")

    legend_elements = [
        Patch(facecolor=WAFER_COLORS[0], label="Blank"),
        Patch(facecolor=WAFER_COLORS[1], label="Normal Die"),
        Patch(facecolor=WAFER_COLORS[2], label="Broken Die"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=7,
        facecolor="#262730",
        edgecolor="#444",
        labelcolor="white",
    )
    fig.tight_layout(pad=0.5)
    return fig


def render_confidence_chart(probabilities: np.ndarray, top_n: int = 5) -> plt.Figure:
    """Render a horizontal bar chart of top-N predicted classes."""
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    top_names = [ID_TO_PATTERN[i] for i in top_indices]
    top_probs = probabilities[top_indices]

    fig, ax = plt.subplots(figsize=(7, max(2, top_n * 0.45)), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Top prediction in green, rest in blue
    colors = ["#4CAF50" if i == 0 else "#2196F3" for i in range(top_n)]

    # Plot in reverse so highest is at top
    bars = ax.barh(range(top_n), top_probs[::-1], color=colors[::-1], height=0.6)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], color="white", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")

    # Percentage labels on bars
    for bar, prob in zip(bars, top_probs[::-1], strict=False):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            color="white",
            fontsize=8,
        )

    fig.tight_layout()
    return fig


def build_results_dataframe(results: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from batch prediction results."""
    return pd.DataFrame([
        {
            "Wafer #": r["index"] + 1,
            "Predicted Pattern": r["pattern_name"],
            "Confidence": f"{r['confidence']:.1%}",
            "Class ID": r["class_id"],
        }
        for r in results
    ])
