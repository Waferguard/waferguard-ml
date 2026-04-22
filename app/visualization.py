"""Visualization: wafer maps and confidence charts."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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
    colors = ["#92d400" if i == 0 else "#00a1de" for i in range(top_n)]

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


def _format_currency(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def render_kpi_cards(summary_payload: dict) -> None:
    """Render leadership KPI cards for financial decision support."""
    total_wafers = int(summary_payload.get("total_wafers", 0))
    low_conf_count = int(summary_payload.get("low_conf_count", 0))
    low_conf_share = (low_conf_count / total_wafers * 100) if total_wafers > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily Loss", _format_currency(summary_payload.get("total_daily_loss", 0.0)))
    c2.metric("Defect Rate", f"{summary_payload.get('defect_rate', 0.0) * 100:.1f}%")
    c3.metric("Avg Confidence", f"{summary_payload.get('avg_confidence', 0.0) * 100:.1f}%")
    c4.metric("Low Confidence", f"{low_conf_count} ({low_conf_share:.1f}%)")


def render_pattern_card(base_pattern: str, pattern_metrics: dict[str, float]) -> None:
    """Render selected base-pattern insights with dedicated callout metrics."""
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{base_pattern}-related Count", int(pattern_metrics.get("count", 0)))
    c2.metric(f"{base_pattern}-related Share", f"{pattern_metrics.get('batch_pct', 0.0):.1f}%")
    c3.metric(f"{base_pattern} Daily Loss", _format_currency(pattern_metrics.get("daily_loss", 0.0)))


def render_donut_card(donut_metrics: dict[str, float]) -> None:
    """Backward-compatible wrapper for existing calls."""
    render_pattern_card("Donut", donut_metrics)


def render_action_table(df_actions: pd.DataFrame, top_n: int = 5) -> None:
    """Render prioritized repair actions for leadership."""
    if df_actions.empty:
        st.info("No action items available for this selection.")
        return

    show = df_actions.head(top_n)[
        [
            "repair_action",
            "process_step",
            "risk_level",
            "daily_loss_savings",
            "break_even_days",
            "evoa_30d",
        ]
    ].rename(
        columns={
            "repair_action": "Action",
            "process_step": "Process Step",
            "risk_level": "Risk",
            "daily_loss_savings": "Daily Savings",
            "break_even_days": "Break-even (days)",
            "evoa_30d": "30d EVoA",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)


def render_base_anomaly_chart(df_anomaly: pd.DataFrame) -> None:
    """Render base defect prevalence for fast operational review."""
    if df_anomaly.empty:
        st.info("No base anomaly data available.")
        return

    chart_df = df_anomaly.sort_values("count", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bars = ax.barh(
        chart_df["pattern_name"].iloc[::-1],
        chart_df["count"].iloc[::-1],
        color="#00a1de",
        edgecolor="#1d3f58",
    )
    ax.set_xlabel("Wafer Count", color="white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", color="white", fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
