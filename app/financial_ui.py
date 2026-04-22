"""Financial decision-support helpers for Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from app.labels import ID_TO_PATTERN
from financial_impact import financial

# Bit order follows existing training labels: Center, Donut, Edge_Loc,
# Edge_Ring, Loc, Near_Full, Scratch, Random.
_DEFECT_TO_BIT_INDEX = {
    "Center": 0,
    "Donut": 1,
    "Edge_Loc": 2,
    "Edge_Ring": 3,
    "Loc": 4,
    "Near_Full": 5,
    "Scratch": 6,
    "Random": 7,
}
BASE_PATTERNS = tuple(_DEFECT_TO_BIT_INDEX.keys())


@dataclass(frozen=True)
class ScenarioConfig:
    normal_share: float = 0.80
    target_share: float = 0.16
    random_seed: int = 42


def format_currency(value: float) -> str:
    """Format values into readable currency units for leadership dashboards."""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def format_percent(value: float, decimals: int = 1) -> str:
    """Format percentages consistently in Streamlit metric cards."""
    return f"{float(value):.{decimals}f}%"


def pattern_name_to_binary(pattern_name: str) -> str:
    """Convert human-readable pattern names to 8-bit binary labels."""
    if pattern_name == "Normal":
        return "00000000"

    bits = ["0"] * 8
    for part in pattern_name.split("+"):
        bit_index = _DEFECT_TO_BIT_INDEX.get(part)
        if bit_index is not None:
            bits[bit_index] = "1"
    return "".join(bits)


def build_id_to_binary() -> dict[int, str]:
    """Build class-id to binary-label mapping from app label registry."""
    return {class_id: pattern_name_to_binary(name) for class_id, name in ID_TO_PATTERN.items()}


def build_label_mapping() -> dict[str, str]:
    """Build binary-label to pattern-name mapping from app label registry."""
    return {pattern_name_to_binary(name): name for name in ID_TO_PATTERN.values()}


def get_base_binary(base_pattern: str) -> str:
    """Convert a base pattern name into its one-hot 8-bit label."""
    if base_pattern not in _DEFECT_TO_BIT_INDEX:
        valid = ", ".join(BASE_PATTERNS)
        raise ValueError(f"Unknown base pattern '{base_pattern}'. Valid options: {valid}")

    bits = ["0"] * 8
    bits[_DEFECT_TO_BIT_INDEX[base_pattern]] = "1"
    return "".join(bits)


def is_pattern_related(binary_label: str, base_pattern: str) -> bool:
    """Return True when a pattern contains the selected base component."""
    if binary_label == "00000000":
        return False
    return get_base_binary(base_pattern) in financial.decompose(binary_label)


def is_pattern_name_related(pattern_name: str, base_pattern: str) -> bool:
    """Return True when a human-readable pattern contains the selected base component."""
    return is_pattern_related(pattern_name_to_binary(pattern_name), base_pattern)


def summarize_family_distribution(binary_labels: list[str], base_pattern: str) -> dict[str, int]:
    """Summarize labels into Normal/target-related/Other for scenario reporting."""
    counts = {"normal": 0, "target_related": 0, "other_defects": 0}
    for label in binary_labels:
        if label == "00000000":
            counts["normal"] += 1
        elif is_pattern_related(label, base_pattern):
            counts["target_related"] += 1
        else:
            counts["other_defects"] += 1
    return counts


def _draw_indices(rng: np.random.Generator, source: list[int], size: int, fallback: list[int]) -> list[int]:
    """Draw indices with replacement, using a fallback group when source is empty."""
    if size <= 0:
        return []

    draw_pool = source if source else fallback
    if not draw_pool:
        return []

    chosen = rng.choice(np.array(draw_pool, dtype=int), size=size, replace=True)
    return [int(i) for i in chosen]


def apply_pattern_scenario(results: list[dict], scenario: ScenarioConfig, base_pattern: str) -> list[dict]:
    """Resample predictions for a normal-majority, selected-pattern-dominant scenario."""
    if not results:
        return []

    id_to_binary = build_id_to_binary()
    normal_indices: list[int] = []
    target_indices: list[int] = []
    other_indices: list[int] = []

    for idx, row in enumerate(results):
        binary_label = id_to_binary.get(int(row["class_id"]), "00000000")
        if binary_label == "00000000":
            normal_indices.append(idx)
        elif is_pattern_related(binary_label, base_pattern):
            target_indices.append(idx)
        else:
            other_indices.append(idx)

    n_rows = len(results)
    n_normal = round(n_rows * scenario.normal_share)
    n_target = round(n_rows * scenario.target_share)
    n_other = max(0, n_rows - n_normal - n_target)

    # Keep totals exact even after rounding.
    delta = n_rows - (n_normal + n_target + n_other)
    n_normal += delta

    rng = np.random.default_rng(scenario.random_seed)
    fallback_all = list(range(n_rows))

    picked = []
    picked.extend(_draw_indices(rng, normal_indices, n_normal, fallback_all))
    picked.extend(_draw_indices(rng, target_indices, n_target, fallback_all))
    picked.extend(_draw_indices(rng, other_indices, n_other, fallback_all))

    if len(picked) < n_rows:
        picked.extend(_draw_indices(rng, fallback_all, n_rows - len(picked), fallback_all))
    picked = picked[:n_rows]
    rng.shuffle(picked)

    scenario_rows = []
    for out_idx, base_idx in enumerate(picked):
        row = dict(results[base_idx])
        row["index"] = out_idx
        scenario_rows.append(row)

    return scenario_rows


def compute_pattern_metrics(df_financial: pd.DataFrame, base_pattern: str) -> dict[str, float]:
    """Aggregate selected-pattern financial metrics for leadership insights."""
    if df_financial.empty:
        return {
            "count": 0,
            "batch_pct": 0.0,
            "daily_loss": 0.0,
            "avg_confidence": 0.0,
            "priority_score": 0.0,
        }

    mask = df_financial["binary_label"].map(lambda binary_label: is_pattern_related(binary_label, base_pattern))
    df_target = df_financial.loc[mask]

    if df_target.empty:
        return {
            "count": 0,
            "batch_pct": 0.0,
            "daily_loss": 0.0,
            "avg_confidence": 0.0,
            "priority_score": 0.0,
        }

    return {
        "count": int(df_target["count"].sum()),
        "batch_pct": float(df_target["batch_pct"].sum()),
        "daily_loss": float(df_target["weighted_daily_loss"].sum()),
        "avg_confidence": float(df_target["avg_confidence"].mean()),
        "priority_score": float(df_target["priority_score"].sum()),
    }


def build_executive_recommendation(
    df_actions: pd.DataFrame, pattern_metrics: dict[str, float], base_pattern: str
) -> str:
    """Build a concise immediate recommendation from action and selected-pattern metrics."""
    if df_actions.empty:
        return "No immediate corrective action required. Continue monitoring wafer quality and confidence trends."

    top = df_actions.iloc[0]
    pattern_note = ""
    if pattern_metrics["count"] > 0:
        pattern_note = (
            f" {base_pattern}-related defects account for {format_percent(pattern_metrics['batch_pct'])} "
            f"of the batch with {format_currency(pattern_metrics['daily_loss'])} daily impact."
        )

    return (
        f"Prioritize '{top['repair_action']}' in {top['process_step']} to recover "
        f"{format_currency(top['daily_loss_savings'])} per day and break even in "
        f"{top['break_even_days']:.1f} days.{pattern_note}"
    )


def _results_to_arrays(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract class IDs and confidences from streamlit prediction payloads."""
    class_ids = np.array([int(row["class_id"]) for row in results], dtype=int)
    confidences = np.array([float(row["confidence"]) for row in results], dtype=float)
    return class_ids, confidences


def analyze_results(
    results: list[dict],
    config: dict,
    confidence_threshold: float,
    use_pattern_scenario: bool = False,
    focus_pattern: str = "Donut",
    scenario: ScenarioConfig | None = None,
) -> dict:
    """Run financial analysis for prediction rows and return leadership-ready payloads."""
    if not results:
        empty = pd.DataFrame()
        return {
            "results": [],
            "df_batch": empty,
            "df_financial": empty,
            "df_anomaly": empty,
            "df_actions": empty,
            "summary_payload": {},
            "pattern_metrics": {
                "count": 0,
                "batch_pct": 0.0,
                "daily_loss": 0.0,
                "avg_confidence": 0.0,
                "priority_score": 0.0,
            },
            "executive_recommendation": "",
            "family_distribution": {"normal": 0, "target_related": 0, "other_defects": 0},
        }

    scenario_cfg = scenario or ScenarioConfig()
    analysis_results = (
        apply_pattern_scenario(results, scenario_cfg, focus_pattern) if use_pattern_scenario else list(results)
    )

    class_ids, confidences = _results_to_arrays(analysis_results)
    id_to_binary = build_id_to_binary()
    label_mapping = build_label_mapping()
    binary_labels, _ = financial.decode_labels(class_ids, id_to_binary, ID_TO_PATTERN)

    df_batch = financial.build_batch_df(binary_labels, confidences, label_mapping)
    df_anomaly = financial.build_base_anomaly_df(binary_labels, confidences, label_mapping)

    effective_config = dict(config)
    effective_config["BATCH_ID"] = config.get("BATCH_ID") or datetime.now().strftime("BATCH_%Y%m%d_%H%M%S")
    effective_config["total_wafers"] = len(analysis_results)
    effective_config["low_conf_count"] = int((confidences < confidence_threshold).sum())
    effective_config["CONFIDENCE_THRESHOLD"] = confidence_threshold

    df_financial, summary_payload = financial.compute_financials(df_batch, effective_config)
    df_actions = financial.compute_action_table(df_financial, config=effective_config)
    pattern_metrics = compute_pattern_metrics(df_financial, focus_pattern)
    recommendation = build_executive_recommendation(df_actions, pattern_metrics, focus_pattern)

    return {
        "results": analysis_results,
        "df_batch": df_batch,
        "df_financial": df_financial,
        "df_anomaly": df_anomaly,
        "df_actions": df_actions,
        "summary_payload": summary_payload,
        "pattern_metrics": pattern_metrics,
        "executive_recommendation": recommendation,
        "family_distribution": summarize_family_distribution(binary_labels, focus_pattern),
    }
