"""Financial impact library extracted from waferguard_financial_impact.ipynb.

Provides pure financial analysis functions: merge predictions, decode labels,
build batch summaries, compute financial metrics, and save reports.

Fully decoupled from inference and artifact loading (Streamlit-compatible).
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

# --- Financial mapping (copied from notebook) ---------------------------------
FINANCIAL_PARAMS: dict[str, dict] = {
    "00000000": {
        "repair": 0,
        "replace": 0,
        "dt": 0,
        "ylo": 0.00,
        "yhi": 0.00,
        "priority": 5,
        "risk": "Normal",
        "process": "N/A",
        "action": "No action required — wafer passes electrical test",
    },
    "10000000": {
        "repair": 275_000,
        "replace": 6_000_000,
        "dt": 2_050_000,
        "ylo": 0.10,
        "yhi": 0.40,
        "priority": 3,
        "risk": "High",
        "process": "Deposition / Plasma Process",
        "action": "Chamber clean & kit replace; RF/temp recalibration; requalification metrology",
    },
    "01000000": {
        "repair": 130_000,
        "replace": 5_250_000,
        "dt": 550_000,
        "ylo": 0.15,
        "yhi": 0.50,
        "priority": 4,
        "risk": "High",
        "process": "CMP (primary) / Lithography Track (secondary)",
        "action": "CMP: replace pad & slurry, clean heads; Litho track: chemistry dump/refresh",
    },
    "00100000": {
        "repair": 110_000,
        "replace": 6_000_000,
        "dt": 550_000,
        "ylo": 0.01,
        "yhi": 0.10,
        "priority": 5,
        "risk": "Medium",
        "process": "Diffusion / Thermal Processing",
        "action": "Furnace zone temp recalibration; thermocouple/lamp replacement; requalification",
    },
    "00010000": {
        "repair": 255_000,
        "replace": 12_500_000,
        "dt": 2_250_000,
        "ylo": 0.00,
        "yhi": 0.15,
        "priority": 3,
        "risk": "High",
        "process": "Plasma Etch",
        "action": "Replace focus ring / edge hardware; chamber clean; RF recalibration",
    },
    "00001000": {
        "repair": 155_000,
        "replace": 8_500_000,
        "dt": 2_050_000,
        "ylo": 0.01,
        "yhi": 0.10,
        "priority": 5,
        "risk": "Medium",
        "process": "Tool-specific (single chamber / handling subsystem)",
        "action": "Robot vibration check; chamber wall inspection; targeted chamber clean",
    },
    "00000100": {
        "repair": 525_000,
        "replace": 8_500_000,
        "dt": 2_450_000,
        "ylo": 0.50,
        "yhi": 1.00,
        "priority": 1,
        "risk": "Critical",
        "process": "Major Process Excursion — Containment First",
        "action": "IMMEDIATE: lot hold + tool stop; engineering investigation; full requalification",
    },
    "00000010": {
        "repair": 77_500,
        "replace": 5_250_000,
        "dt": 550_000,
        "ylo": 0.05,
        "yhi": 0.50,
        "priority": 2,
        "risk": "Critical",
        "process": "Wafer Handling Path / CMP Contact",
        "action": "Contain handling path; swap/clean robot end effectors; inspect FOUPs/cassettes",
    },
    "00000001": {
        "repair": 130_000,
        "replace": None,
        "dt": 2_050_000,
        "ylo": 0.01,
        "yhi": 0.20,
        "priority": 5,
        "risk": "Medium",
        "process": "Environment / Broad Contamination Control",
        "action": "FOUP/carrier cleaning; filter inspection; load lock particle monitoring",
    },
}

BIT_TO_SINGLE = {
    7: "10000000",
    6: "01000000",
    5: "00100000",
    4: "00010000",
    3: "00001000",
    2: "00000100",
    1: "00000010",
    0: "00000001",
}


def merge_predictions(pred_cnn: np.ndarray, pred_tl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge two model prediction matrices by taking higher-confidence per sample.

    Returns (class_ids, confidences, use_cnn_mask).
    """
    cnn_class_ids = pred_cnn.argmax(axis=1)
    tl_class_ids = pred_tl.argmax(axis=1)
    cnn_confidences = pred_cnn.max(axis=1)
    tl_confidences = pred_tl.max(axis=1)
    use_cnn = cnn_confidences >= tl_confidences
    class_ids = np.where(use_cnn, cnn_class_ids, tl_class_ids)
    confidences = np.where(use_cnn, cnn_confidences, tl_confidences)
    return class_ids, confidences, use_cnn


def decode_labels(
    class_ids: np.ndarray, id_to_binary: dict[int, str], id_to_pattern: dict[int, str]
) -> tuple[list[str], list[str]]:
    """Decode integer class IDs to binary label strings and human pattern names."""
    binary_labels = [id_to_binary.get(int(cid), "00000000") for cid in class_ids]
    pattern_names = [id_to_pattern.get(int(cid), "Unknown") for cid in class_ids]
    return binary_labels, pattern_names


def build_batch_df(binary_labels: list[str], confidences: np.ndarray, label_mapping: dict[str, str]) -> pd.DataFrame:
    """Turn predictions into a batch summary DataFrame (same schema as notebook).

    Columns: binary_label, pattern_name, count, batch_freq, batch_pct,
    avg_confidence, min_confidence, max_confidence
    """
    total_wafers = len(binary_labels)
    pattern_counts = Counter(binary_labels)
    batch_rows = []
    for binary_label, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        name = label_mapping.get(binary_label, "Unknown")
        freq = count / total_wafers if total_wafers > 0 else 0.0
        confs_this = [float(confidences[i]) for i, lbl in enumerate(binary_labels) if lbl == binary_label]
        batch_rows.append({
            "binary_label": binary_label,
            "pattern_name": name,
            "count": count,
            "batch_freq": round(freq, 6),
            "batch_pct": round(freq * 100, 2),
            "avg_confidence": round(float(np.mean(confs_this)) if confs_this else 0.0, 4),
            "min_confidence": round(float(np.min(confs_this)) if confs_this else 0.0, 4),
            "max_confidence": round(float(np.max(confs_this)) if confs_this else 0.0, 4),
        })
    df_batch = pd.DataFrame(batch_rows)
    return df_batch


def decompose(binary_label: str) -> list[str]:
    """Return list of single-defect binary labels present in a combo."""
    return [
        BIT_TO_SINGLE[bit]
        for bit in range(7, -1, -1)
        if binary_label[7 - bit] == "1" and BIT_TO_SINGLE[bit] in FINANCIAL_PARAMS
    ]


def get_params(binary_label: str) -> dict:
    """Look up financial params for any pattern. Aggregates multi-defect combos."""
    if binary_label in FINANCIAL_PARAMS:
        return FINANCIAL_PARAMS[binary_label]
    components = decompose(binary_label)
    if not components:
        return FINANCIAL_PARAMS["00000000"]
    ds = [FINANCIAL_PARAMS[c] for c in components]
    repair = sum(d["repair"] for d in ds)
    replaces = [d["replace"] for d in ds if d.get("replace")]
    replace = sum(replaces) if replaces else None
    dt = max(d["dt"] for d in ds)
    ylos = sorted([d["ylo"] for d in ds], reverse=True)
    yhis = sorted([d["yhi"] for d in ds], reverse=True)
    ylo = round(ylos[0] + 0.5 * sum(ylos[1:]), 3)
    yhi = min(round(yhis[0] + 0.5 * sum(yhis[1:]), 3), 1.0)
    priority = min(d["priority"] for d in ds)
    risks = [d["risk"] for d in ds]
    risk = "Critical" if "Critical" in risks else "High" if "High" in risks else "Medium"
    process = " + ".join(dict.fromkeys(d["process"].split(" (")[0].split(" /")[0] for d in ds))
    action = f"Address priority-{priority} component first. " + "; then ".join(
        d["action"].split(";")[0] for d in sorted(ds, key=lambda x: x["priority"])
    )
    return {
        "repair": repair,
        "replace": replace,
        "dt": dt,
        "ylo": ylo,
        "yhi": yhi,
        "priority": priority,
        "risk": risk,
        "process": process,
        "action": action,
    }


def compute_financials(
    df_batch: pd.DataFrame, config: dict, financial_params: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Compute financial metrics for each pattern and return DataFrame + summary payload.

    Config keys needed: WPH, VALUE_PER_WAFER, REPAIR_HOURS, PLANNING_HORIZON, BATCH_ID, CONFIDENCE_THRESHOLD
    """
    if financial_params is None:
        financial_params = FINANCIAL_PARAMS
    rows = []
    for _, batch_row in df_batch.iterrows():
        binary_label = batch_row["binary_label"]
        params = get_params(binary_label)
        y_mid = (params["ylo"] + params["yhi"]) / 2
        freq = float(batch_row.get("batch_freq", 0.0))
        wph = config.get("WPH", 100)
        value_per_wafer = config.get("VALUE_PER_WAFER", 5_000)
        planning_horizon = config.get("PLANNING_HORIZON", 30)
        repair_hours = config.get("REPAIR_HOURS", 8)
        daily_loss = freq * y_mid * wph * value_per_wafer * 24
        break_even = (params["repair"] / daily_loss) if daily_loss > 0 else 9_999
        avoided = daily_loss * planning_horizon
        dt_cost = params["dt"] * repair_hours
        evoa = avoided - params["repair"] - dt_cost
        p_score = daily_loss * (1 / params["priority"]) if params["priority"] > 0 else 0
        rows.append({
            "binary_label": binary_label,
            "pattern_name": batch_row["pattern_name"],
            "count": int(batch_row.get("count", 0)),
            "batch_pct": batch_row.get("batch_pct", 0.0),
            "avg_confidence": batch_row.get("avg_confidence", 0.0),
            "triage_priority": params["priority"],
            "risk_level": params["risk"],
            "yield_range": f"{int(params['ylo'] * 100)}%-{int(params['yhi'] * 100)}%",
            "yield_mid": round(y_mid, 3),
            "repair_cost": params["repair"],
            "replacement_cost": params.get("replace"),
            "downtime_per_hr": params["dt"],
            "weighted_daily_loss": round(daily_loss, 2),
            "break_even_days": round(break_even, 1),
            "evoa_30d": round(evoa, 2),
            "priority_score": round(p_score, 2),
            "process_step": params["process"],
            "repair_action": params["action"],
        })
    df_financial = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
    # summary payload
    total_daily_loss = float(df_financial["weighted_daily_loss"].sum()) if not df_financial.empty else 0.0
    defective_wafers = (
        int(df_batch.loc[df_batch["binary_label"] != "00000000", "count"].sum()) if not df_batch.empty else 0
    )
    total_wafers = int(config.get("total_wafers", df_batch["count"].sum() if not df_batch.empty else 0))
    low_conf_count = int(config.get("low_conf_count", 0))
    top_action = (
        df_financial[df_financial["risk_level"] != "Normal"].iloc[0]["repair_action"]
        if not df_financial[df_financial["risk_level"] != "Normal"].empty
        else "None"
    )
    summary_payload = {
        "batch_id": config.get("BATCH_ID", ""),
        "total_wafers": total_wafers,
        "defective_wafers": defective_wafers,
        "defect_rate": round(defective_wafers / total_wafers, 4) if total_wafers > 0 else 0.0,
        "avg_confidence": round(float(df_batch["avg_confidence"].mean()) if not df_batch.empty else 0.0, 4),
        "low_conf_count": low_conf_count,
        "dominant_pattern": df_batch[df_batch["binary_label"] != "00000000"].iloc[0]["binary_label"]
        if not df_batch[df_batch["binary_label"] != "00000000"].empty
        else "N/A",
        "total_daily_loss": round(total_daily_loss, 2),
        "top_action": top_action,
        "patterns": {
            row["binary_label"]: {
                "pattern_name": row["pattern_name"],
                "count": int(row["count"]),
                "batch_pct": float(row["batch_pct"]),
                "weighted_daily_loss": float(row["weighted_daily_loss"]),
                "priority_score": float(row["priority_score"]),
            }
            for _, row in df_financial.iterrows()
        },
    }
    return df_financial, summary_payload


def save_reports(
    df_batch: pd.DataFrame, df_financial: pd.DataFrame, summary_payload: dict, outdir: str, batch_id: str
) -> tuple[str, str, str]:
    """Save CSV and JSON reports to outdir. Returns (csv_path, freq_path, json_path)."""
    import os

    os.makedirs(outdir, exist_ok=True)
    csv_path = f"{outdir}/financial_report_{batch_id}.csv"
    freq_path = f"{outdir}/batch_frequency_{batch_id}.csv"
    json_path = f"{outdir}/batch_summary_{batch_id}.json"
    df_financial.to_csv(csv_path, index=False)
    df_batch.to_csv(freq_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    return csv_path, freq_path, json_path
