"""Financial impact library extracted from waferguard_financial_impact.ipynb.

Provides pure financial analysis functions: merge predictions, decode labels,
build batch summaries, compute financial metrics, and save reports.

Fully decoupled from inference and artifact loading (Streamlit-compatible).
Financial parameters are loaded from data/Defect_Financial_Mapping.xlsx so
updates to the mapping table take effect without touching source code.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Financial parameter loading from Excel
# ---------------------------------------------------------------------------

_EXCEL_PATH = Path(__file__).parent.parent / "data" / "Defect_Financial_Mapping.xlsx"


def _parse_yield_range(value: object) -> tuple[float, float]:
    """Parse '10%-40%' style strings into (ylo, yhi) fractions."""
    s = str(value).replace("%", "").replace("\u2013", "-").replace("\u2014", "-")
    if "-" in s:
        lo, hi = s.split("-", 1)
        return float(lo) / 100, float(hi) / 100
    v = float(s) / 100
    return v, v


def _int_val(val: object, default: int = 0) -> int:
    s = str(val).strip()
    if s in ("\u2014", "\u2013", "-", "nan", ""):
        return default
    try:
        return int(float(s.replace(",", "")))
    except ValueError:
        return default


def _replace_val(val: object) -> int | None:
    if pd.isna(val):
        return None
    return _int_val(val)


def _load_financial_params(path: Path = _EXCEL_PATH) -> dict[str, dict]:
    """Load FINANCIAL_PARAMS from the Excel mapping table."""
    df = pd.read_excel(path, sheet_name="Defect Financial Mapping", header=2)
    df.columns = [
        "binary_label",
        "pattern_id",
        "pattern_name",
        "mix_type",
        "process",
        "tool",
        "root_cause",
        "yield_range",
        "repair",
        "replace",
        "dt",
        "priority",
        "risk",
        "action",
        "notes",
    ]
    df = df[df["binary_label"].astype(str).str.match(r"^[01]{8}$")].copy()

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        lbl = str(row["binary_label"])
        ylo, yhi = _parse_yield_range(row["yield_range"])
        risk = str(row["risk"])
        process = str(row["process"])
        action = str(row["action"])
        result[lbl] = {
            "repair": _int_val(row["repair"]),
            "replace": _replace_val(row["replace"]),
            "dt": _int_val(row["dt"]),
            "ylo": ylo,
            "yhi": yhi,
            "priority": _int_val(row["priority"], default=5),
            "risk": risk if risk != "nan" else "Normal",
            "process": process if process != "nan" else "N/A",
            "action": action if action != "nan" else "No action required",
        }
    return result


FINANCIAL_PARAMS: dict[str, dict] = _load_financial_params()

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


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def get_predictions(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract class IDs and confidences from a single model's prediction matrix.

    Args:
        predictions: probability array of shape (N, num_classes) from the user-selected model.

    Returns (class_ids, confidences).

    """
    class_ids = predictions.argmax(axis=1)
    confidences = predictions.max(axis=1)
    return class_ids, confidences


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
    # Single pass: collect confidences per label
    label_to_confs: dict[str, list[float]] = defaultdict(list)
    for lbl, conf in zip(binary_labels, confidences, strict=True):
        label_to_confs[lbl].append(float(conf))
    batch_rows = []
    for binary_label, count in sorted(Counter(binary_labels).items(), key=lambda x: -x[1]):
        name = label_mapping.get(binary_label, "Unknown")
        freq = count / total_wafers if total_wafers > 0 else 0.0
        confs_this = label_to_confs[binary_label]
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
    return pd.DataFrame(batch_rows)


def build_base_anomaly_df(
    binary_labels: list[str],
    confidences: np.ndarray,
    label_mapping: dict[str, str],
    financial_params: dict | None = None,
) -> pd.DataFrame:
    """Build a batch summary decomposed to the 8 base anomaly types.

    Each wafer detection is decomposed into its constituent base anomalies.
    A wafer classified as 'Donut+Edge_Loc+Scratch' contributes +1 count to
    each of Donut, Edge_Loc, and Scratch independently.

    Normal wafers (00000000) contribute no count to any base anomaly.
    All 8 base anomaly types are always present in the output (count may be 0).

    Returns a DataFrame with the same schema as build_batch_df.

    """
    if financial_params is None:
        financial_params = FINANCIAL_PARAMS

    total_wafers = len(binary_labels)
    base_confs: dict[str, list[float]] = defaultdict(list)

    for lbl, conf in zip(binary_labels, confidences, strict=True):
        for base in decompose(lbl):
            base_confs[base].append(float(conf))

    all_bases = sorted(lbl for lbl in financial_params if lbl != "00000000" and lbl.count("1") == 1)

    rows = []
    for base in all_bases:
        name = label_mapping.get(base, "Unknown")
        confs = base_confs.get(base, [])
        count = len(confs)
        freq = count / total_wafers if total_wafers > 0 else 0.0
        rows.append({
            "binary_label": base,
            "pattern_name": name,
            "count": count,
            "batch_freq": round(freq, 6),
            "batch_pct": round(freq * 100, 2),
            "avg_confidence": round(float(np.mean(confs)) if confs else 0.0, 4),
            "min_confidence": round(float(np.min(confs)) if confs else 0.0, 4),
            "max_confidence": round(float(np.max(confs)) if confs else 0.0, 4),
        })

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def decompose(binary_label: str) -> list[str]:
    """Return list of single-defect binary labels present in a combo."""
    return [
        BIT_TO_SINGLE[bit]
        for bit in range(7, -1, -1)
        if binary_label[7 - bit] == "1" and BIT_TO_SINGLE[bit] in FINANCIAL_PARAMS
    ]


def _get_action_to_singles(financial_params: dict) -> dict[str, dict]:
    """Map each unique base repair action to its single-defect pattern metadata."""
    action_map: dict[str, dict] = {}
    for lbl, params in financial_params.items():
        if lbl == "00000000" or lbl.count("1") != 1:
            continue
        for act in (a.strip() for a in params["action"].split(";") if a.strip()):
            if act not in action_map:
                action_map[act] = {
                    "singles": [],
                    "process": params["process"],
                    "risk": params["risk"],
                    "priority": params["priority"],
                    "repair": params["repair"],
                }
            action_map[act]["singles"].append(lbl)
    return action_map


def get_params(binary_label: str, financial_params: dict | None = None) -> dict:
    """Look up financial params for any pattern. Aggregates multi-defect combos."""
    if financial_params is None:
        financial_params = FINANCIAL_PARAMS
    if binary_label in financial_params:
        return financial_params[binary_label]
    components = decompose(binary_label)
    if not components:
        return financial_params["00000000"]
    ds = [financial_params[c] for c in components if c in financial_params]
    if not ds:
        return financial_params["00000000"]
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
        params = get_params(binary_label, financial_params)
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
            "yield_loss_pct": round(y_mid * 100, 1),
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


def compute_action_table(df_financial: pd.DataFrame, financial_params: dict | None = None) -> pd.DataFrame:
    """Build an action-oriented financial table.

    Each row = one repair/maintenance action. For each action, aggregates the
    financial impact across all 38 patterns it resolves (fully or partially).

    Full resolution: action addresses all root-cause components of the pattern.
    Partial resolution: action addresses some components; savings are proportional
    to the fraction of components fixed.

    Args:
        df_financial: output of compute_financials (pattern-oriented rows).
        financial_params: optional override; defaults to FINANCIAL_PARAMS.

    Returns:
        DataFrame sorted by daily_loss_savings descending with columns:
        repair_action, process_step, risk_level, triage_priority, repair_cost,
        patterns_fully_resolved, patterns_partially_resolved,
        daily_loss_savings, break_even_days, evoa_30d.

    """
    if financial_params is None:
        financial_params = FINANCIAL_PARAMS

    action_map = _get_action_to_singles(financial_params)

    # Build lookup: binary_label → observed daily loss from this batch
    fin_lookup: dict[str, float] = {
        str(row["binary_label"]): float(row["weighted_daily_loss"]) for _, row in df_financial.iterrows()
    }

    rows = []
    for action, meta in action_map.items():
        single_set = set(meta["singles"])
        fully_resolved: list[str] = []
        partially_resolved: list[str] = []
        total_savings = 0.0

        for binary_label, daily_loss in fin_lookup.items():
            if binary_label == "00000000":
                continue
            components = set(decompose(binary_label)) or {binary_label}
            overlap = components & single_set
            if not overlap:
                continue
            if overlap >= components:
                fully_resolved.append(binary_label)
                total_savings += daily_loss
            else:
                partially_resolved.append(binary_label)
                total_savings += daily_loss * len(overlap) / len(components)

        if not fully_resolved and not partially_resolved:
            continue

        repair = meta["repair"]
        rows.append({
            "repair_action": action,
            "process_step": meta["process"],
            "risk_level": meta["risk"],
            "triage_priority": meta["priority"],
            "repair_cost": repair,
            "patterns_fully_resolved": len(fully_resolved),
            "patterns_partially_resolved": len(partially_resolved),
            "daily_loss_savings": round(total_savings, 2),
            "break_even_days": round(repair / total_savings, 1) if total_savings > 0 else 9_999,
            "evoa_30d": round(total_savings * 30 - repair, 2),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("daily_loss_savings", ascending=False).reset_index(drop=True)


def save_reports(
    df_batch: pd.DataFrame, df_financial: pd.DataFrame, summary_payload: dict, outdir: str, batch_id: str
) -> tuple[str, str, str]:
    """Save CSV and JSON reports to outdir. Returns (csv_path, freq_path, json_path)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"financial_report_{batch_id}.csv"
    freq_path = out / f"batch_frequency_{batch_id}.csv"
    json_path = out / f"batch_summary_{batch_id}.json"
    df_financial.to_csv(csv_path, index=False)
    df_batch.to_csv(freq_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    return str(csv_path), str(freq_path), str(json_path)
