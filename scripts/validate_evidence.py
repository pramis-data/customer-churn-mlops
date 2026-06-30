"""Run repository-level evidence checks and write a JSON summary."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_validation import validate_prediction_frame
from src.evaluation import classification_metrics


def main() -> None:
    evidence_root = ROOT / "evidence"

    prediction_file = (
        evidence_root
        / "final_model_comparison"
        / "12_final_test_prediction_evidence.csv"
    )
    comparison_file = (
        evidence_root
        / "final_model_comparison"
        / "11_final_same_test_comparison.csv"
    )
    overlap_status_file = (
        evidence_root
        / "customer_overlap"
        / "09_final_customer_overlap_status.csv"
    )
    drift_file = (
        evidence_root
        / "drift_monitoring"
        / "01_feature_drift_summary.csv"
    )

    predictions = validate_prediction_frame(
        pd.read_csv(prediction_file)
    )
    comparison = pd.read_csv(comparison_file)
    overlap_status = pd.read_csv(overlap_status_file).iloc[0]
    drift = pd.read_csv(drift_file)

    manual_metrics = classification_metrics(
        predictions["actual_class"],
        predictions["manual_probability"],
    )
    automl_metrics = classification_metrics(
        predictions["actual_class"],
        predictions["automl_probability"],
    )

    checks = {
        "prediction_rows_equal_94669": len(predictions) == 94669,
        "manual_roc_auc_gate": manual_metrics["roc_auc"] >= 0.90,
        "automl_roc_auc_gate": automl_metrics["roc_auc"] >= 0.90,
        "manual_average_precision_gate": (
            manual_metrics["average_precision"] >= 0.70
        ),
        "automl_average_precision_gate": (
            automl_metrics["average_precision"] >= 0.70
        ),
        "customer_overlap_zero": (
            int(overlap_status["pairwise_overlap_total"]) == 0
            and int(overlap_status["three_way_overlap_count"]) == 0
        ),
        "psi_non_negative": bool(
            (drift["psi"].dropna() >= 0).all()
        ),
        "comparison_has_two_models": len(comparison) == 2,
    }

    summary = {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "manual_test_metrics": manual_metrics,
        "automl_test_metrics": automl_metrics,
    }

    output_path = evidence_root / "ci_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
