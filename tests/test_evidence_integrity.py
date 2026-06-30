from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_customer_overlap_evidence_passed():
    status_file = (
        ROOT
        / "evidence"
        / "customer_overlap"
        / "09_final_customer_overlap_status.csv"
    )
    status = pd.read_csv(status_file).iloc[0]

    assert status["status"] == "PASSED"
    assert int(status["pairwise_overlap_total"]) == 0
    assert int(status["three_way_overlap_count"]) == 0


def test_drift_psi_values_are_non_negative():
    drift_file = (
        ROOT
        / "evidence"
        / "drift_monitoring"
        / "01_feature_drift_summary.csv"
    )
    drift = pd.read_csv(drift_file)

    assert (drift["psi"].dropna() >= 0).all()


def test_final_comparison_contains_both_models():
    comparison_file = (
        ROOT
        / "evidence"
        / "final_model_comparison"
        / "11_final_same_test_comparison.csv"
    )
    comparison = pd.read_csv(comparison_file)

    assert set(comparison["model"]) == {
        "Random Forest",
        "Azure AutoML selected model",
    }
