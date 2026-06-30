from pathlib import Path
import pandas as pd

from src.evaluation import classification_metrics


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_FILE = (
    ROOT
    / "evidence"
    / "final_model_comparison"
    / "12_final_test_prediction_evidence.csv"
)


def test_model_quality_gates():
    frame = pd.read_csv(PREDICTION_FILE)

    manual = classification_metrics(
        frame["actual_class"],
        frame["manual_probability"],
        threshold=0.50,
    )
    automl = classification_metrics(
        frame["actual_class"],
        frame["automl_probability"],
        threshold=0.50,
    )

    assert manual["roc_auc"] >= 0.90
    assert automl["roc_auc"] >= 0.90
    assert manual["average_precision"] >= 0.70
    assert automl["average_precision"] >= 0.70


def test_automl_outperforms_manual_on_primary_test_metrics():
    frame = pd.read_csv(PREDICTION_FILE)

    manual = classification_metrics(
        frame["actual_class"],
        frame["manual_probability"],
    )
    automl = classification_metrics(
        frame["actual_class"],
        frame["automl_probability"],
    )

    assert automl["roc_auc"] > manual["roc_auc"]
    assert automl["average_precision"] > manual["average_precision"]
    assert automl["f1_churn"] > manual["f1_churn"]
