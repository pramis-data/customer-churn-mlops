from pathlib import Path
import pandas as pd

from src.data_validation import validate_prediction_frame


EVIDENCE_FILE = (
    Path(__file__).resolve().parents[1]
    / "evidence"
    / "final_model_comparison"
    / "12_final_test_prediction_evidence.csv"
)


def test_prediction_evidence_has_expected_schema():
    frame = pd.read_csv(EVIDENCE_FILE)
    validated = validate_prediction_frame(frame)

    assert len(validated) == 94669
    assert validated["actual_class"].isna().sum() == 0
    assert validated["manual_probability"].between(0, 1).all()
    assert validated["automl_probability"].between(0, 1).all()
