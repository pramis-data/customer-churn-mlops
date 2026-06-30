import numpy as np

from src.drift_monitoring import (
    classify_psi,
    drift_alert,
    numeric_psi,
    psi_from_proportions,
)


def test_identical_numeric_distributions_have_near_zero_psi():
    reference = np.arange(1, 101)
    current = reference.copy()

    assert numeric_psi(reference, current) < 1e-9


def test_psi_is_non_negative():
    value = psi_from_proportions(
        [0.50, 0.30, 0.20],
        [0.30, 0.40, 0.30],
    )

    assert value >= 0


def test_psi_threshold_classification():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "warning"
    assert classify_psi(0.30) == "significant"


def test_significant_drift_creates_alert():
    alert = drift_alert("is_auto_renew_mean", 0.30)

    assert alert is not None
    assert alert["severity"] == "significant"
