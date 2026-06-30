"""Model evaluation and profit-aware decision utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def apply_threshold(
    probabilities,
    threshold: float = 0.50,
) -> np.ndarray:
    """Convert probabilities into binary predictions."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie between 0 and 1.")

    probability_array = np.asarray(probabilities, dtype=float)

    if np.isnan(probability_array).any():
        raise ValueError("probabilities contain missing values.")

    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("probabilities must lie between 0 and 1.")

    return (probability_array >= threshold).astype(int)


def classification_metrics(
    actual,
    probabilities,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Calculate the core binary-classification metrics."""

    actual_array = np.asarray(actual, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    prediction = apply_threshold(probability_array, threshold)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(actual_array, prediction),
        "balanced_accuracy": balanced_accuracy_score(
            actual_array,
            prediction,
        ),
        "precision_churn": precision_score(
            actual_array,
            prediction,
            zero_division=0,
        ),
        "recall_churn": recall_score(
            actual_array,
            prediction,
            zero_division=0,
        ),
        "f1_churn": f1_score(
            actual_array,
            prediction,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            actual_array,
            probability_array,
        ),
        "average_precision": average_precision_score(
            actual_array,
            probability_array,
        ),
        "log_loss": log_loss(
            actual_array,
            probability_array,
            labels=[0, 1],
        ),
    }


def profit_outcome(
    actual,
    probabilities,
    threshold: float,
    retained_customer_value: float,
    intervention_success_rate: float,
    cost_per_contact: float,
    false_positive_penalty: float = 0.0,
    fixed_campaign_cost: float = 0.0,
) -> dict[str, float]:
    """Calculate scenario-based campaign outcomes."""

    actual_array = np.asarray(actual, dtype=int)
    prediction = apply_threshold(probabilities, threshold)

    true_positive = int(
        np.sum((actual_array == 1) & (prediction == 1))
    )
    false_positive = int(
        np.sum((actual_array == 0) & (prediction == 1))
    )
    false_negative = int(
        np.sum((actual_array == 1) & (prediction == 0))
    )
    contacted = true_positive + false_positive

    expected_retained_value = (
        true_positive
        * intervention_success_rate
        * retained_customer_value
    )
    contact_cost = contacted * cost_per_contact
    false_positive_cost = false_positive * false_positive_penalty
    campaign_fixed_cost = fixed_campaign_cost if contacted else 0.0
    total_cost = contact_cost + false_positive_cost + campaign_fixed_cost
    net_profit = expected_retained_value - total_cost

    return {
        "threshold": float(threshold),
        "customers_contacted": contacted,
        "true_churners_contacted": true_positive,
        "false_interventions": false_positive,
        "missed_churners": false_negative,
        "expected_retained_value": float(expected_retained_value),
        "total_campaign_cost": float(total_cost),
        "net_profit": float(net_profit),
    }


def threshold_grid(
    actual,
    probabilities,
    assumptions: dict[str, float],
    step: float = 0.01,
) -> pd.DataFrame:
    """Evaluate profit across thresholds from zero to one."""

    thresholds = np.round(
        np.arange(0.0, 1.0 + step, step),
        10,
    )

    rows = [
        profit_outcome(
            actual=actual,
            probabilities=probabilities,
            threshold=float(threshold),
            **assumptions,
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(rows)
