"""Prediction helpers that avoid model-specific assumptions."""

from __future__ import annotations

import numpy as np


def positive_class_probability(model, features) -> np.ndarray:
    """Return the probability associated with class 1."""

    if not hasattr(model, "predict_proba"):
        raise TypeError("The model does not implement predict_proba.")

    probability_matrix = np.asarray(
        model.predict_proba(features)
    )

    if probability_matrix.ndim != 2:
        raise ValueError(
            "predict_proba must return a two-dimensional matrix."
        )

    classes = getattr(model, "classes_", None)

    if classes is None:
        positive_index = 1
    else:
        classes = np.asarray(classes).astype(str)
        positions = np.where(classes == "1")[0]
        positive_index = int(positions[0]) if len(positions) == 1 else 1

    probabilities = probability_matrix[:, positive_index]

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("The model returned invalid probabilities.")

    return probabilities.astype(float)
