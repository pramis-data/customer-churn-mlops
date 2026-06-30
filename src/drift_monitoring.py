"""Population and prediction drift utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


PSI_EPSILON = 1e-6


def classify_psi(psi_value: float) -> str:
    """Classify PSI using the dissertation monitoring thresholds."""

    if pd.isna(psi_value):
        return "not_available"

    if psi_value < 0.10:
        return "stable"

    if psi_value < 0.25:
        return "warning"

    return "significant"


def psi_from_proportions(
    reference_proportions,
    current_proportions,
) -> float:
    """Calculate PSI from aligned probability vectors."""

    reference = np.clip(
        np.asarray(reference_proportions, dtype=float),
        PSI_EPSILON,
        None,
    )
    current = np.clip(
        np.asarray(current_proportions, dtype=float),
        PSI_EPSILON,
        None,
    )

    if reference.shape != current.shape:
        raise ValueError("PSI vectors must have matching shapes.")

    return float(
        np.sum(
            (current - reference)
            * np.log(current / reference)
        )
    )


def numeric_psi(
    reference_series,
    current_series,
    number_of_bins: int = 10,
) -> float:
    """Calculate numeric PSI using reference quantile bins."""

    reference = pd.to_numeric(
        pd.Series(reference_series),
        errors="coerce",
    ).dropna()
    current = pd.to_numeric(
        pd.Series(current_series),
        errors="coerce",
    ).dropna()

    if reference.empty or current.empty:
        return float("nan")

    if reference.nunique() <= 1:
        if current.nunique() <= 1 and current.iloc[0] == reference.iloc[0]:
            return 0.0
        return float("nan")

    edges = np.unique(
        reference.quantile(
            np.linspace(0, 1, number_of_bins + 1)
        ).to_numpy()
    )

    if len(edges) < 3:
        edges = np.linspace(
            float(reference.min()),
            float(reference.max()),
            number_of_bins + 1,
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_bins = pd.cut(
        reference,
        bins=edges,
        include_lowest=True,
    )
    current_bins = pd.cut(
        current,
        bins=edges,
        include_lowest=True,
    )

    reference_proportions = (
        reference_bins.value_counts(
            normalize=True,
            sort=False,
        ).to_numpy()
    )
    current_proportions = (
        current_bins.value_counts(
            normalize=True,
            sort=False,
        )
        .reindex(
            reference_bins.cat.categories,
            fill_value=0,
        )
        .to_numpy()
    )

    return psi_from_proportions(
        reference_proportions,
        current_proportions,
    )


def categorical_psi(
    reference_series,
    current_series,
) -> float:
    """Calculate PSI for categorical values."""

    reference = (
        pd.Series(reference_series)
        .astype("string")
        .fillna("__MISSING__")
    )
    current = (
        pd.Series(current_series)
        .astype("string")
        .fillna("__MISSING__")
    )

    categories = sorted(
        set(reference.unique()) | set(current.unique())
    )

    reference_proportions = (
        reference.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy()
    )
    current_proportions = (
        current.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy()
    )

    return psi_from_proportions(
        reference_proportions,
        current_proportions,
    )


def drift_alert(
    item: str,
    psi_value: float,
) -> dict[str, object] | None:
    """Return a monitoring alert when PSI exceeds 0.10."""

    severity = classify_psi(psi_value)

    if severity == "stable" or severity == "not_available":
        return None

    return {
        "alert_type": "drift",
        "item": item,
        "severity": severity,
        "observed_value": float(psi_value),
        "warning_threshold": 0.10,
        "significant_threshold": 0.25,
    }
