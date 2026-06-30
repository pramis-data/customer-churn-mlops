"""Data and prediction validation utilities."""

from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd


DEFAULT_PREDICTION_COLUMNS = {
    "actual_class",
    "manual_probability",
    "automl_probability",
}


def validate_binary_target(values: Iterable, name: str = "target") -> pd.Series:
    """Return a validated binary integer target series."""

    target = pd.to_numeric(
        pd.Series(values),
        errors="raise",
    )

    if target.isna().any():
        raise ValueError(f"{name} contains missing values.")

    unique_values = set(target.unique().tolist())

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"{name} must contain only 0 and 1; found {sorted(unique_values)}."
        )

    return target.astype(int)


def validate_probability_series(
    values: Iterable,
    name: str = "probability",
) -> pd.Series:
    """Return a numeric probability series constrained to [0, 1]."""

    probabilities = pd.to_numeric(
        pd.Series(values),
        errors="raise",
    )

    if probabilities.isna().any():
        raise ValueError(f"{name} contains missing values.")

    valid = probabilities.between(
        0,
        1,
        inclusive="both",
    )

    if not valid.all():
        raise ValueError(f"{name} contains values outside [0, 1].")

    return probabilities.astype(float)


def validate_prediction_frame(
    frame: pd.DataFrame,
    required_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Validate the shared test-prediction evidence schema."""

    required = required_columns or DEFAULT_PREDICTION_COLUMNS
    missing = required - set(frame.columns)

    if missing:
        raise KeyError(
            "Missing prediction columns: "
            + ", ".join(sorted(missing))
        )

    validated = frame.copy()
    validated["actual_class"] = validate_binary_target(
        validated["actual_class"],
        name="actual_class",
    )

    for column in sorted(required - {"actual_class"}):
        validated[column] = validate_probability_series(
            validated[column],
            name=column,
        )

    return validated


def compare_schema(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict[str, list[str]]:
    """Return missing, additional and reordered columns."""

    reference_columns = list(reference.columns)
    current_columns = list(current.columns)

    return {
        "missing_columns": sorted(
            set(reference_columns) - set(current_columns)
        ),
        "additional_columns": sorted(
            set(current_columns) - set(reference_columns)
        ),
        "order_matches": reference_columns == current_columns,
    }


def customer_overlap_summary(
    frame: pd.DataFrame,
    id_column: str,
    split_column: str,
) -> pd.DataFrame:
    """Calculate customer overlap across training, validation and test."""

    if id_column not in frame.columns:
        raise KeyError(f"Missing customer-ID column: {id_column}")

    if split_column not in frame.columns:
        raise KeyError(f"Missing split column: {split_column}")

    working = frame[[id_column, split_column]].dropna().copy()
    working[id_column] = working[id_column].astype(str).str.strip()
    working[split_column] = (
        working[split_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "train": "training",
                "val": "validation",
                "valid": "validation",
                "testing": "test",
            }
        )
    )

    split_names = ["training", "validation", "test"]
    customer_sets = {
        split_name: set(
            working.loc[
                working[split_column] == split_name,
                id_column,
            ]
        )
        for split_name in split_names
    }

    rows = []

    for first, second in [
        ("training", "validation"),
        ("training", "test"),
        ("validation", "test"),
    ]:
        overlap = customer_sets[first] & customer_sets[second]
        rows.append(
            {
                "comparison": f"{first}_vs_{second}",
                "overlapping_customers": len(overlap),
            }
        )

    three_way = (
        customer_sets["training"]
        & customer_sets["validation"]
        & customer_sets["test"]
    )

    rows.append(
        {
            "comparison": "training_vs_validation_vs_test",
            "overlapping_customers": len(three_way),
        }
    )

    return pd.DataFrame(rows)
