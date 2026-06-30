import pandas as pd
import pytest

from src.data_validation import (
    customer_overlap_summary,
    validate_binary_target,
    validate_probability_series,
)


def test_binary_target_accepts_zero_and_one():
    result = validate_binary_target([0, 1, 0, 1])
    assert result.tolist() == [0, 1, 0, 1]


def test_binary_target_rejects_other_values():
    with pytest.raises(ValueError):
        validate_binary_target([0, 1, 2])


def test_probability_validation_rejects_out_of_range():
    with pytest.raises(ValueError):
        validate_probability_series([0.1, 1.2])


def test_customer_overlap_is_zero_for_independent_splits():
    frame = pd.DataFrame(
        {
            "msno": ["a", "b", "c", "d", "e", "f"],
            "split": [
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
            ],
        }
    )

    summary = customer_overlap_summary(
        frame,
        id_column="msno",
        split_column="split",
    )

    assert summary["overlapping_customers"].sum() == 0


def test_customer_overlap_detects_reused_customer():
    frame = pd.DataFrame(
        {
            "msno": ["a", "b", "a"],
            "split": ["train", "validation", "test"],
        }
    )

    summary = customer_overlap_summary(
        frame,
        id_column="msno",
        split_column="split",
    )

    assert summary["overlapping_customers"].sum() > 0
