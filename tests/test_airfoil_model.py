"""Tests for the Airfoil Self-Noise Prediction project."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.airfoil_utils import (
    FEATURE_COLUMNS,
    evaluate_loco,
    validate_airfoil_input,
)


# Expected structure of the Airfoil Self-Noise dataset
COLUMN_NAMES = [
    "frequency",
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "displacement_thickness",
    "sound_pressure_level",
]

# Chord length reserved for final model evaluation
TEST_CHORD = 0.1524

# Locate the dataset relative to the repository root
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "airfoil_self_noise.dat"
)


@pytest.fixture
def airfoil_data():
    """Load the Airfoil Self-Noise dataset for testing.

    Returns
    -------
    pandas.DataFrame
        Dataset containing the five predictors and SPL target.
    """
    return pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=COLUMN_NAMES,
    )


@pytest.fixture
def valid_input():
    """Return one valid prediction input within the experimental domain.

    Returns
    -------
    pandas.DataFrame
        Single valid set of airfoil predictor values.
    """
    return pd.DataFrame({
        "frequency": [1000.0],
        "angle_of_attack": [5.0],
        "chord_length": [0.1524],
        "free_stream_velocity": [55.5],
        "displacement_thickness": [0.005],
    })


# ---------------------------------------------------------------------
# Dataset integrity tests
# ---------------------------------------------------------------------

def test_dataset_shape(airfoil_data):
    """Verify that the complete dataset has the expected dimensions."""
    assert airfoil_data.shape == (1503, 6)


def test_dataset_columns(airfoil_data):
    """Verify that the dataset contains the expected variables."""
    assert list(airfoil_data.columns) == COLUMN_NAMES


def test_dataset_has_no_missing_values(airfoil_data):
    """Verify the documented absence of missing dataset values."""
    assert not airfoil_data.isna().any().any()


# ---------------------------------------------------------------------
# Experimental design tests
# ---------------------------------------------------------------------

def test_withheld_chord_is_excluded_from_development_data(
    airfoil_data,
):
    """Verify that the final test chord is excluded from development data."""
    test_mask = np.isclose(
        airfoil_data["chord_length"],
        TEST_CHORD,
    )

    development_data = airfoil_data.loc[~test_mask]
    test_data = airfoil_data.loc[test_mask]

    assert not np.isclose(
        development_data["chord_length"],
        TEST_CHORD,
    ).any()

    assert np.isclose(
        test_data["chord_length"],
        TEST_CHORD,
    ).all()


def test_development_and_test_sizes(airfoil_data):
    """Verify the expected chord-based development and test sizes."""
    test_mask = np.isclose(
        airfoil_data["chord_length"],
        TEST_CHORD,
    )

    assert (~test_mask).sum() == 1232
    assert test_mask.sum() == 271


# ---------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------

def test_valid_input_is_accepted(airfoil_data, valid_input):
    """Verify that valid predictor values pass input validation."""
    assert validate_airfoil_input(
        valid_input,
        airfoil_data,
    ) is True


def test_missing_predictor_is_rejected(
    airfoil_data,
    valid_input,
):
    """Verify that an input with a missing predictor is rejected."""
    invalid_input = valid_input.drop(
        columns=["frequency"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required predictors",
    ):
        validate_airfoil_input(
            invalid_input,
            airfoil_data,
        )


def test_nan_input_is_rejected(airfoil_data, valid_input):
    """Verify that NaN predictor values are rejected."""
    invalid_input = valid_input.copy()
    invalid_input.loc[0, "frequency"] = np.nan

    with pytest.raises(
        ValueError,
        match="finite numerical values",
    ):
        validate_airfoil_input(
            invalid_input,
            airfoil_data,
        )


def test_negative_physical_input_is_rejected(
    airfoil_data,
    valid_input,
):
    """Verify that negative physical predictor values are rejected."""
    invalid_input = valid_input.copy()
    invalid_input.loc[0, "frequency"] = -100.0

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        validate_airfoil_input(
            invalid_input,
            airfoil_data,
        )


def test_out_of_range_input_is_rejected(
    airfoil_data,
    valid_input,
):
    """Verify that inputs outside the experimental domain are rejected."""
    invalid_input = valid_input.copy()
    invalid_input.loc[0, "frequency"] = 25000.0

    with pytest.raises(
        ValueError,
        match="experimental range",
    ):
        validate_airfoil_input(
            invalid_input,
            airfoil_data,
        )


# ---------------------------------------------------------------------
# Machine-learning validation tests
# ---------------------------------------------------------------------

def test_loco_withholds_each_chord_once(airfoil_data):
    """Verify that LOCO validation evaluates every chord exactly once."""
    X = airfoil_data[FEATURE_COLUMNS]
    y = airfoil_data["sound_pressure_level"]

    groups = airfoil_data[
        "chord_length"
    ].to_numpy()

    results = evaluate_loco(
        LinearRegression(),
        X,
        y,
        groups,
    )

    expected_chords = np.sort(
        airfoil_data["chord_length"].unique()
    )

    evaluated_chords = np.sort(
        results["Withheld Chord"].to_numpy()
    )

    assert len(results) == len(expected_chords)

    assert np.allclose(
        evaluated_chords,
        expected_chords,
    )
