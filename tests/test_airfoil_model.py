"""Tests for the Airfoil Self-Noise Prediction project."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.Airfoil_Functions import (
    FEATURE_COLUMNS,
    evaluate_loco,
    median_ms,
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
        Dataset containing the five predictor variables and sound
        pressure level target.
    """
    return pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=COLUMN_NAMES,
    )


@pytest.fixture
def valid_input():
    """Create one valid prediction input within the experimental domain.

    Returns
    -------
    pandas.DataFrame
        Single observation containing valid values for all required
        airfoil predictor variables.
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
    """Test that the dataset has the expected dimensions.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Airfoil Self-Noise dataset supplied by the pytest fixture.

    Returns
    -------
    None
        This test passes if the dataset contains 1,503 observations
        and six columns.
    """
    assert airfoil_data.shape == (1503, 6)


def test_dataset_columns(airfoil_data):
    """Test that the dataset contains the expected variables.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Airfoil Self-Noise dataset supplied by the pytest fixture.

    Returns
    -------
    None
        This test passes if the dataset columns match the expected
        predictor and target variable names.
    """
    assert list(airfoil_data.columns) == COLUMN_NAMES


def test_dataset_has_no_missing_values(airfoil_data):
    """Test that the dataset contains no missing values.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Airfoil Self-Noise dataset supplied by the pytest fixture.

    Returns
    -------
    None
        This test passes if no missing values are present in the
        dataset.
    """
    assert not airfoil_data.isna().any().any()


# ---------------------------------------------------------------------
# Experimental design tests
# ---------------------------------------------------------------------

def test_withheld_chord_is_excluded_from_development_data(
    airfoil_data,
):
    """Test that the final test chord is excluded from development data.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Airfoil Self-Noise dataset supplied by the pytest fixture.

    Returns
    -------
    None
        This test passes if the 0.1524 m chord is absent from the
        development data and all test observations belong to that chord.
    """
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
    """Test the expected chord-based development and test set sizes.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Airfoil Self-Noise dataset supplied by the pytest fixture.

    Returns
    -------
    None
        This test passes if the development set contains 1,232
        observations and the withheld test set contains 271 observations.
    """
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
    """Test that valid predictor values pass input validation.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Reference experimental dataset used to determine valid ranges.
    valid_input : pandas.DataFrame
        Valid predictor values supplied by the pytest fixture.

    Returns
    -------
    None
        This test passes if validation returns ``True`` for valid input.
    """
    assert validate_airfoil_input(
        valid_input,
        airfoil_data,
    ) is True


def test_missing_predictor_is_rejected(
    airfoil_data,
    valid_input,
):
    """Test that an input with a missing predictor is rejected.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Reference experimental dataset used to determine valid ranges.
    valid_input : pandas.DataFrame
        Valid predictor values used to construct the invalid input.

    Returns
    -------
    None
        This test passes if a ``ValueError`` is raised when a required
        predictor is missing.
    """
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
    """Test that non-finite predictor values are rejected.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Reference experimental dataset used to determine valid ranges.
    valid_input : pandas.DataFrame
        Valid predictor values used to construct the invalid input.

    Returns
    -------
    None
        This test passes if a ``ValueError`` is raised when a predictor
        contains a NaN value.
    """
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
    """Test that non-positive physical predictor values are rejected.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Reference experimental dataset used to determine valid ranges.
    valid_input : pandas.DataFrame
        Valid predictor values used to construct the invalid input.

    Returns
    -------
    None
        This test passes if a ``ValueError`` is raised when a physical
        predictor has a non-positive value.
    """
    invalid_input = valid_input.copy()
    invalid_input.loc[0, "frequency"] = -100.0

    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        validate_airfoil_input(
            invalid_input,
            airfoil_data,
        )


def test_out_of_range_input_is_rejected(
    airfoil_data,
    valid_input,
):
    """Test that inputs outside the experimental domain are rejected.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Reference experimental dataset used to determine valid ranges.
    valid_input : pandas.DataFrame
        Valid predictor values used to construct the invalid input.

    Returns
    -------
    None
        This test passes if a ``ValueError`` is raised when a predictor
        falls outside its observed experimental range.
    """
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
    """Test that LOCO validation evaluates every chord exactly once.

    Parameters
    ----------
    airfoil_data : pandas.DataFrame
        Airfoil Self-Noise dataset supplied by the pytest fixture.

    Returns
    -------
    None
        This test passes if every unique chord length is withheld
        exactly once during leave-one-chord-out validation.
    """
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


# ---------------------------------------------------------------------
# Computational utility tests
# ---------------------------------------------------------------------

def test_median_ms_returns_non_negative_time():
    """Test that ``median_ms`` returns a non-negative execution time.

    Returns
    -------
    None
        This test passes if the measured median execution time is
        greater than or equal to zero milliseconds.
    """
    result = median_ms(
        lambda: None,
        repeats=3,
    )

    assert result >= 0


def test_median_ms_rejects_invalid_repeats():
    """Test that ``median_ms`` rejects an invalid repeat count.

    Returns
    -------
    None
        This test passes if ``median_ms`` raises a ``ValueError`` when
        the requested number of repeats is less than one.
    """
    with pytest.raises(
        ValueError,
        match="repeats must be at least 1",
    ):
        median_ms(
            lambda: None,
            repeats=0,
        )
