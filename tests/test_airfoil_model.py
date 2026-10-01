"""Tests for the Airfoil Self-Noise Prediction project."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


COLUMN_NAMES = [
    "frequency",
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "displacement_thickness",
    "sound_pressure_level",
]

FEATURE_COLUMNS = [
    "frequency",
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "displacement_thickness",
]

TEST_CHORD = 0.1524

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "airfoil_self_noise.dat"
)


@pytest.fixture
def airfoil_data():
    """Load the airfoil dataset for testing.

    Returns
    -------
    pandas.DataFrame
        Airfoil Self-Noise dataset with named columns.
    """
    return pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=COLUMN_NAMES,
    )


def test_dataset_shape(airfoil_data):
    """Verify that the expected dataset has been loaded."""
    assert airfoil_data.shape == (1503, 6)


def test_dataset_columns(airfoil_data):
    """Verify that the dataset contains the expected variables."""
    assert list(airfoil_data.columns) == COLUMN_NAMES


def test_dataset_has_no_missing_values(airfoil_data):
    """Verify the documented absence of missing dataset values."""
    assert not airfoil_data.isna().any().any()


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
    """Verify the expected chord-based development/test split."""
    test_mask = np.isclose(
        airfoil_data["chord_length"],
        TEST_CHORD,
    )

    assert (~test_mask).sum() == 1232
    assert test_mask.sum() == 271
