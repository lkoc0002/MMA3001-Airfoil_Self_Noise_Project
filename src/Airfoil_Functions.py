"""Utility functions for the Airfoil Self-Noise Prediction project.

This module contains reusable functions used for model validation and
input checking in the airfoil self-noise machine-learning workflow.
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneGroupOut


FEATURE_COLUMNS = [
    "frequency",
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "displacement_thickness",
]


def evaluate_loco(model, X, y, groups):
    """Evaluate a regression model using leave-one-chord-out validation.

    Parameters
    ----------
    model : sklearn estimator
        Regression model to evaluate.
    X : pandas.DataFrame
        Predictor variables used by the regression model.
    y : pandas.Series
        Sound pressure level target values.
    groups : array-like
        Chord length associated with each observation.

    Returns
    -------
    pandas.DataFrame
        Training and validation performance for each withheld chord.
    """
    logo = LeaveOneGroupOut()
    results = []

    for train_index, validation_index in logo.split(X, y, groups):
        X_train = X.iloc[train_index]
        X_validation = X.iloc[validation_index]

        y_train = y.iloc[train_index]
        y_validation = y.iloc[validation_index]

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        train_predictions = fitted_model.predict(X_train)
        validation_predictions = fitted_model.predict(X_validation)

        withheld_chord = np.asarray(groups)[validation_index][0]

        results.append({
            "Withheld Chord": withheld_chord,
            "Train RMSE": np.sqrt(
                mean_squared_error(y_train, train_predictions)
            ),
            "MAE": mean_absolute_error(
                y_validation,
                validation_predictions,
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    y_validation,
                    validation_predictions,
                )
            ),
            "R2": r2_score(
                y_validation,
                validation_predictions,
            ),
        })

    return pd.DataFrame(results)


def validate_airfoil_input(input_data, reference_data):
    """Validate airfoil inputs before model prediction.

    Parameters
    ----------
    input_data : pandas.DataFrame
        Predictor values to be supplied to the regression model.
    reference_data : pandas.DataFrame
        Experimental dataset used to determine the supported input ranges.

    Returns
    -------
    bool
        True when all predictor values pass validation.

    Raises
    ------
    ValueError
        If required predictors are missing, contain non-finite values,
        are physically invalid, or fall outside the experimental range.
    """
    # Check that every predictor required by the model is present.
    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in input_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required predictors: {missing_columns}"
        )

    predictor_data = input_data[FEATURE_COLUMNS]

    # Reject NaN and infinite values supplied for prediction.
    if not np.isfinite(predictor_data.to_numpy()).all():
        raise ValueError(
            "Prediction inputs must contain finite numerical values."
        )

    # Frequency, chord length, velocity and thickness cannot be negative.
    positive_features = [
        "frequency",
        "chord_length",
        "free_stream_velocity",
        "displacement_thickness",
    ]

    if (predictor_data[positive_features] < 0).any().any():
        raise ValueError(
            "Physical predictor values cannot be negative."
        )

    # Reject extrapolation beyond the experimental domain.
    for feature in FEATURE_COLUMNS:
        minimum = reference_data[feature].min()
        maximum = reference_data[feature].max()

        if (
            (predictor_data[feature] < minimum).any()
            or (predictor_data[feature] > maximum).any()
        ):
            raise ValueError(
                f"{feature} must be within the experimental range "
                f"[{minimum}, {maximum}]."
            )

    return True
