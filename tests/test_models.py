"""Contract tests: every predictor fits on some grades and predicts unseen ones."""

import numpy as np
import pandas as pd
import pytest

from jominy.data import load_curves, to_frame
from jominy.models import (
    H_END_GRID,
    ROSTER,
    CalibratedPhysics,
    DataGBDT,
    HybridGBDT,
    HybridGP,
    LinearRegression,
    Physics,
    Predictor,
    Surrogate,
    calibrate_h_end,
    h_end_errors,
)
from jominy.physics import PhysicsModel
from jominy.synth import synthetic_frame


@pytest.fixture(scope="module")
def split() -> tuple[pd.DataFrame, pd.DataFrame]:
    real = to_frame(load_curves())
    test = real["grade"].isin(["42CrMo4", "16MnCr5"])
    return real[~test], real[test]


@pytest.fixture(scope="module")
def surrogate() -> Surrogate:
    return Surrogate(synthetic_frame(200, 0, PhysicsModel()), seed=0)


MODELS = [Physics, CalibratedPhysics, LinearRegression, DataGBDT, HybridGBDT, HybridGP]


@pytest.mark.parametrize("make", MODELS, ids=lambda m: m.key)
def test_predictions_are_finite_and_plausible(make, split) -> None:  # type: ignore[no-untyped-def]
    train, test = split
    model: Predictor = make().fit(train)
    pred, std = model.predict_with_std(test)
    assert pred.shape == (len(test),)
    assert np.all(np.isfinite(pred))
    assert np.all((pred > 0) & (pred < 75))
    assert std.shape == pred.shape
    assert np.isfinite(std).all() == isinstance(model, HybridGP)


def test_surrogate_ignores_training_data(surrogate: Surrogate, split) -> None:  # type: ignore[no-untyped-def]
    train, test = split
    before = surrogate.predict(test)
    assert np.array_equal(surrogate.fit(train).predict(test), before)


def test_roster_keys_are_unique_and_match_the_classes() -> None:
    keys = [spec.key for spec in ROSTER]
    assert len(keys) == len(set(keys))
    assert {m.key for m in (*MODELS, Surrogate)} == set(keys)


def test_calibration_picks_the_grid_minimum(split) -> None:  # type: ignore[no-untyped-def]
    train, _ = split
    errors = h_end_errors(train)
    assert set(errors) == set(H_END_GRID)
    assert errors[calibrate_h_end(train)] == min(errors.values())
