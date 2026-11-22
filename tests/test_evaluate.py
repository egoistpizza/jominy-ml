import numpy as np
import pandas as pd

from jominy.evaluate import (
    CVResult,
    cluster_bootstrap,
    holm,
    leave_one_group_out,
    sign_flip_pvalue,
)
from jominy.models import Predictor


class Spy(Predictor):
    """Predicts the number of groups it was trained on and refuses leaked rows."""

    key = "spy"
    name = "spy"

    def fit(self, train: pd.DataFrame) -> "Spy":
        self.trained_on = set(train["grade"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert not set(df["grade"]) & self.trained_on, "test group leaked into training"
        return np.full(len(df), float(len(self.trained_on)))


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "grade": ["A", "A", "B", "C", "C", "C"],
            "kind": "heat",
            "hrc": [1.0, 2, 3, 4, 5, 6],
            "curve_id": list("aabccc"),
        }
    )


def test_every_row_is_predicted_by_a_model_that_never_saw_its_group() -> None:
    cv = leave_one_group_out(_frame(), [Spy], n_jobs=1).frame
    assert (cv["pred"] == 2.0).all()
    assert (cv["model"] == "spy").all()
    assert cv["std"].isna().all()


def test_parallel_and_serial_runs_agree() -> None:
    serial = leave_one_group_out(_frame(), [Spy], n_jobs=1).frame
    parallel = leave_one_group_out(_frame(), [Spy], n_jobs=2).frame
    pd.testing.assert_frame_equal(serial, parallel)


def test_fold_frame_is_built_per_fold_from_the_held_out_grades() -> None:
    seen: list[frozenset[str]] = []

    def fold_frame(held_out: frozenset[str]) -> pd.DataFrame:
        seen.append(held_out)
        return _frame()

    leave_one_group_out(_frame(), [Spy], n_jobs=1, fold_frame=fold_frame)
    assert sorted(seen, key=sorted) == [frozenset("A"), frozenset("B"), frozenset("C")]


def test_cluster_bootstrap_interval_contains_estimate() -> None:
    rng = np.random.default_rng(1)
    errors = pd.Series(rng.normal(size=300))
    groups = pd.Series(np.repeat(np.arange(10), 30))
    ci = cluster_bootstrap(errors, groups, lambda e: float(np.mean(np.abs(e))), n_boot=300)
    assert ci.low <= ci.estimate <= ci.high


def test_sign_flip_is_exact_for_all_positive_differences() -> None:
    # With 5 positive values only the all-plus and all-minus patterns are as extreme.
    assert sign_flip_pvalue(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 2 / 32


def test_holm_is_monotone_and_capped() -> None:
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adjusted == {"a": 0.03, "b": 0.08, "c": 0.5}


def _result() -> CVResult:
    rows = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"] * 2,
            "kind": ["heat", "band_min", "band_max", "heat"] * 2,
            "hrc": [50.0, 40.0, 45.0, 30.0] * 2,
            "distance_mm": [1.5, 20.0, 1.5, 20.0] * 2,
            "model": ["x"] * 4 + ["y"] * 4,
            "pred": [51.0, 42.0, 45.0, 30.0, 50.0, 40.0, 49.0, 34.0],
            "std": [np.nan] * 4 + [1.0, 1.0, 1.0, 1.0],
        }
    )
    return CVResult(rows, n_boot=50)


def test_cv_result_subsets_by_curve_kind() -> None:
    cv = _result()
    assert cv.mae("x", "heat") == 0.5
    assert cv.mae("x", "band") == 1.0
    assert cv.mae("y", "band_max") == 4.0


def test_cv_result_per_group_and_near_far() -> None:
    cv = _result()
    assert cv.mae_per_group().loc["A", "x"] == 1.5
    assert cv.group_differences("x", "y").tolist() == [1.5, -4.0]
    assert cv.mae_near_far("x", near_mm=3.0) == (0.5, 1.0)


def test_cv_result_coverage_counts_points_inside_the_interval() -> None:
    assert _result().coverage("y") == 0.5
