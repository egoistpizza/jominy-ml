"""Leave-one-group-out cross-validation and the statistics computed on its result."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed

from jominy.models import Predictor

FloatArray = npt.NDArray[np.float64]

#: Which curves a statistic is computed on. "band" means both limits of a band.
CurveKind = Literal["heat", "band", "band_min", "band_max"]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def _fold(
    make: Callable[[], Predictor], train: pd.DataFrame, test: pd.DataFrame
) -> tuple[str, FloatArray, FloatArray]:
    model = make().fit(train)
    pred, std = model.predict_with_std(test)
    return model.key, pred, std


def leave_one_group_out(
    df: pd.DataFrame,
    factories: Sequence[Callable[[], Predictor]],
    group_col: str = "grade",
    *,
    n_jobs: int = -1,
    n_boot: int = 2000,
    seed: int = 0,
    fold_frame: Callable[[frozenset[str]], pd.DataFrame] | None = None,
) -> CVResult:
    """Predict every row from models trained on all *other* groups.

    Folds are independent and run in parallel. ``n_boot`` and ``seed`` configure the
    bootstrap statistics of the returned result.

    If part of the data preparation depends on other rows, ``fold_frame`` rebuilds the
    frame for each fold from the set of held-out grades, so that the preparation sees
    training rows only. It must return the rows of ``df`` in the same order.
    """
    df = df.reset_index(drop=True).assign(group=df[group_col].to_numpy())
    groups = list(df["group"].unique())
    masks = [(df["group"] == g).to_numpy() for g in groups]
    frames = [df] * len(masks)
    if fold_frame is not None:
        frames = [_aligned(fold_frame(frozenset(df.loc[m, "grade"])), df) for m in masks]
    jobs = [
        (make, mask, frame) for make in factories for mask, frame in zip(masks, frames, strict=True)
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fold)(make, frame[~mask], frame[mask]) for make, mask, frame in jobs
    )
    parts = []
    for i in range(len(factories)):
        folds = results[i * len(masks) : (i + 1) * len(masks)]
        pred = np.full(len(df), np.nan)
        std = np.full(len(df), np.nan)
        for mask, (_, p, s) in zip(masks, folds, strict=True):
            pred[mask], std[mask] = p, s
        parts.append(df.assign(model=folds[0][0], pred=pred, std=std))
    return CVResult(pd.concat(parts, ignore_index=True), n_boot=n_boot, seed=seed)


def _aligned(frame: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    if not frame["curve_id"].equals(df["curve_id"]) or not frame["hrc"].equals(df["hrc"]):
        raise ValueError("fold_frame must return the rows of df in the same order")
    return frame


def _mae(err: npt.ArrayLike) -> float:
    return float(np.mean(np.abs(err)))


def _abs_error(rows: pd.DataFrame) -> pd.Series:
    return (rows["pred"] - rows["hrc"]).abs()


def _rmse(err: npt.ArrayLike) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def cluster_bootstrap(
    errors: pd.Series,
    groups: pd.Series,
    stat: Callable[[FloatArray], float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    per_group: bool = False,
) -> Interval:
    """Percentile 95 % interval, resampling whole groups with replacement.

    Points of the same grade are strongly correlated, so resampling points would
    understate the uncertainty. With ``per_group`` the statistic is computed for each
    group and averaged, so every grade counts equally regardless of its number of
    points; otherwise every point counts equally.
    """
    rng = np.random.default_rng(seed)
    by_group = [errors[groups == g].to_numpy() for g in groups.unique()]
    k = len(by_group)

    def estimate(parts: list[FloatArray]) -> float:
        if per_group:
            return float(np.mean([stat(part) for part in parts]))
        return stat(np.concatenate(parts))

    samples = [estimate([by_group[i] for i in rng.integers(0, k, size=k)]) for _ in range(n_boot)]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return Interval(estimate(by_group), float(lo), float(hi))


@dataclass(frozen=True)
class CVResult:
    """Cross-validated predictions of several models on the same rows.

    ``frame`` has the input columns plus ``model`` (predictor key), ``pred``, ``std``
    and ``group`` (the held-out unit). Every statistic of the study is a method here,
    so callers never compute errors from the columns themselves.
    """

    frame: pd.DataFrame
    n_boot: int = 2000
    seed: int = 0

    @property
    def n_groups(self) -> int:
        return int(self.frame["group"].nunique())

    def subset(self, model: str | None = None, kind: CurveKind | None = None) -> pd.DataFrame:
        """Rows of one model and/or one kind of curve, in the original row order."""
        rows = self.frame
        if model is not None:
            rows = rows[rows["model"] == model]
        if kind == "band":
            rows = rows[rows["kind"] != "heat"]
        elif kind is not None:
            rows = rows[rows["kind"] == kind]
        return rows.reset_index(drop=True)

    def errors(self, model: str, kind: CurveKind | None = None) -> pd.Series:
        """Signed errors (prediction minus measurement) of one model."""
        rows = self.subset(model, kind)
        return rows["pred"] - rows["hrc"]

    def mae(self, model: str, kind: CurveKind | None = None) -> float:
        return _mae(self.errors(model, kind))

    def mae_per_group(self) -> pd.DataFrame:
        """MAE of every model (columns) on every held-out group (rows)."""
        ae = self.frame.assign(ae=_abs_error(self.frame))
        return ae.groupby(["group", "model"])["ae"].mean().unstack()

    def mae_by_distance(self, model: str, bins: Sequence[float]) -> pd.Series:
        rows = self.subset(model)
        return (
            _abs_error(rows).groupby(pd.cut(rows["distance_mm"], list(bins)), observed=False).mean()
        )

    def mae_near_far(self, model: str, near_mm: float = 3.0) -> tuple[float, float]:
        """MAE up to ``near_mm`` from the quenched end and beyond it."""
        rows = self.subset(model)
        ae = _abs_error(rows)
        near = rows["distance_mm"] <= near_mm
        return float(ae[near].mean()), float(ae[~near].mean())

    def summary(self) -> pd.DataFrame:
        """MAE and RMSE per model with bootstrap intervals, per point and per group."""
        out = []
        for model in self.frame["model"].unique():
            rows = self.subset(model)
            err = rows["pred"] - rows["hrc"]
            mae = cluster_bootstrap(err, rows["group"], _mae, n_boot=self.n_boot, seed=self.seed)
            mae_grade = cluster_bootstrap(
                err, rows["group"], _mae, n_boot=self.n_boot, seed=self.seed, per_group=True
            )
            out.append(
                {
                    "model": model,
                    "mae": mae.estimate,
                    "mae_low": mae.low,
                    "mae_high": mae.high,
                    "mae_grade": mae_grade.estimate,
                    "mae_grade_low": mae_grade.low,
                    "mae_grade_high": mae_grade.high,
                    "rmse": _rmse(err),
                    "bias": float(err.mean()),
                    "mae_heats": self.mae(model, "heat"),
                    "mae_bands": self.mae(model, "band"),
                }
            )
        return pd.DataFrame(out).set_index("model")

    def paired_difference(
        self, a: str, b: str, *, per_group: bool = False, kind: CurveKind | None = None
    ) -> Interval:
        """Bootstrap interval of MAE(a) - MAE(b) on identical test points."""
        rows_a, rows_b = self.subset(a, kind), self.subset(b, kind)
        diff = _abs_error(rows_a) - _abs_error(rows_b)
        return cluster_bootstrap(
            diff,
            rows_a["group"],
            lambda d: float(np.mean(d)),
            n_boot=self.n_boot,
            seed=self.seed,
            per_group=per_group,
        )

    def group_differences(self, a: str, b: str) -> FloatArray:
        """Per-group MAE(a) - MAE(b), one value per held-out group."""
        per_group = self.mae_per_group()
        return np.asarray(per_group[a] - per_group[b], dtype=np.float64)

    def coverage(self, model: str, kind: CurveKind | None = None, z: float = 1.96) -> float:
        """Share of points inside the central 95 % predictive interval of ``model``."""
        rows = self.subset(model, kind).dropna(subset=["std"])
        inside = (rows["hrc"] - rows["pred"]).abs() <= z * rows["std"]
        return float(inside.mean())


def sign_flip_pvalue(differences: FloatArray) -> float:
    """Exact two-sided sign-flip test of a zero mean over all 2^k sign patterns.

    Under the null hypothesis each grade's difference is equally likely to have either
    sign, which makes the test exact for the k held-out grades.
    """
    k = differences.size
    signs = ((np.arange(2**k)[:, None] >> np.arange(k)) & 1) * 2 - 1
    means = np.abs((signs * differences).mean(axis=1))
    return float(np.mean(means >= abs(differences.mean()) - 1e-12))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for a family of pre-specified comparisons."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        adjusted[name] = running
    return adjusted
