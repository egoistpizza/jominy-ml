"""Predictors that map (composition, distance) -> HRC behind one small interface.

Every predictor takes and returns long-format frames with the columns of
:data:`jominy.data.COLUMNS`. ``fit`` sees only training rows; nothing about the
held-out grade leaks into it. Each class carries a short ``key`` (stable, used in
file names and LaTeX macros) and a German display ``name`` (used in figures and tables).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jominy.materials import ELEMENTS, Composition
from jominy.physics import H_NOMINAL, PhysicsModel

FloatArray = npt.NDArray[np.float64]

#: Grid for calibrating the quench-end heat transfer coefficient, W/(m^2 K).
H_END_GRID = (2.5e3, 5e3, 7.5e3, 1e4, 1.5e4, 2e4, 3e4, 5e4)

#: Elements the learned models see. P, V and Cu are left out: the sources report them
#: inconsistently (missing in the heats, set only at the rich corner of a band), so
#: their values would tell a model which kind of curve it is looking at.
ML_ELEMENTS = ["C", "Si", "Mn", "Cr", "Ni", "Mo"]
FEATURES = [*ML_ELEMENTS, "distance_mm"]

#: One fixed tree configuration for every gradient-boosting model, so that the pure
#: data model and the hybrid correction differ only in what they learn, not in capacity.
#: The values were set once before the comparison and are not tuned on any fold.
TREE_PARAMS: dict[str, float | int | bool] = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_samples_leaf": 15,
    "l2_regularization": 5.0,
    "early_stopping": False,
}


def physics_predict(df: pd.DataFrame, h_end: float) -> FloatArray:
    """Physics-chain prediction for every row of ``df`` (one simulation per composition)."""
    model = PhysicsModel(h_end=h_end)
    out = np.empty(len(df))
    for _, idx in df.groupby(list(ELEMENTS), sort=False).groups.items():
        rows = df.loc[idx]
        first = rows.iloc[0]
        comp = Composition(**{e: float(first[e]) for e in ELEMENTS})
        out[df.index.get_indexer(idx)] = model.predict_curve(comp, rows["distance_mm"].to_numpy())
    return out


def h_end_errors(df: pd.DataFrame) -> dict[float, float]:
    """MAE of the physics chain on ``df`` for every value of :data:`H_END_GRID`."""
    return {h: float(np.mean(np.abs(physics_predict(df, h) - df["hrc"]))) for h in H_END_GRID}


def calibrate_h_end(df: pd.DataFrame) -> float:
    """Heat transfer coefficient from :data:`H_END_GRID` with the lowest MAE on ``df``."""
    errors = h_end_errors(df)
    return min(errors, key=errors.__getitem__)


class Predictor(ABC):
    key: ClassVar[str]
    name: ClassVar[str]

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Predictor: ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> FloatArray: ...

    def predict_with_std(self, df: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
        """Prediction and predictive standard deviation (NaN if the model has none)."""
        return self.predict(df), np.full(len(df), np.nan)


class Physics(Predictor):
    """Creusot-Loire chain with a fixed heat transfer coefficient (no training)."""

    key = "Phys"
    name = "Physik (h fest)"

    def __init__(self, h_end: float = H_NOMINAL) -> None:
        self.h_end = h_end

    def fit(self, train: pd.DataFrame) -> Physics:
        return self

    def predict(self, df: pd.DataFrame) -> FloatArray:
        return physics_predict(df, self.h_end)


class CalibratedPhysics(Physics):
    """Physics chain whose only free parameter h_end is fitted on the training grades."""

    key = "Cal"
    name = "Physik (h kalibriert)"

    def fit(self, train: pd.DataFrame) -> CalibratedPhysics:
        self.h_end = calibrate_h_end(train)
        return self


def _just_features(df: pd.DataFrame) -> FloatArray:
    """Regressors with the functional form of Just's hardenability formula: sqrt(C)
    for the martensite hardness, distance terms and their interaction with sqrt(C),
    and linear alloy terms. The coefficients are refitted, not taken from Just."""
    E = df["distance_mm"].to_numpy()
    sqrt_c = np.sqrt(df["C"].to_numpy())
    alloys = df[["Si", "Mn", "Cr", "Ni", "Mo"]].to_numpy()
    return np.column_stack([sqrt_c, E**2 * sqrt_c, np.sqrt(E), E, alloys])


class LinearRegression(Predictor):
    """Ridge regression on classical hardenability regressors."""

    key = "Lin"
    name = "Lineare Regression"

    def fit(self, train: pd.DataFrame) -> LinearRegression:
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 2, 11)))
        self.model.fit(_just_features(train), train["hrc"])
        return self

    def predict(self, df: pd.DataFrame) -> FloatArray:
        return np.asarray(self.model.predict(_just_features(df)), dtype=np.float64)


class DataGBDT(Predictor):
    """Gradient boosting on real data only; hardness constrained to fall with distance."""

    key = "Gbdt"
    name = "GBDT (nur Messdaten)"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _fit_trees(self, data: pd.DataFrame) -> None:
        self.model = HistGradientBoostingRegressor(
            **TREE_PARAMS,
            monotonic_cst=[0] * len(ML_ELEMENTS) + [-1],
            random_state=self.seed,
        ).fit(data[FEATURES], data["hrc"])

    def fit(self, train: pd.DataFrame) -> DataGBDT:
        self._fit_trees(train)
        return self

    def predict(self, df: pd.DataFrame) -> FloatArray:
        return np.asarray(self.model.predict(df[FEATURES]), dtype=np.float64)


class Surrogate(DataGBDT):
    """The same trees trained once on physics-generated curves; never sees real data."""

    key = "Surr"
    name = "GBDT-Surrogat (nur Simulation)"

    def __init__(self, synthetic: pd.DataFrame, seed: int = 0) -> None:
        super().__init__(seed)
        self._fit_trees(synthetic)

    def fit(self, train: pd.DataFrame) -> Surrogate:
        return self  # training data is fixed at construction


class _ResidualHybrid(Predictor):
    """Calibrated physics plus a learned correction of its residual."""

    def fit(self, train: pd.DataFrame) -> _ResidualHybrid:
        self.physics = CalibratedPhysics().fit(train)
        base = self.physics.predict(train)
        self._fit_correction(train, base, train["hrc"].to_numpy() - base)
        return self

    def predict_with_std(self, df: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
        base = self.physics.predict(df)
        correction, std = self._correct(df, base)
        return base + correction, std

    def predict(self, df: pd.DataFrame) -> FloatArray:
        return self.predict_with_std(df)[0]

    @abstractmethod
    def _fit_correction(self, df: pd.DataFrame, base: FloatArray, residual: FloatArray) -> None: ...

    @abstractmethod
    def _correct(self, df: pd.DataFrame, base: FloatArray) -> tuple[FloatArray, FloatArray]: ...


class HybridGBDT(_ResidualHybrid):
    """Residual learned by shallow, strongly regularised gradient boosting."""

    key = "Hyb"
    name = "Hybrid (Physik + GBDT)"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    @staticmethod
    def _features(df: pd.DataFrame, base: FloatArray) -> FloatArray:
        return np.column_stack([df[FEATURES].to_numpy(), base])

    def _fit_correction(self, df: pd.DataFrame, base: FloatArray, residual: FloatArray) -> None:
        self.model = HistGradientBoostingRegressor(**TREE_PARAMS, random_state=self.seed).fit(
            self._features(df, base), residual
        )

    def _correct(self, df: pd.DataFrame, base: FloatArray) -> tuple[FloatArray, FloatArray]:
        pred = np.asarray(self.model.predict(self._features(df, base)), dtype=np.float64)
        return pred, np.full(len(df), np.nan)


class HybridGP(_ResidualHybrid):
    """Residual learned by a Gaussian process, which also yields a predictive std."""

    key = "Gp"
    name = "Hybrid (Physik + GP)"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    @staticmethod
    def _features(df: pd.DataFrame) -> FloatArray:
        cols = ["C", "Mn", "Cr", "Ni", "Mo"]
        return np.column_stack([df[cols].to_numpy(), np.log(df["distance_mm"].to_numpy())])

    def _fit_correction(self, df: pd.DataFrame, base: FloatArray, residual: FloatArray) -> None:
        X = self._features(df)
        kernel = ConstantKernel(4.0, (1e-2, 1e3)) * RBF(
            length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-1, 1e2)
        ) + WhiteKernel(1.0, (1e-2, 1e2))
        self.scaler = StandardScaler().fit(X)
        self.model = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=self.seed
        ).fit(self.scaler.transform(X), residual)

    def _correct(self, df: pd.DataFrame, base: FloatArray) -> tuple[FloatArray, FloatArray]:
        mean, std = self.model.predict(self.scaler.transform(self._features(df)), return_std=True)
        return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


@dataclass(frozen=True)
class ModelSpec:
    """One compared model: its key, labels and how to build it for a study.

    ``build`` receives the study seed and the simulated training curves and returns a
    factory that the cross-validation calls once per fold. The surrogate trains once
    in ``build`` and hands out the same fitted instance, because it never sees real
    data.
    """

    key: str
    name: str
    short: str
    build: Callable[[int, pd.DataFrame], Callable[[], Predictor]]


def _surrogate_factory(seed: int, synthetic: pd.DataFrame) -> Callable[[], Predictor]:
    fitted = Surrogate(synthetic, seed=seed)
    return lambda: fitted


#: Every compared model, in table order.
ROSTER: tuple[ModelSpec, ...] = (
    ModelSpec(Physics.key, Physics.name, "Physik fest", lambda seed, syn: Physics),
    ModelSpec(
        CalibratedPhysics.key,
        CalibratedPhysics.name,
        "Physik kal.",
        lambda seed, syn: CalibratedPhysics,
    ),
    ModelSpec(
        LinearRegression.key, LinearRegression.name, "Lin. Reg.", lambda seed, syn: LinearRegression
    ),
    ModelSpec(DataGBDT.key, DataGBDT.name, "GBDT", lambda seed, syn: lambda: DataGBDT(seed)),
    ModelSpec(Surrogate.key, Surrogate.name, "Surrogat", _surrogate_factory),
    ModelSpec(
        HybridGBDT.key, HybridGBDT.name, "Hybrid", lambda seed, syn: lambda: HybridGBDT(seed)
    ),
    ModelSpec(HybridGP.key, HybridGP.name, "Hybrid GP", lambda seed, syn: lambda: HybridGP(seed)),
)
