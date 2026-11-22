"""SHAP attributions of tree models, split by position on the Jominy bar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

from jominy.models import FEATURES

#: Positions near the quenched end (martensite) and far from it (mixed microstructure).
NEAR_MM = 5.0
FAR_MM = 15.0


def mean_abs_shap(model: HistGradientBoostingRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature for points near and far from the quenched end."""
    values = shap.TreeExplainer(model).shap_values(X[FEATURES])
    sv = pd.DataFrame(np.abs(values), columns=FEATURES, index=X.index)
    near = X["distance_mm"] <= NEAR_MM
    far = X["distance_mm"] >= FAR_MM
    return pd.DataFrame({"near": sv[near].mean(), "far": sv[far].mean()})
