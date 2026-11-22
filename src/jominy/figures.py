"""Figures. Each function takes data and returns a Figure; ``apply_style`` sets the
shared matplotlib style once.

Colours are assigned to entities in a fixed order, so a model keeps its colour in
every figure. Axis labels are German.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jominy.explain import FAR_MM, NEAR_MM

FloatArray = npt.NDArray[np.float64]

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

#: Fixed colour per model key.
MODEL_COLORS = {"Cal": ORANGE, "Lin": AQUA, "Gbdt": YELLOW, "Hyb": BLUE, "Phys": VIOLET}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
            "font.family": "serif",
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "legend.frameon": False,
        }
    )


class _DecimalComma(mpl.ticker.ScalarFormatter):
    """Tick labels with a decimal comma, as used in German technical writing."""

    def __call__(self, x: float, pos: int | None = None) -> str:
        return super().__call__(x, pos).replace(".", ",")


def _german_ticks(fig: Figure) -> Figure:
    """Decimal comma on every linear axis of ``fig``."""
    for ax in fig.axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            numeric = isinstance(axis.get_major_formatter(), mpl.ticker.ScalarFormatter)
            if scale == "linear" and numeric:
                axis.set_major_formatter(_DecimalComma())
    return fig


def _legend_below(fig: Figure, ax: Axes, ncol: int = 3) -> None:
    """One shared legend in a strip under the panels, clear of axis labels."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=ncol, fontsize=8, bbox_to_anchor=(0.5, -0.04)
    )


def cooling(distance_mm: FloatArray, rates: Mapping[float, FloatArray]) -> Figure:
    """Cooling rate at 700 degC along the bar for several heat transfer coefficients."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for (h, rate), color in zip(rates.items(), (BLUE, ORANGE, AQUA, YELLOW), strict=True):
        ax.plot(distance_mm, rate, color=color, label=f"$h$ = {h:,.0f}".replace(",", "\u2009"))
    ax.set_yscale("log")
    ax.set_xlabel("Abstand von der Stirnfläche $E$ / mm")
    ax.set_ylabel("Abkühlrate bei 700 °C / K s$^{-1}$")
    ax.legend(title="W m$^{-2}$ K$^{-1}$", fontsize=8)
    return _german_ticks(fig)


def phases(distance_mm: FloatArray, fractions: Mapping[str, FloatArray]) -> Figure:
    """Stacked martensite / bainite / ferrite-pearlite fractions for each named steel."""
    fig, axes = plt.subplots(1, len(fractions), figsize=(6.3, 2.6), sharey=True)
    for ax, (title, f) in zip(axes, fractions.items(), strict=True):
        ax.stackplot(
            distance_mm,
            f.T * 100,
            colors=(BLUE, ORANGE, AQUA),
            edgecolor="white",
            linewidth=0.8,
            labels=("Martensit", "Bainit", "Ferrit-Perlit"),
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("$E$ / mm")
        ax.set_xlim(distance_mm[0], distance_mm[-1])
        ax.set_ylim(0, 100)
        ax.grid(False)
    axes[0].set_ylabel("Volumenanteil / %")
    _legend_below(fig, axes[0])
    return _german_ticks(fig)


def sensitivity(errors: Mapping[float, float]) -> Figure:
    """Physics-chain MAE against the heat transfer coefficient."""
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(list(errors), list(errors.values()), "o-", color=BLUE)
    ax.set_xscale("log")
    ticks = [2.5e3, 5e3, 1e4, 2e4, 5e4]
    ax.set_xticks(ticks, [f"{t:,.0f}".replace(",", "\u2009") for t in ticks])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Wärmeübergangskoeffizient $h$ / W m$^{-2}$ K$^{-1}$")
    ax.set_ylabel("MAE der Physikkette / HRC")
    return _german_ticks(fig)


def _measured_vs_predicted(
    ax: Axes,
    g: pd.DataFrame,
    models: Mapping[str, str],
    *,
    measured_style: str,
    predicted_style: str,
    measured_label: str | None,
) -> None:
    """Measured curve once, then each model's prediction; labels only if requested."""
    first = g[g["model"] == next(iter(models))]
    ax.plot(
        first["distance_mm"],
        first["hrc"],
        measured_style,
        color=INK,
        ms=3.5,
        lw=1.0,
        label=measured_label,
    )
    for key, name in models.items():
        m = g[g["model"] == key]
        ax.plot(
            m["distance_mm"],
            m["pred"],
            predicted_style,
            color=MODEL_COLORS[key],
            label=name if measured_label else None,
        )


def heats(cv: pd.DataFrame, models: Mapping[str, str]) -> Figure:
    """Measured heats against cross-validated predictions, one panel per heat."""
    data = cv[cv["kind"] == "heat"]
    ids = list(dict.fromkeys(data["curve_id"]))
    fig, axes = plt.subplots(1, len(ids), figsize=(6.6, 2.4), sharey=True)
    for ax, cid in zip(axes, ids, strict=True):
        g = data[data["curve_id"] == cid]
        _measured_vs_predicted(
            ax, g, models, measured_style="o", predicted_style="-", measured_label="Messung"
        )
        ax.set_title(g["grade"].iloc[0], fontsize=8.5)
        ax.set_xlabel("$E$ / mm")
    axes[0].set_ylabel("Härte / HRC")
    _legend_below(fig, axes[0])
    return _german_ticks(fig)


def bands(
    cv: pd.DataFrame, models: Mapping[str, str], variants: Sequence[tuple[str, str]]
) -> Figure:
    """Hardenability bands against predicted limit curves of the lean and rich corners."""
    fig, axes = plt.subplots(1, len(variants), figsize=(6.6, 2.4), sharey=True)
    for ax, (grade, variant) in zip(axes, variants, strict=True):
        g = cv[(cv["grade"] == grade) & (cv["variant"] == variant)]
        for kind in ("band_max", "band_min"):
            _measured_vs_predicted(
                ax,
                g[g["kind"] == kind],
                models,
                measured_style="-",
                predicted_style="--",
                measured_label="Streuband" if kind == "band_max" else None,
            )
        ax.set_title(f"{grade}\n{variant}", fontsize=8)
        ax.set_xlabel("$E$ / mm")
    axes[0].set_ylabel("Härte / HRC")
    _legend_below(fig, axes[0])
    return _german_ticks(fig)


#: Distance bins (mm) and their labels for the error-by-distance figure.
DISTANCE_BINS = (0, 4, 8, 12, 17, 27, 37, 51)
DISTANCE_LABELS = ("≤ 3", "5–7", "9–11", "13–15", "20–25", "30–35", "40–50")


def error_by_distance(errors: Mapping[str, tuple[str, pd.Series]]) -> Figure:
    """MAE per distance bin; ``errors`` maps a model key to its name and binned MAE."""
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(len(DISTANCE_LABELS))
    for key, (name, mae) in errors.items():
        ax.plot(x, mae.to_numpy(), "o-", color=MODEL_COLORS[key], label=name)
    ax.set_xticks(x, DISTANCE_LABELS)
    ax.set_xlabel("Abstand $E$ / mm")
    ax.set_ylabel("MAE / HRC")
    ax.legend(fontsize=7.5, ncol=2)
    return _german_ticks(fig)


def shap_bars(panels: Mapping[str, pd.DataFrame]) -> Figure:
    """Mean |SHAP| per feature near and far from the quenched end, one panel per model."""
    features = ["C", "Mn", "Cr", "Ni", "Mo", "Si", "distance_mm"]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4, 2.6), sharey=True)
    y = np.arange(len(features))
    for ax, (title, s) in zip(axes, panels.items(), strict=True):
        for offset, pos, color, label in (
            (-0.2, "near", BLUE, f"$E$ ≤ {NEAR_MM:.0f} mm"),
            (0.2, "far", ORANGE, f"$E$ ≥ {FAR_MM:.0f} mm"),
        ):
            ax.barh(y + offset, s.loc[features, pos], height=0.38, color=color, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("mittlerer |SHAP-Wert| / HRC")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f if f != "distance_mm" else "E" for f in features])
    axes[0].invert_yaxis()
    _legend_below(fig, axes[0], ncol=2)
    return _german_ticks(fig)


def parity(reference: FloatArray, predicted: FloatArray) -> Figure:
    """Surrogate against the physics chain it imitates."""
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.plot(reference, predicted, ".", color=BLUE, ms=1.5, alpha=0.5)
    lim = (float(reference.min()) - 2, float(reference.max()) + 2)
    ax.plot(lim, lim, color=INK_2, lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Physikkette / HRC")
    ax.set_ylabel("Surrogat / HRC")
    return _german_ticks(fig)
