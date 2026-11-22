"""Loading the real Jominy data into one tidy table.

Two kinds of curves are used:

* ``heat``: a measured Jominy curve of one cast with its analysed composition.
* ``band_min`` / ``band_max``: the lower and upper limit of a hardenability
  band (H-band). The lower limit is paired with the lean corner of the
  composition range (all elements at their minimum), the upper limit with the
  rich corner (all at their maximum). This pairing is an approximation: a band
  is the envelope of many casts, not the response of a single corner composition.

Every row carries the bibliography key of its source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jominy.materials import ELEMENTS, Composition

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

COLUMNS = [
    "curve_id",
    "source",
    "grade",
    "variant",
    "kind",
    *ELEMENTS,
    "distance_mm",
    "hrc",
]


@dataclass(frozen=True)
class Curve:
    curve_id: str
    source: str
    grade: str
    variant: str
    kind: str
    composition: Composition
    distance_mm: np.ndarray
    hrc: np.ndarray


def _split_band(x: list[float], y: list[float]) -> tuple[dict[float, float], dict[float, float]]:
    """Split a closed band polygon into upper and lower limits keyed by distance.

    The polygon runs along the upper limit to the largest distance and back along
    the lower limit. Where the source specifies no lower limit, the chart repeats
    the upper value at that distance; such points are dropped.
    """
    turn = int(np.argmax(x))
    upper = dict(zip(x[: turn + 1], y[: turn + 1], strict=True))
    lower: dict[float, float] = {}
    for xi, yi in zip(x[turn + 1 :], y[turn + 1 :], strict=True):
        if xi in upper and yi == upper[xi]:
            continue
        lower[xi] = yi
    return upper, lower


#: Elements that occur as residuals in every real heat. Where a band gives no lower
#: limit (or omits the element), the corner gets the residual level instead of zero;
#: a zero would mark the curve as a band corner, which no measured heat looks like.
RESIDUAL_ELEMENTS = ("Si", "Ni", "Mo", "Cu")


def _band_corners(
    ranges: dict[str, list[float | None]], residual: dict[str, float]
) -> tuple[Composition, Composition]:
    """Lean and rich corner of a composition range, with residual levels filled in."""
    lean: dict[str, float] = {}
    rich: dict[str, float] = {}
    for e, (lo, hi) in ranges.items():
        floor = residual.get(e, 0.0)
        lean[e] = lo if lo is not None else min(floor, hi if hi is not None else floor)
        rich[e] = hi if hi is not None else lean[e]
    for e, level in residual.items():
        lean.setdefault(e, level)
        rich.setdefault(e, level)
    return Composition.from_mapping(lean), Composition.from_mapping(rich)


def _band_curves(path: Path, residual: dict[str, float]) -> list[Curve]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    curves: list[Curve] = []
    for v in raw["variants"]:
        lean, rich = _band_corners(v["composition"], residual)
        upper, lower = _split_band(v["x"], v["y"])
        source = "ovako2026" if v["issuer"] == "Ovako" else "iso683"
        for kind, comp, limit in (("band_min", lean, lower), ("band_max", rich, upper)):
            d = np.array(sorted(limit))
            curves.append(
                Curve(
                    curve_id=f"{v['grade']} {v['variant']} {kind[-3:]}",
                    source=source,
                    grade=v["grade"],
                    variant=v["variant"],
                    kind=kind,
                    composition=comp,
                    distance_mm=d,
                    hrc=np.array([limit[k] for k in d]),
                )
            )
    return curves


def _heat_curves(path: Path) -> list[Curve]:
    df = pd.read_csv(path, comment="#")
    jcols = [c for c in df.columns if c.startswith("J")]
    distances = np.array([float(c[1:]) for c in jcols])
    curves = []
    for _, row in df.iterrows():
        comp = Composition.from_mapping(
            {e: float(row[e]) for e in ELEMENTS if e in row and pd.notna(row[e])}
        )
        curves.append(
            Curve(
                curve_id=str(row["heat"]),
                source="tomasic2025",
                grade=str(row["grade"]),
                variant="heat",
                kind="heat",
                composition=comp,
                distance_mm=distances,
                hrc=row[jcols].to_numpy(dtype=float),
            )
        )
    return curves


def residual_levels(heats: list[Curve]) -> dict[str, float]:
    """Lowest content of each residual element observed in the measured heats (wt%)."""
    return {e: min(getattr(h.composition, e) for h in heats) for e in RESIDUAL_ELEMENTS}


def load_curves(data_dir: Path = DATA_DIR, held_out: frozenset[str] = frozenset()) -> list[Curve]:
    """All real curves: measured heats first, then hardenability-band limits.

    The residual levels of the band corners are taken from the heats whose grade is
    not in ``held_out``, so that a cross-validation fold can build its band
    compositions without the composition of a held-out heat.
    """
    heats = _heat_curves(data_dir / "tomasic_2025_heats.csv")
    levels = residual_levels([h for h in heats if h.grade not in held_out])
    bands = _band_curves(data_dir / "ovako_steelnavigator_2026-11-22.json", levels)
    return heats + bands


def to_frame(curves: list[Curve]) -> pd.DataFrame:
    """Long format: one row per (curve, distance)."""
    rows = []
    for c in curves:
        comp = c.composition.as_dict()
        for d, h in zip(c.distance_mm, c.hrc, strict=True):
            rows.append(
                {
                    "curve_id": c.curve_id,
                    "source": c.source,
                    "grade": c.grade,
                    "variant": c.variant,
                    "kind": c.kind,
                    **comp,
                    "distance_mm": float(d),
                    "hrc": float(h),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)
