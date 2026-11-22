"""Hardness after continuous cooling: Creusot-Loire model (Maynier et al.).

All regression equations are taken verbatim from the compilation in
J. Trzaska, Archives of Materials Science and Engineering 61/2 (2013) 87-92,
equations (1)-(13), which reproduces Maynier, Dollet & Bastien (1978) and
Maynier, Jungmann & Dollet (1978).

* Critical cooling rates v_i (degC/h, at 700 degC) bound the microstructure:
  v_M (100 % martensite), v_M90, v_M50, v_B (100 % bainite), v_B90, v_B50,
  v_FP90 (90 % ferrite-pearlite) and v_FP (100 % ferrite-pearlite).
* Each constituent has its own hardness equation in the cooling rate v_R.
* The total hardness is the volume-fraction weighted mean.

Two choices are ours, not part of the published method:

1. Phase fractions between the tabulated critical rates are interpolated
   linearly in log10(v). A critical rate that crosses its faster neighbour is
   dropped (see :func:`_consistent_anchors`).
2. "Other constituents" at v_M90 and v_M50 are taken as bainite, and at v_B90
   and v_B50 as ferrite-pearlite.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from jominy.materials import Composition

FloatArray = npt.NDArray[np.float64]

GAS_CONSTANT = 8.314  # J/(mol K)
ACTIVATION_ENERGY = 460.55e3  # J/mol, grain growth in low-alloy steels (Trzaska 2013)


def austenitizing_parameter(T_celsius: float, hours: float) -> float:
    """Austenitizing parameter P_A (Trzaska 2013, eq. 9).

    P_A = [1/T - (n R / dH) log10(t / t0)]^-1 with n = ln 10 and t0 = 1 h.
    The source prints the unit as degC; the bracket yields kelvin, so we convert
    to degC (P_A then equals the austenitizing temperature for a 1 h hold).
    """
    T = T_celsius + 273.15
    inv = 1.0 / T - (math.log(10) * GAS_CONSTANT / ACTIVATION_ENERGY) * math.log10(hours)
    return 1.0 / inv - 273.15


def critical_log_rates(c: Composition, PA: float) -> dict[str, float]:
    """log10 of the critical cooling rates in degC/h (Trzaska 2013, eqs. 1-8)."""
    C, Mn, Cr, Mo, Ni = c.C, c.Mn, c.Cr, c.Mo, c.Ni
    return {
        "M": 9.81 - (4.62 * C + 1.05 * Mn + 0.5 * Cr + 0.66 * Mo + 0.54 * Ni + 0.00183 * PA),
        "M90": 8.76 - (4.04 * C + 0.96 * Mn + 0.58 * Cr + 0.97 * Mo + 0.49 * Ni + 0.001 * PA),
        "M50": 8.50 - (4.13 * C + 0.86 * Mn + 0.41 * Cr + 0.94 * Mo + 0.57 * Ni + 0.0012 * PA),
        "B": 10.17 - (3.8 * C + 1.07 * Mn + 0.57 * Cr + 1.58 * Mo + 0.7 * Ni + 0.0032 * PA),
        "B90": 10.55 - (3.65 * C + 1.08 * Mn + 0.61 * Cr + 1.49 * Mo + 0.77 * Ni + 0.0032 * PA),
        "B50": 8.74 - (2.23 * C + 0.86 * Mn + 0.59 * Cr + 1.60 * Mo + 0.56 * Ni + 0.0032 * PA),
        "FP90": 7.51 - (1.38 * C + 0.35 * Mn + 0.11 * Cr + 2.31 * Mo + 0.93 * Ni + 0.0033 * PA),
        "FP": 6.36
        - (
            0.43 * C
            + 0.49 * Mn
            + 0.26 * Cr
            + 0.38 * Mo
            + 2 * math.sqrt(Mo)
            + 0.78 * Ni
            + 0.0019 * PA
        ),
    }


# (martensite, bainite, ferrite-pearlite) fractions at each critical rate, fastest first.
_ANCHORS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("M", (1.0, 0.0, 0.0)),
    ("M90", (0.9, 0.1, 0.0)),
    ("M50", (0.5, 0.5, 0.0)),
    ("B", (0.0, 1.0, 0.0)),
    ("B90", (0.0, 0.9, 0.1)),
    ("B50", (0.0, 0.5, 0.5)),
    ("FP90", (0.0, 0.1, 0.9)),
    ("FP", (0.0, 0.0, 1.0)),
)


#: What the non-martensitic share at v_M90 and v_M50 is taken to be. The source leaves
#: this open; bainite suits alloyed steels, ferrite-pearlite suits plain carbon steels.
MartensiteRest = Literal["bainite", "ferrite_pearlite"]


def phase_fractions(
    c: Composition, log_vR: FloatArray, PA: float, rest: MartensiteRest = "bainite"
) -> FloatArray:
    """Volume fractions (M, B, FP) for each cooling rate; shape (n, 3)."""
    log_v, fractions = _consistent_anchors(critical_log_rates(c, PA), rest)
    # np.interp needs increasing abscissae; anchors are ordered fastest-first.
    out = np.empty((np.size(log_vR), 3))
    for k in range(3):
        out[:, k] = np.interp(log_vR, log_v[::-1], fractions[::-1, k])
    return out


def _anchors(rest: MartensiteRest) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if rest == "bainite":
        return _ANCHORS
    swap = {"M90": (0.9, 0.0, 0.1), "M50": (0.5, 0.0, 0.5)}
    return tuple((name, swap.get(name, f)) for name, f in _ANCHORS)


def _consistent_anchors(
    crit: dict[str, float], rest: MartensiteRest = "bainite"
) -> tuple[FloatArray, FloatArray]:
    """Anchors whose critical rates fall strictly from fastest to slowest.

    For every low-alloy composition, log v_B90 lies above log v_B (their difference is
    0.38 + 0.15 C - 0.01 Mn - 0.04 Cr + 0.09 Mo - 0.07 Ni). A crossing anchor contradicts
    its faster neighbour, so it is dropped and the fractions are interpolated between
    the remaining anchors. This keeps the phase fractions continuous in the cooling rate.
    """
    kept = set(_kept_anchor_names(crit))
    rows = [(crit[name], f) for name, f in _anchors(rest) if name in kept]
    return np.array([v for v, _ in rows]), np.array([f for _, f in rows])


def _kept_anchor_names(crit: dict[str, float]) -> list[str]:
    """Anchor names, fastest first, keeping each only if its rate is below the last kept."""
    kept: list[str] = []
    last = math.inf
    for name, _ in _ANCHORS:
        if crit[name] < last:
            kept.append(name)
            last = crit[name]
    return kept


def dropped_anchors(c: Composition, PA: float) -> list[str]:
    """Names of critical rates discarded by :func:`phase_fractions` for ``c``."""
    kept = _kept_anchor_names(critical_log_rates(c, PA))
    return [name for name, _ in _ANCHORS if name not in kept]


def _constituent_hardness(c: Composition, log_vR: FloatArray) -> FloatArray:
    """Vickers hardness of martensite, bainite, ferrite-pearlite (Trzaska 2013, eqs. 10-12)."""
    C, Si, Mn, Cr, Ni, Mo, V = c.C, c.Si, c.Mn, c.Cr, c.Ni, c.Mo, c.V
    hv_m = 127 + 949 * C + 27 * Si + 11 * Mn + 16 * Cr + 8 * Ni + 21 * log_vR
    hv_b = (
        -323
        + 185 * C
        + 330 * Si
        + 153 * Mn
        + 144 * Cr
        + 191 * Mo
        + 65 * Ni
        + log_vR * (89 + 53 * C - 55 * Si - 22 * Mn - 20 * Cr - 33 * Mo - 10 * Ni)
    )
    hv_fp = (
        42
        + 223 * C
        + 53 * Si
        + 30 * Mn
        + 7 * Cr
        + 19 * Mo
        + 12.6 * Ni
        + log_vR * (10 - 19 * Si + 8 * Cr + 4 * Ni + 130 * V)
    )
    return np.column_stack([hv_m, hv_b, hv_fp])


def log_rate_per_hour(rate_K_per_s: FloatArray) -> FloatArray:
    """log10 of a cooling rate given in K/s, converted to the degC/h the equations use."""
    out: FloatArray = np.log10(np.asarray(rate_K_per_s, dtype=np.float64) * 3600.0)
    return out


def hardness_hv(
    c: Composition, rate_K_per_s: FloatArray, PA: float, rest: MartensiteRest = "bainite"
) -> FloatArray:
    """Mixture hardness HV for cooling rates at 700 degC given in K/s."""
    log_vR = log_rate_per_hour(rate_K_per_s)
    fractions = phase_fractions(c, log_vR, PA, rest)
    return np.sum(fractions * _constituent_hardness(c, log_vR), axis=1)


# Approximate HV -> HRC conversion for non-austenitic steels (ASTM E140, Table 1).
_HV_TABLE = np.array([238.0, 266.0, 302.0, 345.0, 392.0, 446.0, 513.0, 595.0, 697.0, 832.0, 940.0])
_HRC_TABLE = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 68.0])


def hv_to_hrc(hv: FloatArray) -> FloatArray:
    """Convert Vickers to Rockwell C by piecewise-linear interpolation of ASTM E140.

    Below 238 HV (20 HRC, the lower limit of the Rockwell C scale) the first
    segment is extrapolated linearly so that soft positions on the Jominy bar can
    still be compared with data reported on the HRC scale.
    """
    hv = np.asarray(hv, dtype=np.float64)
    slope = (_HRC_TABLE[1] - _HRC_TABLE[0]) / (_HV_TABLE[1] - _HV_TABLE[0])
    low = _HRC_TABLE[0] + slope * (hv - _HV_TABLE[0])
    hrc: FloatArray = np.where(hv < _HV_TABLE[0], low, np.interp(hv, _HV_TABLE, _HRC_TABLE))
    return hrc
