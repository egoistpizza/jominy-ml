"""Physics chain: Jominy cooling simulation -> Creusot-Loire hardness -> HRC."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from jominy.hardness import MartensiteRest, austenitizing_parameter, hardness_hv, hv_to_hrc
from jominy.materials import Composition
from jominy.thermal import JominySetup, simulate

FloatArray = npt.NDArray[np.float64]

#: Austenitizing condition assumed for all steels (30 min hold).
T_AUSTENITIZE = 850.0
HOLD_HOURS = 0.5
PA = austenitizing_parameter(T_AUSTENITIZE, HOLD_HOURS)

#: Nominal heat transfer coefficient of the water jet, W/(m^2 K).
H_NOMINAL = 1.0e4

#: Fine grid on which cooling rates are tabulated once per heat transfer coefficient.
_GRID_MM = np.linspace(0.0, 60.0, 241)


@lru_cache(maxsize=32)
def _rate_table(h_end: float, conductivity_scale: float) -> tuple[float, ...]:
    setup = JominySetup(
        h_end=h_end, T_austenitize=T_AUSTENITIZE, conductivity_scale=conductivity_scale
    )
    return tuple(simulate(setup).rate_at_distance(_GRID_MM).tolist())


def cooling_rate(
    distance_mm: FloatArray, h_end: float, conductivity_scale: float = 1.0
) -> FloatArray:
    """Cooling rate at 700 degC (K/s) at the given distances from the quenched end."""
    table = np.array(_rate_table(float(h_end), float(conductivity_scale)))
    rates: FloatArray = np.interp(distance_mm, _GRID_MM, table)
    return rates


@dataclass(frozen=True)
class PhysicsModel:
    """Composition + distance -> HRC without any fitted parameter except ``h_end``."""

    h_end: float = H_NOMINAL
    rest: MartensiteRest = "bainite"

    def predict_curve(self, c: Composition, distance_mm: FloatArray) -> FloatArray:
        rate = cooling_rate(np.asarray(distance_mm, dtype=np.float64), self.h_end)
        return hv_to_hrc(hardness_hv(c, rate, PA, self.rest))
