"""Transient heat conduction in a Jominy end-quench specimen.

The bar (DIN EN ISO 642: diameter 25 mm, length 100 mm) is modelled as a 1D fin
along its axis z. Radial temperature gradients are neglected; with lateral heat
transfer coefficients of the order of 10^2 W/(m^2 K) the radial Biot number is
below 0.05, so the bar cross-section is nearly isothermal. The water jet at z = 0
is a Robin boundary; the lateral surface and the far end lose heat to still air
by natural convection and radiation.

Governing equation (implicit Euler, conservative finite volumes):

    rho cp(T) dT/dt = d/dz( k(T) dT/dz ) - (4/d) h_air(T) (T - T_amb)

Thermal properties of carbon steel follow EN 1993-1-2 (Eurocode 3, fire part),
section 3.4.1. The latent-heat peak that EN 1993-1-2 places in cp between 600 and
900 degC describes equilibrium pearlite formation in slowly heated steel and is
deliberately omitted: in a quenched specimen austenite is undercooled through this
range and transforms (if at all) at lower temperatures. This is a modelling
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_banded

FloatArray = npt.NDArray[np.float64]

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2 K^4)
KELVIN = 273.15


def conductivity(T: FloatArray) -> FloatArray:
    """Thermal conductivity in W/(m K) (EN 1993-1-2, section 3.4.1)."""
    return np.where(T < 800.0, 54.0 - 3.33e-2 * T, 27.3)


def specific_heat(T: FloatArray) -> FloatArray:
    """Specific heat in J/(kg K): EN 1993-1-2 (section 3.4.1) below 600 degC, then a linear
    bridge to the high-temperature value 650 J/(kg K) at 900 degC (no latent-heat peak)."""
    Tc = np.clip(T, 20.0, None)

    def cubic(x: FloatArray | float) -> FloatArray:
        return np.asarray(425.0 + 7.73e-1 * x - 1.69e-3 * x**2 + 2.22e-6 * x**3)

    cp600 = cubic(600.0)
    bridge = cp600 + (650.0 - cp600) * (Tc - 600.0) / 300.0
    return np.where(Tc < 600.0, cubic(Tc), np.where(Tc < 900.0, bridge, 650.0))


DENSITY = 7850.0  # kg/m^3, unit mass of steel in EN 1993-1-2


@dataclass(frozen=True, slots=True)
class JominySetup:
    """Geometry, initial state and boundary conditions of the end-quench test."""

    length: float = 0.100  # m
    diameter: float = 0.025  # m
    T_austenitize: float = 850.0  # degC
    T_water: float = 20.0  # degC
    T_ambient: float = 20.0  # degC
    h_end: float = 1.0e4  # W/(m^2 K), water jet on the quenched face
    conductivity_scale: float = 1.0  # multiplies lambda(T); for sensitivity studies
    h_air_convection: float = 10.0  # W/(m^2 K), natural convection
    emissivity: float = 0.8
    n_nodes: int = 401
    t_end: float = 900.0  # s
    dt_initial: float = 1.0e-3  # s
    dt_max: float = 0.5  # s
    dt_growth: float = 1.02


@dataclass(frozen=True)
class CoolingHistory:
    """Temperature field T[time, node] on the axial grid z (m)."""

    z: FloatArray
    t: FloatArray
    T: FloatArray = field(repr=False)

    def cooling_rate_at(self, temperature: float = 700.0) -> FloatArray:
        """Cooling rate -dT/dt in K/s at every node when it passes ``temperature``.

        Nodes that never reach the temperature within the simulated time get NaN.
        """
        rates = np.full(self.z.size, np.nan)
        for j in range(self.z.size):
            Tj = self.T[:, j]
            below = np.flatnonzero(Tj <= temperature)
            if below.size == 0 or below[0] == 0:
                continue
            i = below[0]
            rates[j] = (Tj[i - 1] - Tj[i]) / (self.t[i] - self.t[i - 1])
        return rates

    def rate_at_distance(self, distance_mm: FloatArray, temperature: float = 700.0) -> FloatArray:
        """Cooling rate (K/s) at arbitrary distances from the quenched end (mm)."""
        return np.interp(np.asarray(distance_mm) * 1e-3, self.z, self.cooling_rate_at(temperature))


def _h_air(T: FloatArray, setup: JominySetup) -> FloatArray:
    """Combined convection + linearised radiation coefficient to ambient."""
    Tk = T + KELVIN
    Ta = setup.T_ambient + KELVIN
    h_rad = setup.emissivity * STEFAN_BOLTZMANN * (Tk**2 + Ta**2) * (Tk + Ta)
    return setup.h_air_convection + h_rad


def simulate(setup: JominySetup = JominySetup()) -> CoolingHistory:
    """Solve the end-quench cooling problem and return the full temperature history."""
    n = setup.n_nodes
    dz = setup.length / (n - 1)
    z = np.linspace(0.0, setup.length, n)
    # Finite-volume widths: half cells at both ends.
    width = np.full(n, dz)
    width[0] = width[-1] = dz / 2
    lateral_per_volume = 4.0 / setup.diameter  # perimeter / cross-section area

    T = np.full(n, setup.T_austenitize)
    times = [0.0]
    history = [T.copy()]
    t, dt = 0.0, setup.dt_initial
    while t < setup.t_end:
        dt = min(dt, setup.t_end - t)
        # Properties frozen at the old time level (semi-implicit linearisation).
        k_node = setup.conductivity_scale * conductivity(T)
        k_face = 2 * k_node[:-1] * k_node[1:] / (k_node[:-1] + k_node[1:])
        cap = DENSITY * specific_heat(T) * width / dt  # J/(m^2 K) per unit area
        h_lat = _h_air(T, setup) * lateral_per_volume * width
        g = k_face / dz

        diag = cap + h_lat
        diag[:-1] += g
        diag[1:] += g
        rhs = cap * T + h_lat * setup.T_ambient
        # Quenched face (z = 0) and free end (z = L).
        diag[0] += setup.h_end
        rhs[0] += setup.h_end * setup.T_water
        h_far = float(_h_air(T[-1:], setup)[0])
        diag[-1] += h_far
        rhs[-1] += h_far * setup.T_ambient

        banded = np.zeros((3, n))
        banded[0, 1:] = -g
        banded[1] = diag
        banded[2, :-1] = -g
        T = solve_banded((1, 1), banded, rhs)

        t += dt
        times.append(t)
        history.append(T.copy())
        dt = min(dt * setup.dt_growth, setup.dt_max)

    return CoolingHistory(z=z, t=np.array(times), T=np.array(history))
