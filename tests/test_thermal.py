import numpy as np
import pytest
from scipy.special import erf

from jominy import thermal
from jominy.thermal import JominySetup, simulate


def test_matches_semi_infinite_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constant properties, fixed face temperature and no lateral loss reduce the
    problem to the classical semi-infinite solid, T = Ts + (Ti - Ts) erf(z / 2 sqrt(a t))."""
    k, cp = 30.0, 600.0
    monkeypatch.setattr(thermal, "conductivity", lambda T: np.full_like(T, k))
    monkeypatch.setattr(thermal, "specific_heat", lambda T: np.full_like(T, cp))
    setup = JominySetup(
        h_end=1e9, h_air_convection=0.0, emissivity=0.0, t_end=20.0, n_nodes=801, dt_max=0.02
    )
    result = simulate(setup)
    alpha = k / (thermal.DENSITY * cp)
    for t_probe in (2.0, 10.0, 20.0):
        i = int(np.argmin(np.abs(result.t - t_probe)))
        z = result.z[result.z < 0.03]
        exact = 20.0 + (850.0 - 20.0) * erf(z / (2 * np.sqrt(alpha * result.t[i])))
        assert np.max(np.abs(result.T[i, : z.size] - exact)) < 2.0  # K, out of 830 K


def test_cooling_rate_falls_with_distance() -> None:
    rates = simulate(JominySetup()).rate_at_distance(np.array([1.5, 5, 10, 20, 30, 50]))
    assert np.all(rates > 0)
    assert np.all(np.diff(rates) < 0)


def test_stronger_quench_cools_faster_near_the_face() -> None:
    weak = simulate(JominySetup(h_end=5e3)).rate_at_distance(np.array([1.5]))
    strong = simulate(JominySetup(h_end=5e4)).rate_at_distance(np.array([1.5]))
    assert strong[0] > 2 * weak[0]


def test_specific_heat_is_continuous() -> None:
    T = np.array([599.999, 600.0, 899.999, 900.0])
    cp = thermal.specific_heat(T)
    assert cp[0] == pytest.approx(cp[1], abs=0.1)
    assert cp[2] == pytest.approx(cp[3], abs=0.1)


def test_cooling_rates_are_converged_in_space_and_time() -> None:
    distances = np.array([1.5, 5.0, 10.0, 20.0, 40.0])
    base = simulate(JominySetup()).rate_at_distance(distances)
    fine = simulate(
        JominySetup(n_nodes=801, dt_initial=1e-3 / 3, dt_max=0.5 / 3, dt_growth=1.02 ** (1 / 3))
    ).rate_at_distance(distances)
    assert np.max(np.abs(fine / base - 1)) < 0.02
