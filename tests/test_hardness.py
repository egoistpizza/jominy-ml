import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jominy.hardness import (
    austenitizing_parameter,
    critical_log_rates,
    hardness_hv,
    hv_to_hrc,
    phase_fractions,
)
from jominy.materials import Composition
from jominy.synth import SAMPLING_BOX

#: Hypothesis draws compositions from the same box the synthetic training data uses.
compositions = st.builds(
    Composition, **{e: st.floats(lo, hi) for e, (lo, hi) in SAMPLING_BOX.items()}
)


def test_austenitizing_parameter_equals_temperature_for_one_hour() -> None:
    assert austenitizing_parameter(850.0, 1.0) == pytest.approx(850.0)
    assert austenitizing_parameter(850.0, 0.5) < 850.0 < austenitizing_parameter(850.0, 2.0)


def test_critical_rate_for_martensite_against_hand_calculation() -> None:
    c = Composition(C=0.40, Mn=0.70, Cr=1.00, Mo=0.20, Ni=0.10)
    expected = 9.81 - (
        4.62 * 0.40 + 1.05 * 0.70 + 0.5 * 1.00 + 0.66 * 0.20 + 0.54 * 0.10 + 0.00183 * 850.0
    )
    assert critical_log_rates(c, 850.0)["M"] == pytest.approx(expected)


@settings(max_examples=200, deadline=None)
@given(c=compositions, log_v=st.floats(1.0, 7.0))
def test_phase_fractions_are_a_partition(c: Composition, log_v: float) -> None:
    f = phase_fractions(c, np.array([log_v]), 830.0)[0]
    assert np.all(f >= -1e-12)
    assert f.sum() == pytest.approx(1.0)


@settings(max_examples=100, deadline=None)
@given(c=compositions)
def test_martensite_share_never_rises_on_slower_cooling(c: Composition) -> None:
    log_v = np.linspace(7.0, 1.0, 200)  # fast -> slow
    martensite = phase_fractions(c, log_v, 830.0)[:, 0]
    assert np.all(np.diff(martensite) <= 1e-12)


def test_very_fast_cooling_gives_martensite_hardness() -> None:
    c = Composition(C=0.45, Si=0.25, Mn=0.70)
    hv = hardness_hv(c, np.array([1e4]), 830.0)[0]
    hv_m = 127 + 949 * 0.45 + 27 * 0.25 + 11 * 0.70 + 21 * np.log10(1e4 * 3600)
    assert hv == pytest.approx(hv_m)


def test_hv_to_hrc_reproduces_table_and_is_monotone() -> None:
    assert hv_to_hrc(np.array([302.0, 513.0, 697.0])) == pytest.approx([30.0, 50.0, 60.0])
    hv = np.linspace(150, 900, 300)
    assert np.all(np.diff(hv_to_hrc(hv)) > 0)


@settings(max_examples=100, deadline=None)
@given(c=compositions)
def test_phase_fractions_are_continuous_in_the_cooling_rate(c: Composition) -> None:
    # Shrinking the offset by 100x shrinks every change of a continuous piecewise-linear
    # function by 100x; a jump would stay the same size.
    x = np.linspace(1.0, 7.0, 2001)

    def largest_change(eps: float) -> float:
        diff = phase_fractions(c, x + eps, 830.0) - phase_fractions(c, x, 830.0)
        return float(np.abs(diff).max())

    assert largest_change(1e-6) <= largest_change(1e-4) / 50 + 1e-9
