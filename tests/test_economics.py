import pytest

from jominy.economics import (
    USD_PER_EUR,
    AlloyValue,
    aim_shift_saving,
    break_even_tonnes,
    contained_alloy_value,
)
from jominy.materials import Composition


def test_contained_alloy_value_by_hand() -> None:
    # 1 % Ni = 10 kg/t at 15 USD/kg, converted at the ECB rate.
    assert contained_alloy_value(Composition(C=0.4, Ni=1.0)) == pytest.approx(150.0 / USD_PER_EUR)
    expected_cr = 10 * 1.60 * 2.20462 / USD_PER_EUR
    assert contained_alloy_value(Composition(C=0.4, Cr=1.0)) == pytest.approx(expected_cr)


def test_plain_carbon_steel_has_only_manganese_value() -> None:
    assert contained_alloy_value(Composition(C=0.45)) == 0.0
    assert contained_alloy_value(Composition(C=0.45, Mn=1.0)) == pytest.approx(4.5 / USD_PER_EUR)


def test_saving_and_break_even() -> None:
    value = AlloyValue(lean=300.0, rich=450.0)
    assert aim_shift_saving(value, 0.1) == pytest.approx(15.0)
    n = break_even_tonnes(15.0, 3000.0)
    assert (n - 1) * 15.0 < 3000.0 <= n * 15.0
