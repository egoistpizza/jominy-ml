import numpy as np
import pytest

from jominy.data import _split_band, load_curves, residual_levels, to_frame
from jominy.materials import ELEMENTS, Composition


@pytest.fixture(scope="module")
def curves():  # type: ignore[no-untyped-def]
    return load_curves()


def test_inventory(curves) -> None:  # type: ignore[no-untyped-def]
    kinds = [c.kind for c in curves]
    assert kinds.count("heat") == 5
    assert kinds.count("band_min") == kinds.count("band_max") == 17
    assert {c.source for c in curves} == {"tomasic2025", "ovako2026", "iso683"}


def test_band_limits_are_ordered(curves) -> None:  # type: ignore[no-untyped-def]
    bands: dict[tuple[str, str], dict[str, object]] = {}
    for c in curves:
        if c.kind != "heat":
            bands.setdefault((c.grade, c.variant), {})[c.kind] = c
    for pair in bands.values():
        lo, hi = pair["band_min"], pair["band_max"]
        common = np.intersect1d(lo.distance_mm, hi.distance_mm)  # type: ignore[attr-defined]
        lo_v = lo.hrc[np.isin(lo.distance_mm, common)]  # type: ignore[attr-defined]
        hi_v = hi.hrc[np.isin(hi.distance_mm, common)]  # type: ignore[attr-defined]
        assert np.all(lo_v < hi_v)
        lean, rich = lo.composition.as_array(), hi.composition.as_array()  # type: ignore[attr-defined]
        assert np.all(lean <= rich)


def test_hardness_does_not_rise_along_the_bar(curves) -> None:  # type: ignore[no-untyped-def]
    for c in curves:
        assert np.all(np.diff(c.hrc) <= 0.5), c.curve_id  # tolerance: reading resolution


def test_missing_lower_limit_is_dropped() -> None:
    upper, lower = _split_band([1.5, 3, 5, 5, 3, 3, 1.5], [50, 45, 40, 40, 45, 30, 38])
    assert upper == {1.5: 50, 3: 45, 5: 40}
    assert lower == {3: 30, 1.5: 38}


def test_long_frame_has_one_row_per_point(curves) -> None:  # type: ignore[no-untyped-def]
    df = to_frame(curves)
    assert len(df) == sum(c.distance_mm.size for c in curves)
    assert df[list(ELEMENTS)].notna().all().all()


def test_composition_rejects_values_outside_domain() -> None:
    with pytest.raises(ValueError):
        Composition(C=0.4, Cr=5.0)
    with pytest.raises(ValueError):
        Composition.from_mapping({"C": 0.4, "W": 1.0})


def test_held_out_heats_do_not_set_band_residual_levels() -> None:
    # C45E has the lowest Si of the five heats, so holding it out must raise the Si
    # filled into band corners that give no lower Si limit.
    heats = [c for c in load_curves() if c.kind == "heat"]
    si_all = residual_levels(heats)["Si"]
    si_train = residual_levels([h for h in heats if h.grade != "C45E"])["Si"]
    assert si_train > si_all

    def band_si(held_out: frozenset[str]) -> set[float]:
        return {c.composition.Si for c in load_curves(held_out=held_out) if c.kind == "band_min"}

    assert si_all in band_si(frozenset())
    assert si_all not in band_si(frozenset({"C45E"}))
    assert si_train in band_si(frozenset({"C45E"}))
