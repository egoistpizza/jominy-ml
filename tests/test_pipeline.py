"""A small study through the whole pipeline, kept in memory."""

import pytest

from jominy.latex import Outputs
from jominy.pipeline import Study, run

SMALL = Study(
    n_synthetic=60,
    n_synthetic_test=20,
    n_boot=20,
    models=("Phys", "Cal", "Lin", "Gbdt", "Surr", "Hyb"),
)


@pytest.fixture(scope="module")
def outputs() -> Outputs:
    out = Outputs()
    run(out, SMALL)
    return out


def test_every_artefact_is_produced(outputs: Outputs) -> None:
    expected = {
        "results.tex",
        "summary.csv",
        "cv_predictions.csv",
        "tab_results.tex",
        "tab_grades.tex",
        "tab_family.tex",
        "tab_costs.tex",
        "tab_data.tex",
        "fig_cooling.pdf",
        "fig_heats.pdf",
        "fig_bands.pdf",
        "fig_error_distance.pdf",
        "fig_shap.pdf",
        "fig_parity.pdf",
        "fig_phases.pdf",
        "fig_sensitivity.pdf",
    }
    assert expected <= set(outputs.files)


def test_core_numbers_are_defined_and_plausible(outputs: Outputs) -> None:
    for key in ("Phys", "Cal", "Lin", "Gbdt", "Hyb"):
        assert 0 < float(outputs.macros[f"mae{key}"]) < 15
    assert outputs.macros["nCurves"] == "39"
    assert outputs.macros["nFolds"] == "11"
    assert 0 <= float(outputs.macros["pHolmHybCal"]) <= 1


def test_models_outside_the_study_produce_no_macros(outputs: Outputs) -> None:
    assert "maeGp" not in outputs.macros
    assert "gpCoverage" not in outputs.macros
