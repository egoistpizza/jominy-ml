"""End-to-end run: data -> models -> cross-validation -> figures, tables, LaTeX macros.

Every published number is produced here and written through
:class:`~jominy.latex.Outputs`; nothing is transcribed by hand. The study settings
(seed, sample sizes, bootstrap size, models) are one :class:`Study` value, so a test
can run the whole pipeline in a few seconds with a small study.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jominy import economics, figures
from jominy.data import Curve, load_curves, to_frame
from jominy.evaluate import (
    CurveKind,
    CVResult,
    Interval,
    holm,
    leave_one_group_out,
    sign_flip_pvalue,
)
from jominy.explain import mean_abs_shap
from jominy.hardness import (
    MartensiteRest,
    critical_log_rates,
    dropped_anchors,
    log_rate_per_hour,
    phase_fractions,
)
from jominy.latex import Outputs, num
from jominy.models import ROSTER, DataGBDT, Surrogate, h_end_errors
from jominy.physics import H_NOMINAL, PA, PhysicsModel, cooling_rate
from jominy.synth import synthetic_frame

NAMES = {spec.key: spec.name for spec in ROSTER}

#: Models shown in the per-grade table and the error-by-distance figure, and in the
#: curve figures.
TABLE_MODELS = ("Phys", "Cal", "Lin", "Gbdt", "Hyb")
SHOWN = ("Cal", "Hyb")

#: Pre-specified comparisons, tested with the exact sign-flip test and Holm correction.
TESTED_PAIRS = (("Hyb", "Cal"), ("Hyb", "Lin"), ("Gbdt", "Cal"))
#: Further paired differences reported with bootstrap intervals only.
REPORTED_PAIRS = (*TESTED_PAIRS, ("Lin", "Cal"), ("Hyb", "Phys"))

BAND_PANELS = (
    ("42CrMo4", "326C"),
    ("34CrNiMo6", "EN ISO 683-2"),
    ("25CrMo4", "EN ISO 683-2"),
    ("20MnCrS5", "EN ISO 683-3"),
)

#: Steel families for the stricter sensitivity split (by main alloying system).
FAMILIES = {
    "16MnCr5": "MnCr",
    "20MnCrS5": "MnCr",
    "17CrNi6-6": "CrNi",
    "C45E": "C",
    "25CrMo4": "CrMo",
    "42CrMo4": "CrMo",
    "50CrMo4": "CrMo",
    "20CrMo5": "CrMo",
    "34CrNiMo6": "CrNiMo",
    "18CrNiMo7-6": "CrNiMo",
    "20NiCrMo2-2": "CrNiMo",
}

#: Composition range used for the cost table: one band per grade with a published surcharge.
COST_VARIANTS = {
    "16MnCr5": "234K",
    "25CrMo4": "EN ISO 683-2",
    "42CrMo4": "326C",
    "34CrNiMo6": "EN ISO 683-2",
    "18CrNiMo7-6": "159X",
}
#: LaTeX macro names cannot contain digits.
COST_TAGS = {
    "16MnCr5": "SixteenMnCr",
    "25CrMo4": "TwentyFiveCrMo",
    "42CrMo4": "FortyTwoCrMo",
    "34CrNiMo6": "ThirtyFourCrNiMo",
    "18CrNiMo7-6": "EighteenCrNiMo",
}


@dataclass(frozen=True)
class Study:
    """Everything that sizes a run. The defaults reproduce results/."""

    seed: int = 42
    n_synthetic: int = 2000
    n_synthetic_test: int = 500
    n_boot: int = 2000
    models: tuple[str, ...] = field(default_factory=lambda: tuple(s.key for s in ROSTER))
    n_jobs: int = -1


def _fold_frame(held_out: frozenset[str]) -> pd.DataFrame:
    """Real curves with band residual levels taken from the training heats only."""
    return to_frame(load_curves(held_out=held_out))


def run(out: Outputs, study: Study = Study()) -> pd.DataFrame:
    """Produce every artefact in results/; returns the model summary."""
    figures.apply_style()
    curves = load_curves()
    real = to_frame(curves)
    _inventory(out, curves, real)
    _physics_figures(out, curves, real)
    _physics_sensitivity(out, curves)
    _economics(out, curves)

    physics = PhysicsModel(h_end=H_NOMINAL)
    syn = synthetic_frame(study.n_synthetic, study.seed, physics)
    syn_test = synthetic_frame(study.n_synthetic_test, study.seed + 1, physics)
    out.macro("nSynthetic", rf"\num{{{study.n_synthetic}}}")
    factories = [spec.build(study.seed, syn) for spec in ROSTER if spec.key in study.models]

    with warnings.catch_warnings():
        # The GP optimiser reports hyperparameters at their bounds; this is expected
        # with ten training grades and does not affect the reported comparison.
        warnings.simplefilter("ignore", category=UserWarning)
        by_grade = leave_one_group_out(
            real,
            factories,
            n_boot=study.n_boot,
            seed=study.seed,
            n_jobs=study.n_jobs,
            fold_frame=_fold_frame,
        )
        by_family = leave_one_group_out(
            real.assign(family=real["grade"].map(FAMILIES)),
            factories,
            "family",
            n_boot=study.n_boot,
            seed=study.seed,
            n_jobs=study.n_jobs,
            fold_frame=_fold_frame,
        )
    out.csv("cv_predictions", by_grade.frame)
    summary = _model_comparison(out, by_grade)
    _family_sensitivity(out, summary, by_family)
    _surrogate_and_explanations(out, Surrogate(syn, seed=study.seed), syn_test, real, study)
    _versions(out)
    out.write_macros()
    return summary.rename(index=NAMES)


def _inventory(out: Outputs, curves: list[Curve], real: pd.DataFrame) -> None:
    out.macro("nCurves", str(len(curves)))
    out.macro("nHeats", str(sum(c.kind == "heat" for c in curves)))
    out.macro("nBands", str(sum(c.kind == "band_max" for c in curves)))
    out.macro("nBandCurves", str(sum(c.kind != "heat" for c in curves)))
    out.macro("nPoints", str(len(real)))
    out.macro("nGrades", str(real["grade"].nunique()))
    comps = {c.composition for c in curves}
    out.macro("nCompositions", str(len(comps)))
    for anchor, macro in (
        ("B90", "nDropBninety"),
        ("FP90", "nDropFPninety"),
        ("B50", "nDropBfifty"),
    ):
        out.macro(macro, str(sum(anchor in dropped_anchors(c, PA) for c in comps)))
    cite = {
        "tomasic2025": r"\cite{tomasic2025}",
        "ovako2026": r"\cite{ovako2026}",
        "iso683": r"\cite{ovako2026}\textsuperscript{a}",
    }
    variants: dict[tuple[str, str], list[str]] = {}
    for c in curves:
        label = "Schmelze" if c.kind == "heat" else c.variant
        variants.setdefault((c.grade, c.source), [])
        if label not in variants[(c.grade, c.source)]:
            variants[(c.grade, c.source)].append(label)
    out.table(
        "tab_data",
        "llll",
        ["Stahlsorte", "Quelle", "Varianten", "Art"],
        [
            [
                grade,
                cite[src],
                ", ".join(v),
                "Messkurve" if src == "tomasic2025" else "H-Band (min./max.)",
            ]
            for (grade, src), v in sorted(variants.items())
        ],
    )


def _physics_figures(out: Outputs, curves: list[Curve], real: pd.DataFrame) -> None:
    d = np.linspace(1.5, 50, 200)
    out.figure(
        "fig_cooling", figures.cooling(d, {h: cooling_rate(d, h) for h in (5e3, 1e4, 2e4, 5e4)})
    )
    for label, dist in (("A", 1.5), ("B", 10.0), ("C", 50.0)):
        out.macro(f"rate{label}", num(float(cooling_rate(np.array([dist]), H_NOMINAL)[0]), 0))
    for dist, tag in ((15.0, "Fifteen"), (30.0, "Thirty")):
        at = np.array([dist])
        out.macro(
            f"rateRatio{tag}", num(float(cooling_rate(at, 5e4)[0] / cooling_rate(at, 5e3)[0]))
        )

    heats = {c.grade: c.composition for c in curves if c.kind == "heat"}
    log_v = log_rate_per_hour(cooling_rate(d, H_NOMINAL))
    fractions = {g: phase_fractions(heats[g], log_v, PA) for g in ("C45E", "42CrMo4")}
    out.figure(
        "fig_phases",
        figures.phases(d, {f"{g} (Schmelze aus Tomašić 2025)": f for g, f in fractions.items()}),
    )
    out.macro("PA", num(PA, 0))
    for grade, tag in (("C45E", "CFortyFive"), ("42CrMo4", "FortyTwo")):
        martensite = fractions[grade][:, 0]
        out.macro(f"mFull{tag}", num(float(d[np.argmax(martensite < 0.999)])))
        out.macro(f"mHalf{tag}", num(float(d[np.argmax(martensite < 0.5)])))
    for grade, tag in (("C45E", "cfortyfive"), ("42CrMo4", "fortytwo")):
        out.macro(f"vM{tag}", num(10 ** critical_log_rates(heats[grade], PA)["M"] / 3600))

    errors = h_end_errors(real)
    out.figure("fig_sensitivity", figures.sensitivity(errors))
    out.macro("hBest", rf"\num{{{min(errors, key=errors.__getitem__):.0f}}}")
    out.macro("sensMin", num(min(errors.values())))
    out.macro("sensMax", num(max(errors.values())))


def _physics_sensitivity(out: Outputs, curves: list[Curve]) -> None:
    heat = next(c for c in curves if c.grade == "C45E")
    measured = dict(zip(heat.distance_mm.tolist(), heat.hrc.tolist(), strict=True))
    rests: tuple[tuple[MartensiteRest, str], ...] = (
        ("bainite", "Bain"),
        ("ferrite_pearlite", "Fp"),
    )
    for dist, tag in ((20.0, "Twenty"), (25.0, "TwentyFive")):
        at = np.array([dist])
        for rest, name in rests:
            hrc = PhysicsModel(h_end=H_NOMINAL, rest=rest).predict_curve(heat.composition, at)
            out.macro(f"cFortyFive{name}{tag}", num(float(hrc[0])))
        out.macro(f"cFortyFiveMeas{tag}", num(measured[dist]))
    for dist, tag in ((5.0, "Five"), (15.0, "Fifteen"), (30.0, "Thirty")):
        at = np.array([dist])
        ratio = cooling_rate(at, H_NOMINAL, 0.75)[0] / cooling_rate(at, H_NOMINAL)[0]
        out.macro(f"condRatio{tag}", num(float(ratio), 2))


def _model_comparison(out: Outputs, cv: CVResult) -> pd.DataFrame:
    out.macro("nFolds", str(cv.n_groups))
    s = cv.summary()
    out.csv("summary", s.rename(index=NAMES))
    out.table(
        "tab_results",
        "lS[table-format=1.1]cS[table-format=1.1]cS[table-format=-1.1]"
        "S[table-format=1.1]S[table-format=1.1]",
        ["Modell", "{MAE}", "{VB}", "{MAE je Sorte}", "{VB}", "{Bias}", "{Schmelzen}", "{Bänder}"],
        [
            [
                NAMES[str(k)],
                f"{r.mae:.1f}",
                rf"\numrange{{{r.mae_low:.1f}}}{{{r.mae_high:.1f}}}",
                f"{r.mae_grade:.1f}",
                rf"\numrange{{{r.mae_grade_low:.1f}}}{{{r.mae_grade_high:.1f}}}",
                num(r.bias),
                f"{r.mae_heats:.1f}",
                f"{r.mae_bands:.1f}",
            ]
            for k, r in s.iterrows()
        ],
    )
    for k, r in s.iterrows():
        out.interval(f"mae{k}", Interval(r.mae, r.mae_low, r.mae_high))
        out.interval(f"maeGrade{k}", Interval(r.mae_grade, r.mae_grade_low, r.mae_grade_high))
        out.macro(f"mae{k}Heats", num(r.mae_heats))
        out.macro(f"mae{k}Bands", num(r.mae_bands))

    for a, b in REPORTED_PAIRS:
        out.interval(f"diff{a}{b}", cv.paired_difference(a, b), digits=2)
        out.interval(f"diffGrade{a}{b}", cv.paired_difference(a, b, per_group=True), digits=2)
    out.interval("diffHybCalHeats", cv.paired_difference("Hyb", "Cal", kind="heat"), digits=2)
    out.interval("diffHybCalBands", cv.paired_difference("Hyb", "Cal", kind="band"), digits=2)
    raw = {f"{a}{b}": sign_flip_pvalue(cv.group_differences(a, b)) for a, b in TESTED_PAIRS}
    for name, p_adj in holm(raw).items():
        out.macro(f"p{name}", num(raw[name], 2))
        out.macro(f"pHolm{name}", num(p_adj, 2))

    if "Gp" in set(cv.frame["model"]):
        out.macro("gpCoverage", num(100 * cv.coverage("Gp"), 0))
        kinds: tuple[tuple[CurveKind, str], ...] = (
            ("heat", "Heats"),
            ("band_min", "Min"),
            ("band_max", "Max"),
        )
        for kind, tag in kinds:
            out.macro(f"gpCoverage{tag}", num(100 * cv.coverage("Gp", kind), 0))

    per_grade = cv.mae_per_group()
    out.table(
        "tab_grades",
        "l" + "S[table-format=2.1]" * len(TABLE_MODELS),
        ["Sorte (zurückgehalten)", *(f"{{{_short(k)}}}" for k in TABLE_MODELS)],
        [[str(g), *(f"{row[k]:.1f}" for k in TABLE_MODELS)] for g, row in per_grade.iterrows()],
    )
    worse = per_grade.index[per_grade["Hyb"] > per_grade["Cal"]]
    out.macro("nHybWorse", str(len(worse)))
    out.macro("hybWorseGrades", ", ".join(str(g) for g in worse))
    for k in TABLE_MODELS:
        near, far = cv.mae_near_far(k)
        out.macro(f"near{k}", num(near))
        out.macro(f"far{k}", num(far))
    c45 = cv.subset("Cal").query("group == 'C45E'")
    worst = c45.iloc[int(np.argmax((c45["pred"] - c45["hrc"]).abs().to_numpy()))]
    out.macro("cfortyfiveErr", num(float(worst["pred"] - worst["hrc"]), 0))
    out.macro("cfortyfiveAt", num(float(worst["distance_mm"]), 0))

    shown = {k: NAMES[k] for k in SHOWN}
    out.figure("fig_heats", figures.heats(cv.frame, shown))
    out.figure("fig_bands", figures.bands(cv.frame, shown, BAND_PANELS))
    out.figure(
        "fig_error_distance",
        figures.error_by_distance(
            {k: (NAMES[k], cv.mae_by_distance(k, figures.DISTANCE_BINS)) for k in TABLE_MODELS}
        ),
    )
    return s


def _family_sensitivity(out: Outputs, by_grade: pd.DataFrame, cv: CVResult) -> None:
    grade: dict[str, dict[str, float]] = by_grade.to_dict(orient="index")  # type: ignore[assignment]
    family: dict[str, dict[str, float]] = cv.summary().to_dict(orient="index")  # type: ignore[assignment]
    out.macro("nFamilies", str(cv.n_groups))
    out.table(
        "tab_family",
        "lS[table-format=1.1]S[table-format=1.1]S[table-format=1.1]S[table-format=1.1]",
        [
            "Modell",
            "{Sorte: Punkte}",
            "{Sorte: je Sorte}",
            "{Familie: Punkte}",
            "{Familie: je Familie}",
        ],
        [
            [
                NAMES[k],
                f"{grade[k]['mae']:.1f}",
                f"{grade[k]['mae_grade']:.1f}",
                f"{family[k]['mae']:.1f}",
                f"{family[k]['mae_grade']:.1f}",
            ]
            for k in grade
        ],
    )
    for k in ("Cal", "Lin", "Hyb", "Gbdt"):
        out.macro(f"maeFamily{k}", num(family[k]["mae"]))
        out.macro(f"maeFamilyGroup{k}", num(family[k]["mae_grade"]))


def _range(lo: float, hi: float) -> str:
    return rf"\numrange[range-phrase=--]{{{lo:.2f}}}{{{hi:.2f}}}"


def _economics(out: Outputs, curves: list[Curve]) -> None:
    corners = {(c.grade, c.variant, c.kind): c.composition for c in curves if c.kind != "heat"}
    rows = []
    values = {}
    for grade, variant in COST_VARIANTS.items():
        lean = corners[(grade, variant, "band_min")]
        rich = corners[(grade, variant, "band_max")]
        value = economics.alloy_value_range(lean, rich)
        values[grade] = value
        surcharge = economics.ALLOY_SURCHARGE_EUR_PER_T[grade]
        rows.append(
            [
                grade,
                variant,
                *(_range(getattr(lean, e), getattr(rich, e)) for e in ("Mn", "Cr", "Ni", "Mo")),
                f"{value.lean:.0f}",
                f"{value.rich:.0f}",
                f"{surcharge:.0f}",
                f"{100 * surcharge / value.mid:.0f}",
            ]
        )
        tag = COST_TAGS[grade]
        out.macro(f"alloyLean{tag}", num(value.lean, 0))
        out.macro(f"alloyRich{tag}", num(value.rich, 0))
        out.macro(f"surchargeShare{tag}", num(100 * surcharge / value.mid, 0))
    out.table(
        "tab_costs",
        "ll" + "c" * 4 + "S[table-format=3.0]" * 4,
        [
            "Sorte",
            "Band",
            "Mn",
            "Cr",
            "Ni",
            "Mo",
            "{mager}",
            "{reich}",
            "{Zuschlag}",
            r"{Anteil / \%}",
        ],
        rows,
    )
    saving = economics.aim_shift_saving(values["34CrNiMo6"])
    surcharges = economics.ALLOY_SURCHARGE_EUR_PER_T
    out.macro("aimShiftPct", num(100 * economics.AIM_SHIFT_FRACTION, 0))
    out.macro("aimSaving", num(saving, 0))
    out.macro("aimSavingThousand", f"{1000 * saving:.0f}")
    out.macro("substitutionSaving", num(surcharges["34CrNiMo6"] - surcharges["42CrMo4"], 0))
    out.macro("usdPerEur", num(economics.USD_PER_EUR, 4))
    out.macro("jominyCost", num(economics.JOMINY_TEST_EUR, 2))
    out.macro("jominyCostNorm", num(economics.JOMINY_TEST_NORMALISED_EUR, 2))
    out.macro("prepRate", num(economics.SPECIMEN_PREPARATION_EUR_PER_H, 2))
    out.macro("serviceDays", str(economics.SERVICE_DAYS_PER_YEAR))
    out.macro("serviceCost", num(economics.service_cost_eur(), 0))
    out.macro("servicePrice", f"{economics.SERVICE_PRICE_EUR_PER_YEAR:.0f}")
    out.macro("breakEvenTonnes", str(economics.break_even_tonnes(saving)))


def _surrogate_and_explanations(
    out: Outputs, surrogate: Surrogate, syn_test: pd.DataFrame, real: pd.DataFrame, study: Study
) -> None:
    pred = surrogate.predict(syn_test)
    err = np.abs(pred - syn_test["hrc"].to_numpy())
    out.figure("fig_parity", figures.parity(syn_test["hrc"].to_numpy(), pred))
    out.macro("surrMae", num(float(err.mean()), 2))
    out.macro("surrMax", num(float(err.max())))

    panels = {
        "Surrogat (Simulationsdaten)": mean_abs_shap(surrogate.model, syn_test),
        "GBDT (Messdaten)": mean_abs_shap(DataGBDT(seed=study.seed).fit(real).model, real),
    }
    out.figure("fig_shap", figures.shap_bars(panels))
    alloys = ["Mn", "Cr", "Ni", "Mo"]
    for tag, s in zip(("Syn", "Real"), panels.values(), strict=True):
        for pos in ("near", "far"):
            col = s[pos].astype(float)
            out.macro(f"shapC{tag}{pos.capitalize()}", num(float(col["C"])))
            out.macro(f"shapAlloy{tag}{pos.capitalize()}", num(float(col[alloys].sum())))


def _versions(out: Outputs) -> None:
    pkgs = ("numpy", "scipy", "pandas", "scikit-learn", "shap", "matplotlib")
    versions = ", ".join(f"{p} {md.version(p)}" for p in pkgs)
    out.macro("softwareVersions", f"Python {platform.python_version()}, {versions}")


def _short(key: str) -> str:
    return next(spec.short for spec in ROSTER if spec.key == key)
