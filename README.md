# jominy-ml

Predicts the Jominy end-quench hardenability curve of low-alloy steels from their
chemical composition and compares three approaches on public data:

- a physics chain: 1D heat conduction in the Jominy bar, then the Creusot-Loire
  equations for phase fractions and hardness;
- statistical learning: ridge regression and gradient-boosted trees;
- hybrids that keep the physics as a baseline and learn only its residual.

All numbers, tables and figures come from `make results`.

## Results

Leave-one-grade-out cross-validation: every steel grade is predicted by models that never
saw it. 534 hardness values from 11 grades: 5 measured curves and 34 limit curves of 17
hardenability bands. Mean absolute error (MAE) in HRC, 95 % interval from a bootstrap
over grades.

| Model | MAE, all points | MAE, per grade | Measured heats | Band limits |
|---|---|---|---|---|
| Physics, fixed heat transfer coefficient | 4.6 (3.3–5.8) | 4.6 | 5.2 | 4.5 |
| Physics, calibrated heat transfer coefficient | 4.3 (2.9–5.6) | 4.2 | 4.1 | 4.3 |
| Ridge regression | 4.0 (3.1–5.0) | 4.4 | 4.6 | 3.9 |
| Gradient boosting on real data | 5.0 (3.5–6.5) | 4.8 | 3.9 | 5.2 |
| Gradient boosting trained only on simulated curves | 4.9 (3.5–6.7) | 5.6 | 7.3 | 4.6 |
| Physics + gradient-boosted residual | 3.6 (3.0–4.3) | 4.1 | 4.7 | 3.4 |
| Physics + Gaussian-process residual | 4.0 (2.7–5.5) | 4.2 | 5.0 | 3.9 |

What the numbers support:

- The physics chain gets within 4.6 HRC without any parameter fitted to hardness data,
  and it is the most accurate model within 3 mm of the quenched end.
- The physics + residual hybrid has the lowest error overall, but its lead comes from
  the band limits. On the five measured heats the calibrated physics is better.
- None of the three pre-specified comparisons is significant after Holm correction
  (exact sign-flip test over the 11 grades, adjusted p ≥ 0.73).
- The worst case for every model is the plain carbon steel C45E. The Creusot-Loire
  critical cooling rate overestimates its hardenability.

`results/summary.csv` holds the full table, `results/results.tex` every number as a
LaTeX macro.

## Reproduce

Requires Python 3.12.

```bash
make install   # .venv with the versions in requirements-lock.txt
make check     # ruff, mypy --strict, pytest
make results   # data -> models -> results/ (about 1 min on 16 cores)
```

All random generators are seeded. Figures and tables in `results/` are overwritten on
every run.

## Layout

```
data/raw/           source data as transcribed, with provenance (data/SOURCES.md)
src/jominy/
  materials.py      Composition value object (validated wt%)
  thermal.py        implicit finite-volume solver for the Jominy bar
  hardness.py       Creusot-Loire critical cooling rates, phase fractions, HV to HRC
  physics.py        thermal + hardness -> predicted curve
  data.py           loads heats and hardenability bands into one long table
  synth.py          Latin-hypercube compositions -> simulated training curves
  models.py         seven predictors behind one fit/predict interface
  evaluate.py       grouped cross-validation, cluster bootstrap, sign-flip test
  explain.py        SHAP attributions near and far from the quenched end
  economics.py      alloy value and cost figures from dated price sources
  figures.py        figures
  latex.py          writes tables and number macros
  pipeline.py       runs everything and writes results/
tests/              solver against the analytic solution, convergence, phase-fraction
                    properties, leakage checks for the cross-validation
results/            output of make results
```

## Modelling choices

- **No coefficients from memory.** The Creusot-Loire equations are transcribed from
  Trzaska (2013), *Archives of Materials Science and Engineering* 61/2, 87–92. Thermal
  properties follow EN 1993-1-2. The regression baseline refits the functional form of
  Just's hardenability formula instead of quoting its coefficients.
- **Our additions to the Creusot-Loire method.** Phase fractions between two critical
  rates are interpolated linearly in log(rate). A critical rate that is not slower than
  its faster neighbour is dropped. At the M90 and M50 rates the non-martensitic rest is
  taken as bainite. HV is converted to HRC with ASTM E140, extrapolated below 20 HRC.
  Every steel is assumed austenitized at 850 °C for 30 min.
- **Hardenability bands as data.** A band's lower limit is paired with the lean end of
  the composition range, its upper limit with the rich end. Where a range gives no
  lower limit for Si, Ni, Mo or Cu, or omits the element, the lean end gets the lowest
  content seen in the measured heats (never more than the upper limit), so that a zero
  does not mark the curve as a band limit. In cross-validation this level is computed
  from the training heats of each fold. P and V without a lower limit are set to 0.
- **Grouped validation.** Every split holds out a whole grade. A stricter run holds out
  whole steel families (C, MnCr, CrNi, CrMo, CrNiMo). The box from which the simulated
  training compositions are sampled is wide enough to contain every real composition;
  it carries no hardness data.
- **Same settings for all tree models.** Every gradient-boosting model uses the same
  fixed hyperparameters. The pure data model and the surrogate are constrained to fall
  monotonically with distance; the hybrid correction is not, since a residual need not
  be monotone, and it gets the physics prediction as an extra input.
- **Inputs of the learned models.** C, Si, Mn, Cr, Ni, Mo and distance. P, V and Cu are
  left out because the sources give them for some curves only.

## Data

| Source | Content |
|---|---|
| N. Tomašić, PhD thesis, University of Rijeka, 2025, Tables 3.2–3.3 | 5 measured heats, 13 distances each |
| Ovako Steel Navigator data sheets, retrieved 2026-11-22 | 17 boron-free hardenability bands (5 ISO 683-2/-3, 12 producer variants) with composition ranges |

Details and exclusions: [`data/SOURCES.md`](data/SOURCES.md).

## License

Code: MIT, see `LICENSE`. The files in `data/raw/` stay subject to the terms of their
original publishers and are included so that the analysis can be reproduced.
