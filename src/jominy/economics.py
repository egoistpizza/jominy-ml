"""Cost figures for the economic analysis.

Every price below is quoted from a dated, citable source (bibliography key in the
comment). Values marked as assumptions are our own estimates, not quotes. The module
only does arithmetic on these inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jominy.materials import Composition

LB_PER_KG = 2.20462

#: ECB reference rate, annual average 2025, USD per EUR (key ecb2025).
USD_PER_EUR = 1.1300

#: Element prices in USD per kg of contained element, annual estimates 2025 from USGS
#: Mineral Commodity Summaries 2026 (keys usgs2026cr, usgs2026mo, usgs2026ni, usgs2026mn).
ELEMENT_PRICE_USD_PER_KG = {
    "Cr": 1.60 * LB_PER_KG,  # ferrochromium, 1.60 USD per lb of contained Cr
    "Mo": 51.0,  # molybdic oxide, USD per kg of contained Mo
    "Ni": 15_000.0 / 1000.0,  # LME cash, 15 000 USD/t
    "Mn": 4.50 / 10.0,  # ore, 44 % Mn, CIF China: 4.50 USD per metric ton unit (10 kg Mn)
}

#: Alloy surcharge for rolled engineering steel in EUR/t, valid from 01.02.2026
#: (THE STEEL / Frankstahl, key thesteel2026).
ALLOY_SURCHARGE_EUR_PER_T = {
    "16MnCr5": 39.0,
    "25CrMo4": 140.0,
    "42CrMo4": 140.0,
    "34CrNiMo6": 314.0,
    "18CrNiMo7-6": 358.0,
}

#: FH Südwestfalen price list, valid from 01.09.2023, EUR net (key fhswf2023).
JOMINY_TEST_EUR = 189.60  # Jominy test to DIN EN ISO 642 without normalising
JOMINY_TEST_NORMALISED_EUR = 315.80  # with normalising
SPECIMEN_PREPARATION_EUR_PER_H = 95.60  # specimen preparation, per hour

#: Own assumptions, not quoted prices.
AIM_SHIFT_FRACTION = 0.10  # share of the alloy window by which a mill could lower its aim
SERVICE_DAYS_PER_YEAR = 3  # effort to calibrate and validate the model for one site
SERVICE_PRICE_EUR_PER_YEAR = 3000.0  # proposed annual price per site


@dataclass(frozen=True)
class AlloyValue:
    """Value of the Mn, Cr, Ni and Mo contained in one tonne of steel, EUR/t."""

    lean: float
    rich: float

    @property
    def mid(self) -> float:
        return (self.lean + self.rich) / 2


def contained_alloy_value(c: Composition) -> float:
    """EUR per tonne of steel for the Mn, Cr, Ni and Mo it contains."""
    usd = sum(
        float(getattr(c, e)) * 10.0 * price  # wt% -> kg/t
        for e, price in ELEMENT_PRICE_USD_PER_KG.items()
    )
    return float(usd / USD_PER_EUR)


def alloy_value_range(lean: Composition, rich: Composition) -> AlloyValue:
    """Contained-alloy value at the lean and the rich corner of a composition range."""
    return AlloyValue(contained_alloy_value(lean), contained_alloy_value(rich))


def aim_shift_saving(value: AlloyValue, fraction: float = AIM_SHIFT_FRACTION) -> float:
    """EUR/t saved if the aim analysis moves by ``fraction`` of the alloy window."""
    return fraction * (value.rich - value.lean)


def service_cost_eur() -> float:
    """Labour cost of one site-year of the calibration service at the lab hourly rate."""
    return SERVICE_DAYS_PER_YEAR * 8 * SPECIMEN_PREPARATION_EUR_PER_H


def break_even_tonnes(saving_eur_per_t: float, price: float = SERVICE_PRICE_EUR_PER_YEAR) -> int:
    """Tonnes per year at which the saving pays the service price (rounded up)."""
    return math.ceil(price / saving_eur_per_t)
