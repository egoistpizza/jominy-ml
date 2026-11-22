"""Steel chemical composition as a validated value object."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

#: Alloying elements tracked throughout the project, in a fixed order (wt%).
ELEMENTS: tuple[str, ...] = ("C", "Si", "Mn", "P", "Cr", "Ni", "Mo", "V", "Cu")

#: Upper bounds (wt%) of the low-alloy domain the models are meant for.
_DOMAIN_MAX = {
    "C": 0.8,
    "Si": 1.0,
    "Mn": 2.0,
    "P": 0.05,
    "Cr": 2.5,
    "Ni": 3.5,
    "Mo": 1.0,
    "V": 0.3,
    "Cu": 0.5,
}


@dataclass(frozen=True, slots=True)
class Composition:
    """Mass fractions in wt%. Elements not given are taken as 0."""

    C: float
    Si: float = 0.0
    Mn: float = 0.0
    P: float = 0.0
    Cr: float = 0.0
    Ni: float = 0.0
    Mo: float = 0.0
    V: float = 0.0
    Cu: float = 0.0

    def __post_init__(self) -> None:
        for name in ELEMENTS:
            value = getattr(self, name)
            if not 0.0 <= value <= _DOMAIN_MAX[name]:
                raise ValueError(
                    f"{name} = {value} wt% is outside the low-alloy domain [0, {_DOMAIN_MAX[name]}]"
                )
        if self.C <= 0.0:
            raise ValueError("carbon content must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, float | None]) -> Composition:
        """Build from a mapping; missing or ``None`` entries become 0."""
        unknown = set(values) - set(ELEMENTS)
        if unknown:
            raise ValueError(f"unknown elements: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in values.items() if v is not None})

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def as_array(self) -> npt.NDArray[np.float64]:
        return np.array([getattr(self, e) for e in ELEMENTS], dtype=np.float64)
