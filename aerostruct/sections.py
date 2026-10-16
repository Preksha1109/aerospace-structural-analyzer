"""Cross-section geometry for beam bending and shear.

Each section provides the properties the beam model needs:

    A    area                                   [m^2]
    I    second moment of area about the neutral axis   [m^4]
    c    distance from neutral axis to extreme fibre    [m]
    Q    first moment of area above the neutral axis    [m^3]
    b    material width at the neutral axis             [m]

Bending stress is sigma = M c / I and the peak transverse shear stress
(at the neutral axis) is tau = V Q / (I b).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import pi


class Section(ABC):
    @property
    @abstractmethod
    def A(self) -> float: ...

    @property
    @abstractmethod
    def I(self) -> float: ...  # noqa: E743

    @property
    @abstractmethod
    def c(self) -> float: ...

    @property
    @abstractmethod
    def Q(self) -> float: ...

    @property
    @abstractmethod
    def b(self) -> float: ...

    @property
    def Z(self) -> float:
        """Elastic section modulus I / c."""
        return self.I / self.c

    def bending_stress(self, M):
        return abs(M) * self.c / self.I

    def shear_stress(self, V):
        return abs(V) * self.Q / (self.I * self.b)

    def mass_per_length(self, density: float) -> float:
        return self.A * density


@dataclass(frozen=True)
class Rectangle(Section):
    width: float
    height: float

    @property
    def A(self):
        return self.width * self.height

    @property
    def I(self):  # noqa: E743
        return self.width * self.height ** 3 / 12

    @property
    def c(self):
        return self.height / 2

    @property
    def Q(self):
        return self.width * (self.height / 2) ** 2 / 2

    @property
    def b(self):
        return self.width


@dataclass(frozen=True)
class Tube(Section):
    """Circular tube, the usual spar section on small aircraft and UAVs."""
    outer_diameter: float
    wall: float

    def __post_init__(self):
        if not 0 < self.wall < self.outer_diameter / 2:
            raise ValueError("wall thickness must be between 0 and the outer radius")

    @property
    def _r(self):
        return self.outer_diameter / 2, self.outer_diameter / 2 - self.wall

    @property
    def A(self):
        R, r = self._r
        return pi * (R ** 2 - r ** 2)

    @property
    def I(self):  # noqa: E743
        R, r = self._r
        return pi * (R ** 4 - r ** 4) / 4

    @property
    def c(self):
        return self.outer_diameter / 2

    @property
    def Q(self):
        R, r = self._r
        return 2 * (R ** 3 - r ** 3) / 3

    @property
    def b(self):
        return 2 * self.wall


@dataclass(frozen=True)
class IBeam(Section):
    """Doubly symmetric I-section: overall height, flange width and thickness, web thickness."""
    height: float
    flange_width: float
    flange_thickness: float
    web_thickness: float

    def __post_init__(self):
        if 2 * self.flange_thickness >= self.height or self.web_thickness > self.flange_width:
            raise ValueError("inconsistent I-beam dimensions")

    @property
    def _hw(self):
        return self.height - 2 * self.flange_thickness  # clear web height

    @property
    def A(self):
        return 2 * self.flange_width * self.flange_thickness + self.web_thickness * self._hw

    @property
    def I(self):  # noqa: E743
        return (self.flange_width * self.height ** 3
                - (self.flange_width - self.web_thickness) * self._hw ** 3) / 12

    @property
    def c(self):
        return self.height / 2

    @property
    def Q(self):
        flange = self.flange_width * self.flange_thickness * (self.height - self.flange_thickness) / 2
        web = self.web_thickness * (self._hw / 2) ** 2 / 2
        return flange + web

    @property
    def b(self):
        return self.web_thickness


def from_spec(spec: dict) -> Section:
    """Build a section from a config dict, e.g. {"type": "tube", "outer_diameter_mm": 75, "wall_mm": 2.5}."""
    kind = spec["type"].lower()
    mm = {k[:-3]: v * 1e-3 for k, v in spec.items() if k.endswith("_mm")}
    if kind == "tube":
        return Tube(**mm)
    if kind == "rectangle":
        return Rectangle(**mm)
    if kind == "ibeam":
        return IBeam(**mm)
    raise ValueError(f"unknown section type {spec['type']!r}")
