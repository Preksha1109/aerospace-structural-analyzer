"""Isotropic material model and a small material library loaded from CSV.

The library values are typical handbook figures for illustration only.
They are NOT statistically based design allowables (A- or B-basis).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "data" / "materials.csv"


@dataclass(frozen=True)
class Material:
    name: str
    E: float          # Young's modulus [Pa]
    Fty: float        # tensile yield strength [Pa]
    Ftu: float        # ultimate tensile strength [Pa]
    density: float    # [kg/m^3]

    def __post_init__(self):
        if min(self.E, self.Fty, self.Ftu, self.density) <= 0:
            raise ValueError(f"{self.name}: properties must be positive")
        if self.Fty > self.Ftu:
            raise ValueError(f"{self.name}: yield strength exceeds ultimate strength")

    def __str__(self):
        return (f"{self.name} (E = {self.E / 1e9:.1f} GPa, Fty = {self.Fty / 1e6:.0f} MPa, "
                f"Ftu = {self.Ftu / 1e6:.0f} MPa)")


def load_library(path=DEFAULT_LIBRARY) -> dict[str, Material]:
    """Read materials.csv (MPa / GPa columns) into Material objects keyed by name."""
    df = pd.read_csv(path)
    return {
        row.name: Material(row.name, row.E_GPa * 1e9, row.Fty_MPa * 1e6,
                           row.Ftu_MPa * 1e6, row.density_kg_m3)
        for row in df.itertuples(index=False)
    }
