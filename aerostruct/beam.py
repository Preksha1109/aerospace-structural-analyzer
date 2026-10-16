"""Euler-Bernoulli cantilever beam with arbitrary distributed and point loads.

Sign convention: x from the fixed root (0) to the free tip (L); loads are
positive upwards. Internal forces at station x come from equilibrium of the
outboard segment:

    V(x) = integral_x^L w(s) ds + sum P_i  (x_i > x)
    M(x) = integral_x^L V(s) ds          (positive = upward loads bending the tip up)

and the deflection from v'' = M / (E I) with v(0) = v'(0) = 0.
Distributed-load integrals are evaluated numerically (trapezoidal rule), so any
lift distribution can be used; point-load contributions are added exactly.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .materials import Material
from .sections import Section


def _integrate_from_tip(x, f):
    """F(x) = integral_x^L f(s) ds on the grid x."""
    total = cumulative_trapezoid(f, x, initial=0.0)
    return total[-1] - total


@dataclass
class BeamResponse:
    x: np.ndarray
    w: np.ndarray               # distributed load [N/m]
    V: np.ndarray               # shear force [N]
    M: np.ndarray               # bending moment [N m]
    sigma: np.ndarray           # extreme-fibre bending stress [Pa]
    tau: np.ndarray             # peak transverse shear stress (neutral axis) [Pa]
    theta: np.ndarray           # slope [rad]
    v: np.ndarray               # deflection [m]
    point_loads: list = field(default_factory=list)

    @property
    def von_mises(self):
        """Conservative combination sqrt(sigma^2 + 3 tau^2).

        Peak bending and peak shear stress occur at different fibres, so adding
        them at one point overestimates the true von Mises stress slightly.
        """
        return np.sqrt(self.sigma ** 2 + 3 * self.tau ** 2)

    @property
    def root_moment(self):
        return float(self.M[0])

    @property
    def root_shear(self):
        return float(self.V[0])

    @property
    def tip_deflection(self):
        return float(self.v[-1])

    def critical(self):
        """Station and value of the highest combined stress."""
        i = int(np.argmax(self.von_mises))
        return float(self.x[i]), float(self.von_mises[i])


@dataclass
class CantileverBeam:
    length: float
    section: Section
    material: Material
    n_points: int = 801

    @property
    def x(self):
        return np.linspace(0.0, self.length, self.n_points)

    @property
    def EI(self):
        return self.material.E * self.section.I

    def analyse(self, distributed=None, point_loads=()) -> BeamResponse:
        """distributed: callable w(x) [N/m]; point_loads: iterable of (x_station, force)."""
        x = self.x
        w = np.zeros_like(x) if distributed is None else np.asarray(distributed(x), dtype=float)

        V = _integrate_from_tip(x, w)
        M = _integrate_from_tip(x, V)
        for xp, P in point_loads:
            if not 0.0 <= xp <= self.length:
                raise ValueError(f"point load at x = {xp} is off the beam")
            outboard = x < xp
            V = V + P * outboard
            M = M + P * np.clip(xp - x, 0.0, None)

        theta = cumulative_trapezoid(M / self.EI, x, initial=0.0)
        v = cumulative_trapezoid(theta, x, initial=0.0)

        return BeamResponse(
            x=x, w=w, V=V, M=M,
            sigma=self.section.bending_stress(M),
            tau=self.section.shear_stress(V),
            theta=theta, v=v, point_loads=list(point_loads),
        )
