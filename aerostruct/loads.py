"""Spanwise lift distributions for a half-wing treated as a cantilever.

x runs from the wing root (x = 0) to the tip (x = L). Each function returns a
callable w(x) [N/m] that integrates to the requested total lift over the span.
"""

from math import pi

import numpy as np


def uniform(total: float, L: float):
    return lambda x: np.full_like(np.asarray(x, dtype=float), total / L)


def triangular(total: float, L: float):
    """Maximum at the root, zero at the tip (a conservative outboard-light shape)."""
    return lambda x: 2 * total / L * (1 - np.asarray(x) / L)


def elliptic(total: float, L: float):
    """Ideal elliptic loading: w = w0 sqrt(1 - (x/L)^2), with w0 = 4 W / (pi L)."""
    w0 = 4 * total / (pi * L)
    return lambda x: w0 * np.sqrt(np.clip(1 - (np.asarray(x) / L) ** 2, 0, None))


def planform(total: float, L: float, taper: float):
    """Lift proportional to local chord of a linearly tapered wing (taper = tip chord / root chord)."""
    mean = (1 + taper) / 2
    return lambda x: total / L * (1 - (1 - taper) * np.asarray(x) / L) / mean


def schrenk(total: float, L: float, taper: float = 1.0):
    """Schrenk's approximation: the average of the planform and elliptic distributions.

    A standard quick estimate of spanwise lift for preliminary structural sizing
    of unswept wings.
    """
    ell, pf = elliptic(total, L), planform(total, L, taper)
    return lambda x: 0.5 * (ell(x) + pf(x))


DISTRIBUTIONS = {"uniform", "triangular", "elliptic", "schrenk"}


def make(name: str, total: float, L: float, taper: float = 1.0):
    if name == "uniform":
        return uniform(total, L)
    if name == "triangular":
        return triangular(total, L)
    if name == "elliptic":
        return elliptic(total, L)
    if name == "schrenk":
        return schrenk(total, L, taper)
    if name == "none":
        return lambda x: np.zeros_like(np.asarray(x, dtype=float))
    raise ValueError(f"unknown lift distribution {name!r}")
