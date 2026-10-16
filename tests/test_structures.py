"""Validation of the beam model against closed-form textbook results."""

from math import pi

import numpy as np
import pytest

from aerostruct import CantileverBeam, IBeam, Material, Rectangle, Tube, load_library
from aerostruct import loads

AL = Material("test Al", E=70e9, Fty=500e6, Ftu=570e6, density=2800)
L = 2.0


@pytest.fixture
def beam():
    return CantileverBeam(L, Rectangle(0.04, 0.08), AL)


# --------------------------------------------------------------------- sections
def test_rectangle_properties():
    s = Rectangle(0.04, 0.08)
    assert s.I == pytest.approx(0.04 * 0.08 ** 3 / 12)
    assert s.shear_stress(1000) == pytest.approx(1.5 * 1000 / s.A)  # tau_max = 1.5 V / A


def test_tube_properties():
    s = Tube(0.075, 0.0025)
    R, r = 0.0375, 0.035
    assert s.A == pytest.approx(pi * (R ** 2 - r ** 2))
    assert s.I == pytest.approx(pi * (0.075 ** 4 - 0.070 ** 4) / 64)
    # Thin-walled tube: tau_max -> 2 V / A
    thin = Tube(0.1, 0.0005)
    assert thin.shear_stress(1000) == pytest.approx(2 * 1000 / thin.A, rel=1e-2)


def test_ibeam_reduces_to_rectangle_when_web_equals_flange():
    ib = IBeam(height=0.1, flange_width=0.05, flange_thickness=0.01, web_thickness=0.05)
    rect = Rectangle(0.05, 0.1)
    for prop in ("A", "I", "Q"):
        assert getattr(ib, prop) == pytest.approx(getattr(rect, prop))


def test_material_library_loads():
    lib = load_library()
    assert "Al 7075-T6" in lib
    assert all(m.Fty <= m.Ftu for m in lib.values())


# ------------------------------------------------------------------------ beam
def test_tip_point_load(beam):
    P = 1500.0
    r = beam.analyse(point_loads=[(L, P)])
    assert r.root_shear == pytest.approx(P)
    assert r.root_moment == pytest.approx(P * L)
    assert r.tip_deflection == pytest.approx(P * L ** 3 / (3 * beam.EI), rel=1e-5)


def test_uniform_load(beam):
    w = 800.0
    r = beam.analyse(loads.uniform(w * L, L))
    assert r.root_shear == pytest.approx(w * L, rel=1e-9)
    assert r.root_moment == pytest.approx(w * L ** 2 / 2, rel=1e-6)
    assert np.allclose(r.M, w * (L - r.x) ** 2 / 2, rtol=1e-5, atol=1e-6)
    assert r.tip_deflection == pytest.approx(w * L ** 4 / (8 * beam.EI), rel=1e-5)


def test_triangular_load(beam):
    W = 2000.0  # total; w0 = 2W/L at the root
    r = beam.analyse(loads.triangular(W, L))
    assert r.root_moment == pytest.approx(W * L / 3, rel=1e-5)   # resultant at L/3
    w0 = 2 * W / L
    # Load peaking at the fixed end: tip deflection w0 L^4 / (30 E I)
    assert r.tip_deflection == pytest.approx(w0 * L ** 4 / (30 * beam.EI), rel=1e-5)


def test_elliptic_load_centroid(beam):
    W = 2000.0
    r = beam.analyse(loads.elliptic(W, L))
    assert r.root_shear == pytest.approx(W, rel=1e-4)
    assert r.root_moment == pytest.approx(W * 4 * L / (3 * pi), rel=1e-4)


def test_mid_span_point_load_deflection(beam):
    P, a = 1000.0, 0.6 * L
    r = beam.analyse(point_loads=[(a, P)])
    expected = P * a ** 2 * (3 * L - a) / (6 * beam.EI)
    assert r.tip_deflection == pytest.approx(expected, rel=1e-5)
    assert np.allclose(r.V[r.x > a], 0.0)


@pytest.mark.parametrize("name", sorted(loads.DISTRIBUTIONS))
def test_distributions_integrate_to_total(name):
    x = np.linspace(0, L, 20001)
    w = loads.make(name, 1234.0, L, taper=0.5)(x)
    assert np.trapezoid(w, x) == pytest.approx(1234.0, rel=1e-4)


def test_superposition(beam):
    a = beam.analyse(loads.elliptic(1000, L))
    b = beam.analyse(point_loads=[(0.5, -300)])
    both = beam.analyse(loads.elliptic(1000, L), point_loads=[(0.5, -300)])
    assert np.allclose(both.M, a.M + b.M)
    assert np.allclose(both.v, a.v + b.v)
