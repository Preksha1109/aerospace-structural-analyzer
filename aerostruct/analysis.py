"""Load-case pipeline: aircraft config + table of load cases -> results table with margins of safety.

Each half-wing is a cantilever spar fixed at the fuselage. For a load case with
load factor n:

    lift         n * (MTOW g) / 2, spread along the span by the chosen distribution
    wing inertia -n * g * (wing mass per metre), uniform along the span
    pod inertia  -n * g * pod mass, a point load at the pod station (if fitted)

Strength checks follow the usual aerospace convention:

    limit load     no yield:     MS_yield = Fty / sigma_limit - 1
    ultimate load  no failure:   MS_ult   = Ftu / (1.5 sigma_limit) - 1

A margin of safety MS >= 0 passes.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import loads
from .beam import BeamResponse, CantileverBeam
from .materials import Material, load_library
from .sections import Section, from_spec

G = 9.80665
ULTIMATE_FACTOR = 1.5


@dataclass
class Aircraft:
    name: str
    mtow_kg: float
    half_span_m: float
    taper_ratio: float
    wing_mass_per_m: float
    pod_mass_kg: float
    pod_station_m: float
    spar_material: Material
    spar_section: Section

    @classmethod
    def from_json(cls, path, materials=None):
        cfg = json.loads(Path(path).read_text())
        materials = materials or load_library()
        spar = cfg.pop("spar")
        return cls(**cfg, spar_material=materials[spar["material"]],
                   spar_section=from_spec(spar["section"]))

    def spar(self, n_points=801) -> CantileverBeam:
        return CantileverBeam(self.half_span_m, self.spar_section, self.spar_material, n_points)


def apply_case(aircraft: Aircraft, load_factor: float, distribution: str,
               lift: bool = True, pod: bool = True, n_points: int = 801) -> BeamResponse:
    L = aircraft.half_span_m
    half_lift = load_factor * aircraft.mtow_kg * G / 2 if lift else 0.0
    w_lift = loads.make(distribution if lift else "none", half_lift, L, aircraft.taper_ratio)
    w_inertia = -load_factor * G * aircraft.wing_mass_per_m

    def w(x):
        return w_lift(x) + w_inertia

    point = [(aircraft.pod_station_m, -load_factor * G * aircraft.pod_mass_kg)] if pod else []
    return aircraft.spar(n_points).analyse(w, point)


def summarise(resp: BeamResponse, material: Material) -> dict:
    station, vm = resp.critical()
    return {
        "root_shear_N": resp.root_shear,
        "root_moment_Nm": resp.root_moment,
        "max_bending_MPa": resp.sigma.max() / 1e6,
        "max_shear_MPa": resp.tau.max() / 1e6,
        "max_von_mises_MPa": vm / 1e6,
        "critical_station_m": station,
        "tip_deflection_mm": resp.tip_deflection * 1e3,
        "MS_yield": material.Fty / vm - 1 if vm > 0 else np.inf,
        "MS_ultimate": material.Ftu / (ULTIMATE_FACTOR * vm) - 1 if vm > 0 else np.inf,
    }


def run_load_cases(aircraft: Aircraft, cases: pd.DataFrame):
    """Run every row of the load-case table. Returns (results DataFrame, {case: BeamResponse})."""
    required = {"case", "load_factor", "distribution", "lift", "pod"}
    missing = required - set(cases.columns)
    if missing:
        raise ValueError(f"load-case table is missing columns: {sorted(missing)}")

    rows, responses = [], {}
    for case in cases.itertuples(index=False):
        resp = apply_case(aircraft, case.load_factor, case.distribution,
                          bool(case.lift), bool(case.pod))
        responses[case.case] = resp
        rows.append({"case": case.case, "description": getattr(case, "description", ""),
                     "load_factor": case.load_factor,
                     **summarise(resp, aircraft.spar_material)})
    results = pd.DataFrame(rows)
    results["min_MS"] = results[["MS_yield", "MS_ultimate"]].min(axis=1)
    results["status"] = np.where(results["min_MS"] >= 0, "PASS", "FAIL")
    return results, responses
