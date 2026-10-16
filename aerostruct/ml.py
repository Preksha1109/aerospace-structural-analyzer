"""Machine-learning stress prediction (concept demonstration).

The analytical beam model labels randomly generated wing/spar designs with
their peak combined stress at limit load. Regressors then learn to predict
that stress directly from the design parameters, and are compared on:

  * held-out designs from the same parameter ranges (interpolation), and
  * longer wings than any in the training set (extrapolation).

Two ways of presenting the problem to the model are compared:

  raw features         the design parameters as they are
  physics-informed     a nominal stress  S = (n m g / 2) * L / Z  computed from
                       simple beam theory, plus dimensionless ratios; the model
                       only learns the correction  sigma / S  (load shape,
                       inertia relief, shear), which stays in a similar range
                       for any wing size.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import G, Aircraft, apply_case
from .materials import Material
from .sections import Tube

RAW_FEATURES = ["load_factor", "mtow_kg", "half_span_m", "taper_ratio", "wing_mass_per_m",
                "pod_mass_kg", "pod_station_frac", "outer_diameter_mm", "wall_mm"]

TRAINING_RANGES = {
    "load_factor": (1.0, 4.5),
    "mtow_kg": (50.0, 400.0),
    "half_span_m": (1.5, 4.5),
    "taper_ratio": (0.3, 1.0),
    "wing_mass_per_m": (1.0, 5.0),
    "pod_mass_kg": (0.0, 25.0),
    "pod_station_frac": (0.15, 0.6),
    "outer_diameter_mm": (40.0, 120.0),
    "wall_to_diameter": (0.02, 0.08),
}


# Plausibility filters so the dataset only holds designs an engineer might consider.
MAX_WING_MASS_FRACTION = 0.4        # (both wings + both pods) / MTOW
STRESS_WINDOW_MPA = (50.0, 500.0)   # neither grossly oversized nor far beyond the material


def sample_designs(n: int, ranges=TRAINING_RANGES, seed: int = 0) -> pd.DataFrame:
    """Uniform random designs, keeping only those with a plausible wing mass fraction."""
    rng = np.random.default_rng(seed)
    kept = []
    while sum(map(len, kept)) < n:
        df = pd.DataFrame({k: rng.uniform(lo, hi, 2 * n) for k, (lo, hi) in ranges.items()})
        df["wall_mm"] = df.pop("wall_to_diameter") * df["outer_diameter_mm"]
        wing_mass = 2 * (df["wing_mass_per_m"] * df["half_span_m"] + df["pod_mass_kg"])
        kept.append(df[wing_mass <= MAX_WING_MASS_FRACTION * df["mtow_kg"]])
    return pd.concat(kept, ignore_index=True).iloc[:n][RAW_FEATURES]


def generate_dataset(n: int, material: Material, ranges=TRAINING_RANGES, seed: int = 0):
    """Sample, label with the beam model, and keep designs inside STRESS_WINDOW_MPA."""
    lo, hi = STRESS_WINDOW_MPA
    Xs, ys, batch = [], [], 0
    while sum(map(len, ys)) < n:
        X = sample_designs(2 * n, ranges, seed=seed * 1000 + batch)
        y = label(X, material)
        ok = (y >= lo) & (y <= hi)
        Xs.append(X[ok])
        ys.append(y[ok])
        batch += 1
    X = pd.concat(Xs, ignore_index=True).iloc[:n]
    y = pd.concat(ys, ignore_index=True).iloc[:n]
    return X, y


def label(designs: pd.DataFrame, material: Material, n_points: int = 201) -> pd.Series:
    """Peak combined stress at limit load [MPa] from the analytical beam model."""
    out = np.empty(len(designs))
    for i, d in enumerate(designs.itertuples(index=False)):
        ac = Aircraft(
            name="sample", mtow_kg=d.mtow_kg, half_span_m=d.half_span_m, taper_ratio=d.taper_ratio,
            wing_mass_per_m=d.wing_mass_per_m, pod_mass_kg=d.pod_mass_kg,
            pod_station_m=d.pod_station_frac * d.half_span_m, spar_material=material,
            spar_section=Tube(d.outer_diameter_mm * 1e-3, d.wall_mm * 1e-3),
        )
        resp = apply_case(ac, d.load_factor, "schrenk", n_points=n_points)
        out[i] = resp.von_mises.max() / 1e6
    return pd.Series(out, index=designs.index, name="max_von_mises_MPa")


def nominal_stress_mpa(X: pd.DataFrame) -> np.ndarray:
    """Root bending stress if the whole half-wing lift acted at the tip: (n m g / 2) L / Z."""
    Z = np.array([Tube(D * 1e-3, t * 1e-3).Z for D, t in zip(X["outer_diameter_mm"], X["wall_mm"])])
    return X["load_factor"] * X["mtow_kg"] * G / 2 * X["half_span_m"] / Z / 1e6


def physics_features(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "taper_ratio": X["taper_ratio"],
        "pod_mass_ratio": X["pod_mass_kg"] / X["mtow_kg"],
        "pod_station_frac": X["pod_station_frac"],
        "wing_mass_ratio": X["wing_mass_per_m"] * X["half_span_m"] / X["mtow_kg"],
        "wall_to_diameter": X["wall_mm"] / X["outer_diameter_mm"],
        "span_to_diameter": X["half_span_m"] / (X["outer_diameter_mm"] * 1e-3),
    })


class PhysicsInformedRegressor(BaseEstimator, RegressorMixin):
    """Predicts sigma = S * f(dimensionless features), learning log(f) with a base regressor."""

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.model_ = clone(self.base or HistGradientBoostingRegressor(random_state=0))
        self.model_.fit(physics_features(X), np.log(np.asarray(y) / nominal_stress_mpa(X)))
        return self

    def predict(self, X):
        return nominal_stress_mpa(X) * np.exp(self.model_.predict(physics_features(X)))


def models():
    return {
        "Linear regression (raw features)": make_pipeline(StandardScaler(), LinearRegression()),
        "Gradient boosting (raw features)": HistGradientBoostingRegressor(random_state=0),
        "Gradient boosting (physics-informed)": PhysicsInformedRegressor(),
    }


def scores(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pct = np.abs(y_pred - y_true) / y_true * 100
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return {"MAPE_pct": float(pct.mean()), "p95_pct": float(np.percentile(pct, 95)),
            "max_pct": float(pct.max()), "R2": float(1 - ss_res / ss_tot)}
