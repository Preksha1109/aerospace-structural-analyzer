"""ML stress prediction demo.

    python scripts/train_stress_model.py

1. Generate 4,000 random UAV wing/spar designs and label each with the analytical
   beam model (peak combined stress at limit load, Schrenk lift + inertia relief).
   Implausible designs are discarded (wing + pods over 40 % of MTOW, or limit
   stress outside 50-500 MPa).
2. Train three regressors on 80 % of them.
3. Score on the held-out 20 % and on 500 designs with longer wings (4.5-6 m half-span)
   than anything seen in training.

Writes data/generated/ml_dataset.csv, results/ml_metrics.json and docs/ml_stress_prediction.png.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aerostruct import load_library  # noqa: E402
from aerostruct.ml import TRAINING_RANGES, generate_dataset, models, scores  # noqa: E402
from aerostruct.plotting import INK, INK2, MUTED, SERIES, figure, title  # noqa: E402

N_DESIGNS, N_EXTRAP, SEED = 4000, 500, 0
C_TEST, C_EXTRAP = SERIES[0], SERIES[1]


def main():
    material = load_library()["Al 7075-T6"]

    t0 = time.perf_counter()
    X, y = generate_dataset(N_DESIGNS, material, seed=SEED)
    X_ex, y_ex = generate_dataset(N_EXTRAP, material, dict(TRAINING_RANGES, half_span_m=(4.5, 6.0)),
                                  seed=SEED + 1)
    print(f"Generated and labelled {N_DESIGNS + N_EXTRAP} plausible designs in {time.perf_counter() - t0:.1f} s")

    out = ROOT / "data" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([X.assign(split="train/test", max_von_mises_MPa=y),
               X_ex.assign(split="extrapolation", max_von_mises_MPa=y_ex)]).to_csv(
        out / "ml_dataset.csv", index=False, float_format="%.5g")

    idx = np.random.default_rng(SEED).permutation(N_DESIGNS)
    tr, te = idx[: int(0.8 * N_DESIGNS)], idx[int(0.8 * N_DESIGNS):]
    print(f"Stress range in training data: {y.min():.0f}-{y.max():.0f} MPa; "
          f"extrapolation set: {y_ex.min():.0f}-{y_ex.max():.0f} MPa\n")

    metrics, preds = {}, {}
    print(f"{'model':40}{'test MAPE %':>12}{'test R2':>9}{'extrap MAPE %':>15}{'extrap p95 %':>14}")
    for name, model in models().items():
        model.fit(X.iloc[tr], y.iloc[tr])
        p_te, p_ex = model.predict(X.iloc[te]), model.predict(X_ex)
        preds[name] = (p_te, p_ex)
        metrics[name] = {"test": scores(y.iloc[te], p_te), "extrapolation": scores(y_ex, p_ex)}
        m = metrics[name]
        print(f"{name:40}{m['test']['MAPE_pct']:12.2f}{m['test']['R2']:9.4f}"
              f"{m['extrapolation']['MAPE_pct']:15.2f}{m['extrapolation']['p95_pct']:14.2f}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ml_metrics.json").write_text(json.dumps(
        {"n_train": len(tr), "n_test": len(te), "n_extrapolation": N_EXTRAP,
         "training_ranges": TRAINING_RANGES, "extrapolation_half_span_m": [4.5, 6.0],
         "models": metrics}, indent=2))

    # ---------------------------------------------------------------- figure
    fig, axes = figure(1, 3, (14, 4.8))
    lim = (0, max(y.max(), y_ex.max()) * 1.05)
    for ax, (name, letter) in zip(axes[:2], [("Gradient boosting (raw features)", "a"),
                                             ("Gradient boosting (physics-informed)", "b")]):
        p_te, p_ex = preds[name]
        ax.plot(lim, lim, color=MUTED, lw=1, zorder=1)
        ax.scatter(y.iloc[te], p_te, s=9, color=C_TEST, alpha=0.6, lw=0, label="held-out test", zorder=2)
        ax.scatter(y_ex, p_ex, s=9, color=C_EXTRAP, alpha=0.6, lw=0, label="longer wings (extrapolation)",
                   zorder=3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("beam-model stress [MPa]")
        ax.set_ylabel("predicted stress [MPa]")
        title(ax, f"({letter}) {name}")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", markerscale=2)

    ax = axes[2]
    names = list(metrics)
    yy = np.arange(len(names))
    h = 0.36
    test = [metrics[n]["test"]["MAPE_pct"] for n in names]
    ext = [metrics[n]["extrapolation"]["MAPE_pct"] for n in names]
    # y axis is inverted below, so the test bar (yy - h/2) sits above the extrapolation bar
    ax.barh(yy - h / 2, test, height=h, color=C_TEST, label="held-out test")
    ax.barh(yy + h / 2, ext, height=h, color=C_EXTRAP, label="longer wings (extrapolation)")
    for yi, a, b in zip(yy, test, ext):
        ax.text(a, yi - h / 2, f" {a:.1f}%", va="center", fontsize=8.5, color=INK2)
        ax.text(b, yi + h / 2, f" {b:.1f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(yy, [n.replace(" (", "\n(") for n in names], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(min(test + ext) * 0.5, max(test + ext) * 4)
    ticks = [t for t in (0.5, 1, 3, 10, 30, 100) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks, [f"{t:g}%" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("mean absolute error (log scale)")
    ax.grid(axis="y", visible=False)
    title(ax, "(c) Prediction error")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for t in ax.get_yticklabels():
        t.set_color(INK)

    fig.tight_layout(w_pad=2.0)
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "ml_stress_prediction.png", dpi=150)
    print("\nSaved results/ml_metrics.json and docs/ml_stress_prediction.png")


if __name__ == "__main__":
    main()
