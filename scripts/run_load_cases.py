"""Run the load-case table for the example UAV wing spar.

    python scripts/run_load_cases.py [--aircraft data/aircraft.json] [--cases data/load_cases.csv]

Prints the results table, writes results/load_case_results.csv and docs/load_cases.png.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aerostruct import loads  # noqa: E402
from aerostruct.analysis import ULTIMATE_FACTOR, Aircraft, run_load_cases  # noqa: E402
from aerostruct.plotting import INK, INK2, MUTED, SERIES, figure, title  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aircraft", default=ROOT / "data" / "aircraft.json")
    ap.add_argument("--cases", default=ROOT / "data" / "load_cases.csv")
    args = ap.parse_args()

    ac = Aircraft.from_json(args.aircraft)
    cases = pd.read_csv(args.cases)
    results, responses = run_load_cases(ac, cases)

    mat, sec = ac.spar_material, ac.spar_section
    print(ac.name)
    print(f"Spar: {mat}, tube {sec.outer_diameter * 1e3:.0f} x {sec.wall * 1e3:.1f} mm, "
          f"{sec.mass_per_length(mat.density):.2f} kg/m\n")
    show = results[["case", "description", "load_factor", "root_moment_Nm", "max_von_mises_MPa",
                    "tip_deflection_mm", "MS_yield", "MS_ultimate", "status"]]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(show.to_string(index=False))
    crit = results.loc[results["min_MS"].idxmin()]
    print(f"\nCritical case: {crit.case} ({crit.description}), min MS = {crit.min_MS:.2f}")

    (ROOT / "results").mkdir(exist_ok=True)
    results.to_csv(ROOT / "results" / "load_case_results.csv", index=False, float_format="%.4f")

    # ---------------------------------------------------------------- figure
    fig, axes = figure(2, 2, (11, 8.2))
    L = ac.half_span_m
    colours = dict(zip(results["case"], SERIES))

    # (a) lift distribution shapes
    ax = axes[0, 0]
    x = np.linspace(0, L, 400)
    styles = {"uniform": ":", "triangular": "-.", "elliptic": "--", "schrenk": "-"}
    for name, ls in styles.items():
        w = loads.make(name, 1.0, L, ac.taper_ratio)(x) * L
        label = f"Schrenk (taper {ac.taper_ratio})" if name == "schrenk" else name
        ax.plot(x / L, w, ls, color=INK if name == "schrenk" else INK2, lw=2 if name == "schrenk" else 1.4,
                label=label)
    ax.set_xlabel("spanwise position x / L")
    ax.set_ylabel("lift per unit span, w L / W")
    ax.set_ylim(0, 2.1)
    title(ax, "(a) Spanwise lift distributions (same total lift)")
    ax.legend(frameon=False, fontsize=8.5)

    # (b) bending moment
    ax = axes[0, 1]
    for case, r in responses.items():
        ax.plot(r.x, r.M / 1e3, color=colours[case], lw=2, label=case)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("distance from root x [m]")
    ax.set_ylabel("bending moment M [kN·m]")
    title(ax, "(b) Bending moment along the spar")
    ax.axvline(ac.pod_station_m, color=MUTED, lw=0.8, ls=":")
    ax.text(ac.pod_station_m + 0.05, 0.97, f"pod station ({ac.pod_station_m:.1f} m)",
            transform=ax.get_xaxis_transform(), va="top", fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)

    # (c) combined stress vs allowables
    ax = axes[1, 0]
    for case, r in responses.items():
        ax.plot(r.x, r.von_mises / 1e6, color=colours[case], lw=2, label=case)
    yield_line, ult_line = mat.Fty / 1e6, mat.Ftu / ULTIMATE_FACTOR / 1e6
    ax.axhline(yield_line, color=INK, lw=1.2, ls="--")
    ax.axhline(ult_line, color=INK, lw=1.2, ls=":")
    ax.annotate(f"yield allowable, Fty = {yield_line:.0f} MPa", (L, yield_line), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.annotate(f"ultimate allowable, Ftu / {ULTIMATE_FACTOR} = {ult_line:.0f} MPa", (L, ult_line),
                xytext=(0, -5), textcoords="offset points", ha="right", va="top", fontsize=8.5, color=INK2)
    ax.set_ylim(0, yield_line * 1.12)
    ax.set_xlabel("distance from root x [m]")
    ax.set_ylabel("combined stress at limit load [MPa]")
    title(ax, "(c) Stress along the spar vs allowables")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="center right")

    # (d) margins of safety
    ax = axes[1, 1]
    order = results.iloc[::-1]
    y = np.arange(len(order))
    ax.barh(y, order["min_MS"], color=[colours[c] for c in order["case"]], height=0.6)
    for yi, (_, row) in zip(y, order.iterrows()):
        governing = "ult." if row.MS_ultimate <= row.MS_yield else "yield"
        ax.text(row.min_MS, yi, f"  {row.min_MS:.2f} ({governing})", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [f"{c}  {d}" for c, d in zip(order["case"], order["description"])], fontsize=8.5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(0, order["min_MS"].max() * 1.3)
    ax.set_xlabel("minimum margin of safety (≥ 0 passes)")
    ax.grid(axis="y", visible=False)
    title(ax, "(d) Margins of safety by load case")

    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "load_cases.png", dpi=150)
    print("Saved results/load_case_results.csv and docs/load_cases.png")


if __name__ == "__main__":
    main()
