#!/usr/bin/env python
"""Compare the CAPTAIN 3 priority maps across metrics and reward setups.

Uses the predictions of six runs: protpop and protpop_worst (annealed, run names
<metric>_protpop_worst_eq_anneal) for IUCN, FUSE and EDGE2. A cell's priority is the
share of the 50 replicates that protect it; the "core" plan of a run is the cells
protected in at least half of the replicates. Writes to <SHARK_RUNS_DIR>/comparison:

    similarity.csv          Jaccard index of the core plans of every pair of runs
    plans.csv               per run: core cells, cells protected in every replicate
    latitude.csv            per run and 5 degree latitude band: share of the planning
                            units in the core plan
    fig1_priority_maps.png  priority (share of replicates) of the six runs
    fig2_overlap_maps.png   per metric: cells in both setups' core plans or one only
    fig3_similarity.png     Jaccard index of the core plans
    fig4_latitude.png       share of planning units protected by latitude
    range_by_class.csv      per run and class of its own metric: species, mean (sd, se)
                            protected range fraction
    species_below.csv       per run: species with 0%, <5%, <10%, <15% of their range
                            protected (mean, min, max over replicates)
    fig5_range_by_class.png mean protected range fraction by class of the run's metric
    fig6_species_below.png  species with little or no protected range

Protected range fraction: share of a species' range cells (suitability >= 0.5) that are
protected, per replicate (protected_range_fraction in the replicate files), averaged over
the replicates for the class means.

    python compare_maps.py [--worst-run-suffix protpop_worst_eq_anneal]
"""

from __future__ import annotations

import argparse
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import shark_common as sc  # noqa: E402

# Reference palette (dataviz skill): surface, ink, sequential blue, categorical slots 1-3
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
UNPROTECTED = "#dddcd6"  # planning units not in the plan (neutral, recedes)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (validated all-pairs)
SETUP_LABEL = {"protpop": "protpop", "protpop_worst": "protpop + worst-served"}
# IUCN Red List category colours (LC, NT, VU, EN, CR), also used for FUSE / EDGE2 classes 1-5
REDLIST = ["#60C659", "#CCE226", "#F9E814", "#FC7F3F", "#D81E05"]
HATCH = "////"  # second model (protpop + worst-served) in fig 5

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.edgecolor": GRID,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)


def load_run(run: str) -> dict:
    """Priority (share of replicates) and core plan of a run, on the full grid."""
    pred = sc.MODELS_DIR / run / "predictions"
    with rasterio.open(pred / "protection_frequency.tif") as src:
        freq = src.read(1)
    reps = sorted(pred.glob("replicate_*.npz"))
    if not reps:
        raise FileNotFoundError(f"No predictions in {pred}")
    data = [np.load(f) for f in reps]
    prot = np.array([d["protection_matrix"] > 0 for d in data])
    return {
        "freq": freq,
        "core": freq >= 0.5,
        "always": prot.all(axis=0),
        "n_rep": len(reps),
        "range_frac": np.array([d["protected_range_fraction"] for d in data]),  # (rep, sp)
        "species": data[0]["species"].astype(str),
    }


CLASS_LABELS = {
    "IUCN": ["LC", "NT", "VU", "EN", "CR"],
    "FUSE": ["1\n0-0.2", "2\n0.2-0.4", "3\n0.4-0.6", "4\n0.6-0.8", "5\n0.8-1"],
    "EDGE2": ["1\n0-0.2", "2\n0.2-0.4", "3\n0.4-0.6", "4\n0.6-0.8", "5\n0.8-1"],
}
THRESHOLDS = [
    ("0%", lambda f: f == 0),
    ("<5%", lambda f: f < 0.05),
    ("<10%", lambda f: f < 0.10),
    ("<15%", lambda f: f < 0.15),
]


def bar_axis(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def species_plots(runs: dict, out) -> None:
    """Protected range by metric class (fig 5) and species with little protection (fig 6)."""
    setups = ("protpop", "protpop_worst")
    width = 0.38
    class_rows, below_rows = [], []
    for metric in sc.METRICS:
        status = sc.load_species_status(metric, runs[(metric, "protpop")]["species"]) + 1
        for setup in setups:
            r = runs[(metric, setup)]
            frac = r["range_frac"].mean(axis=0)  # per species, over replicates
            for k in range(1, sc.N_CLASSES + 1):
                f = frac[status == k]
                class_rows.append(
                    {
                        "metric": metric,
                        "setup": setup,
                        "class": k,
                        "label": CLASS_LABELS[metric][k - 1].replace("\n", " "),
                        "n_species": len(f),
                        "mean": f.mean() if len(f) else np.nan,
                        "sd": f.std(ddof=1) if len(f) > 1 else np.nan,
                        "se": f.std(ddof=1) / np.sqrt(len(f)) if len(f) > 1 else np.nan,
                    }
                )
            for label, test in THRESHOLDS:
                counts = test(r["range_frac"]).sum(axis=1)  # per replicate
                below_rows.append(
                    {
                        "metric": metric,
                        "setup": setup,
                        "threshold": label,
                        "mean": counts.mean(),
                        "min": counts.min(),
                        "max": counts.max(),
                    }
                )
    by_class = pd.DataFrame(class_rows)
    below = pd.DataFrame(below_rows)
    by_class.to_csv(out / "range_by_class.csv", index=False)
    below.to_csv(out / "species_below.csv", index=False)

    # --- fig 5: mean protected range by class ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, metric in zip(axes, sc.METRICS, strict=True):
        x = np.arange(sc.N_CLASSES)
        for i, setup in enumerate(setups):
            t = by_class[(by_class.metric == metric) & (by_class.setup == setup)]
            # colour = class (Red List colours); hatching = protpop + worst-served
            ax.bar(
                x + (i - 0.5) * width,
                100 * t["mean"],
                width,
                color=REDLIST,
                edgecolor=SURFACE,
                linewidth=1,
                hatch=HATCH if setup == "protpop_worst" else None,
                # 95% CI of the mean over species, capped to 0-100%
                yerr=100
                * np.vstack(
                    [
                        t["mean"] - (t["mean"] - 1.96 * t["se"].fillna(0)).clip(lower=0),
                        (t["mean"] + 1.96 * t["se"].fillna(0)).clip(upper=1) - t["mean"],
                    ]
                ),
                error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 0},
            )
        n = by_class[(by_class.metric == metric) & (by_class.setup == "protpop")]["n_species"]
        ax.set_xticks(
            x,
            [f"{lab}\nn={k}" for lab, k in zip(CLASS_LABELS[metric], n, strict=True)],
            fontsize=7.5,
        )
        ax.axhline(10, color=INK_2, lw=0.8, ls=(0, (3, 3)))
        ax.set_title(f"{metric} model, by {metric} class", loc="left")
        bar_axis(ax)
    axes[0].set_ylabel("Mean protected range (% of range cells)")
    plt.rcParams["hatch.color"] = SURFACE
    axes[-1].legend(
        handles=[
            Patch(facecolor=INK_2, edgecolor=SURFACE, label=SETUP_LABEL["protpop"]),
            Patch(
                facecolor=INK_2, edgecolor=SURFACE, hatch=HATCH, label=SETUP_LABEL["protpop_worst"]
            ),
        ],
        loc="upper left",
        fontsize=8,
    )
    fig.suptitle("Protected share of species ranges by class", x=0.01, ha="left", fontweight="bold")
    fig.text(
        0.01,
        -0.03,
        "Mean over species with 95% CI. Colours: IUCN Red List categories (LC to CR), applied "
        "to FUSE and EDGE2 classes 1-5 in the same order. Dashed line: 10% of planning units.",
        fontsize=8,
        color=INK_2,
    )
    fig.savefig(out / "fig5_range_by_class.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- fig 6: species with little or no protected range ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True, layout="constrained")
    labels = [t[0] for t in THRESHOLDS]
    for ax, metric in zip(axes, sc.METRICS, strict=True):
        x = np.arange(len(labels))
        for i, (setup, color) in enumerate(zip(setups, SERIES[:2], strict=True)):
            t = below[(below.metric == metric) & (below.setup == setup)]
            xs = x + (i - 0.5) * width
            ax.bar(
                xs,
                t["mean"],
                width,
                color=color,
                edgecolor=SURFACE,
                linewidth=1,
                label=SETUP_LABEL[setup],
                yerr=[t["mean"] - t["min"], t["max"] - t["mean"]],
                error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 0},
            )
            for xi, v, hi in zip(xs, t["mean"], t["max"], strict=True):
                ax.text(xi, hi + 4, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Protected share of the range")
        ax.set_title(f"{metric} model", loc="left")
        bar_axis(ax)
    axes[0].set_ylabel("Species (of 1,000)")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Species with little or no protected range (mean over replicates, range min-max)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.savefig(out / "fig6_species_below.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nMean protected range fraction (%) by class:")
    print(
        (100 * by_class.pivot_table(index=["metric", "class"], columns="setup", values="mean"))
        .round(1)
        .to_string()
    )
    print("\nSpecies below thresholds (mean over replicates):")
    print(
        below.pivot_table(index=["metric", "threshold"], columns="setup", values="mean", sort=False)
        .round(1)
        .to_string()
    )


def grid_extent(grid: dict) -> tuple[list[float], np.ndarray]:
    a, _, c, _, e, f = grid["transform"]
    lat = f + e * (np.arange(grid["height"]) + 0.5)
    return [c, c + a * grid["width"], f + e * grid["height"], f], lat


def map_axis(ax, extent):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(-80, 85)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-60, 81, 30))
    ax.tick_params(length=0, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--worst-run-suffix", default="protpop_worst_eq_anneal")
    args = parser.parse_args()

    out = sc.RUNS_ROOT / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    with open(sc.PREPARED_DIR / "grid.json") as f:
        extent, lat = grid_extent(json.load(f))

    runs = {}
    for setup in ("protpop", "protpop_worst"):
        for metric in sc.METRICS:
            name = (
                f"{metric}_protpop" if setup == "protpop" else f"{metric}_{args.worst_run_suffix}"
            )
            runs[(metric, setup)] = {"run": name, **load_run(name)}
    keys = list(runs)
    pu = np.isfinite(runs[keys[0]]["freq"])  # planning units
    n_pu = int(pu.sum())

    # --- tables ---------------------------------------------------------------------
    pd.DataFrame(
        [
            {
                "metric": m,
                "setup": s,
                "run": r["run"],
                "replicates": r["n_rep"],
                "core_cells": int(r["core"].sum()),
                "always_protected": int(r["always"].sum()),
                "partly_protected": int(((r["freq"] > 0) & (r["freq"] < 1)).sum()),
            }
            for (m, s), r in runs.items()
        ]
    ).to_csv(out / "plans.csv", index=False)

    jac = np.zeros((len(keys), len(keys)))
    rows = []
    for i, j in itertools.product(range(len(keys)), repeat=2):
        a, b = runs[keys[i]]["core"], runs[keys[j]]["core"]
        jac[i, j] = (a & b).sum() / (a | b).sum()
        if i < j:
            rows.append(
                {
                    "run_a": runs[keys[i]]["run"],
                    "run_b": runs[keys[j]]["run"],
                    "jaccard_core": round(jac[i, j], 4),
                    "shared_cells": int((a & b).sum()),
                    "core_a": int(a.sum()),
                    "core_b": int(b.sum()),
                }
            )
    pd.DataFrame(rows).to_csv(out / "similarity.csv", index=False)

    bands = np.arange(-80, 90, 5)
    band = np.digitize(np.broadcast_to(lat[:, None], pu.shape), bands)
    lat_rows = []
    for (m, s), r in runs.items():
        for k in np.unique(band[pu]):
            in_band = pu & (band == k)
            lat_rows.append(
                {
                    "metric": m,
                    "setup": s,
                    "lat_from": bands[k - 1],
                    "planning_units": int(in_band.sum()),
                    "share_protected": r["core"][in_band].mean(),
                }
            )
    lat_tbl = pd.DataFrame(lat_rows)
    lat_tbl.to_csv(out / "latitude.csv", index=False)

    # --- fig 1: priority maps --------------------------------------------------------
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.2), layout="constrained")
    for row, metric in enumerate(sc.METRICS):
        for col, setup in enumerate(("protpop", "protpop_worst")):
            ax, r = axes[row, col], runs[(metric, setup)]
            base = np.where(pu, 1.0, np.nan)
            ax.imshow(
                base, extent=extent, cmap=ListedColormap([UNPROTECTED]), interpolation="nearest"
            )
            shown = np.where(r["freq"] > 0, r["freq"], np.nan)
            im = ax.imshow(shown, extent=extent, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            map_axis(ax, extent)
            ax.set_title(f"{metric} · {SETUP_LABEL[setup]}", loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.5, aspect=30, pad=0.01)
    cb.set_label("Share of replicates protecting the cell")
    cb.outline.set_visible(False)
    fig.legend(
        handles=[Patch(color=UNPROTECTED, label="Planning unit, not protected")],
        loc="lower left",
        bbox_to_anchor=(0.01, -0.02),
    )
    fig.suptitle(
        "Priority maps (10% of planning units, 50 replicates)", x=0.01, ha="left", fontweight="bold"
    )
    fig.savefig(out / "fig1_priority_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- fig 2: overlap of the two setups per metric -----------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), layout="constrained")
    cats = ListedColormap([UNPROTECTED, SERIES[0], SERIES[1], SERIES[2]])
    for ax, metric in zip(axes, sc.METRICS, strict=True):
        a, b = runs[(metric, "protpop")]["core"], runs[(metric, "protpop_worst")]["core"]
        code = np.full(pu.shape, np.nan)
        code[pu] = 0
        code[a & b], code[a & ~b], code[~a & b] = 1, 2, 3
        ax.imshow(code, extent=extent, cmap=cats, vmin=-0.5, vmax=3.5, interpolation="nearest")
        map_axis(ax, extent)
        j = (a & b).sum() / (a | b).sum()
        ax.set_title(f"{metric}: core plans of the two setups (Jaccard {j:.2f})", loc="left")
        ax.legend(
            handles=[
                Patch(color=SERIES[0], label=f"Both ({(a & b).sum():,} cells)"),
                Patch(color=SERIES[1], label=f"protpop only ({(a & ~b).sum():,})"),
                Patch(color=SERIES[2], label=f"protpop + worst-served only ({(~a & b).sum():,})"),
            ],
            loc="lower left",
            fontsize=8,
            ncols=3,
            bbox_to_anchor=(0, -0.2),
        )
    fig.savefig(out / "fig2_overlap_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- fig 3: Jaccard matrix -------------------------------------------------------------
    labels = [f"{m}\n{'protpop' if s == 'protpop' else '+ worst'}" for m, s in keys]
    fig, ax = plt.subplots(figsize=(5.6, 4.8), layout="constrained")
    im = ax.imshow(jac, cmap=cmap, vmin=0, vmax=1)
    for i, j in itertools.product(range(len(keys)), repeat=2):
        ax.text(
            j,
            i,
            f"{jac[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=SURFACE if jac[i, j] > 0.55 else INK,
        )
    ax.set_xticks(range(len(keys)), labels, fontsize=7)
    ax.set_yticks(range(len(keys)), labels, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Jaccard index of core plans")
    cb.outline.set_visible(False)
    ax.set_title("Overlap of the core plans", loc="left")
    fig.savefig(out / "fig3_similarity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- fig 4: latitude profile ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True, layout="constrained")
    for ax, metric in zip(axes, sc.METRICS, strict=True):
        for s, color in zip(("protpop", "protpop_worst"), SERIES[:2], strict=True):
            t = lat_tbl[(lat_tbl.metric == metric) & (lat_tbl.setup == s)]
            y = t.lat_from + 2.5
            ax.plot(100 * t.share_protected, y, color=color, lw=2, label=SETUP_LABEL[s])
        ax.set_title(metric, loc="left")
        ax.set_xlabel("Planning units protected (%)")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Latitude (5° bands)")
    axes[0].set_yticks(range(-60, 81, 20))
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Share of planning units in the core plan, by latitude",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.savefig(out / "fig4_latitude.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    species_plots(runs, out)

    print(f"\n{n_pu} planning units; core plans:")
    print(pd.read_csv(out / "plans.csv").to_string(index=False))
    print("\nJaccard of core plans:")
    print(
        pd.DataFrame(
            jac,
            index=[r["run"] for r in runs.values()],
            columns=[f"{m}/{'pp' if s == 'protpop' else 'pp+w'}" for m, s in keys],
        )
        .round(2)
        .to_string()
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
