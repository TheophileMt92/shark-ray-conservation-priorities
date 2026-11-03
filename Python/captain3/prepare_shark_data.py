#!/usr/bin/env python
"""Preprocess the shark SDMs once for the CAPTAIN 3 runs.

Reads the 1,000 continental SDMs (R/Data/tif files continental) one at a time and
applies the CAPTAIN 2 settings: no-data and negative values -> 0, suitability above
FULL_SUITABILITY -> 1 and below MIN_SUITABILITY -> 0. Planning units are the cells
where at least one species remains (as in CAPTAIN 2). Writes to <SHARK_RUNS_DIR>/prepared:

    sdm_values.npy  (n_species, n_planning_units) float32, species in file-name order
    sdm_meta.npz    species names, grid rows/cols of the planning units, grid shape
    grid.json       raster transform and CRS (to write GeoTIFF outputs)

Also checks that every species is in the species table and prints the number of
species per class of each metric.

    uv run --project <captain-ng> python Python/captain3/prepare_shark_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shark_common as sc


def read_sdm(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    arr[~np.isfinite(arr) | (arr < 0)] = 0.0
    arr[arr > sc.FULL_SUITABILITY] = 1.0
    arr[arr < sc.MIN_SUITABILITY] = 0.0
    return arr


def main() -> None:
    files = sorted(sc.SDM_DIR.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No .tif files in {sc.SDM_DIR}")
    names = np.array([f.stem for f in files])

    traits = pd.read_csv(sc.TRAIT_FILE)
    missing = sorted(set(names) - set(traits[sc.SPECIES_COL]))
    if missing:
        raise ValueError(f"{len(missing)} SDM species not in {sc.TRAIT_FILE.name}: {missing[:5]}")
    extra = sorted(set(traits[sc.SPECIES_COL]) - set(names))
    if extra:
        print(f"Warning: {len(extra)} species in the table have no SDM: {extra[:5]}")

    with rasterio.open(files[0]) as src:
        grid_shape = (src.height, src.width)
        grid = {
            "height": src.height,
            "width": src.width,
            "transform": list(src.transform)[:6],
            "crs": src.crs.to_wkt() if src.crs else None,
        }

    # Pass 1: keep each species' range as sparse (flat index, value) pairs
    ranges = []
    occupied = np.zeros(grid_shape, dtype=bool)
    for i, f in enumerate(files):
        arr = read_sdm(f)
        if arr.shape != grid_shape:
            raise ValueError(f"{f.name}: grid {arr.shape} differs from {grid_shape}")
        idx = np.flatnonzero(arr)
        if len(idx) == 0:
            raise ValueError(f"{f.name}: no cell with suitability >= {sc.MIN_SUITABILITY}")
        ranges.append((idx, arr.ravel()[idx]))
        occupied.ravel()[idx] = True
        if (i + 1) % 100 == 0:
            print(f"  read {i + 1}/{len(files)} SDMs")

    # Planning units in row-major order (the order captain's flatten_grid uses)
    rows, cols = np.nonzero(occupied)
    pu_index = np.full(grid_shape[0] * grid_shape[1], -1, dtype=np.int64)
    pu_index[rows * grid_shape[1] + cols] = np.arange(len(rows))

    values = np.zeros((len(files), len(rows)), dtype=np.float32)
    for i, (idx, val) in enumerate(ranges):
        values[i, pu_index[idx]] = val

    sc.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(sc.PREPARED_DIR / "sdm_values.npy", values)
    np.savez(
        sc.PREPARED_DIR / "sdm_meta.npz",
        names=names,
        rows=rows,
        cols=cols,
        grid_shape=np.array(grid_shape),
    )
    with open(sc.PREPARED_DIR / "grid.json", "w") as f:
        json.dump(grid, f, indent=2)

    range_cells = (values > 0).sum(axis=1)
    print(
        f"{len(files)} species, grid {grid_shape}, {len(rows)} planning units "
        f"(10% = {int(np.floor(sc.TARGET_FRACTION * len(rows)))} cells)"
    )
    print(
        f"Range size (cells): min {range_cells.min()}, median {np.median(range_cells):.0f}, "
        f"max {range_cells.max()}"
    )
    tbl = traits.set_index(sc.SPECIES_COL).loc[names]
    for metric in sc.METRICS:
        counts = tbl[metric].value_counts().reindex(range(1, sc.N_CLASSES + 1), fill_value=0)
        print(f"{metric:6s} species per class 1-5: {counts.tolist()}")
    print(f"Saved to {sc.PREPARED_DIR}")


if __name__ == "__main__":
    main()
