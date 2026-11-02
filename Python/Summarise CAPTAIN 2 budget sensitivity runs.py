"""
Summarises the CAPTAIN 2 predictions at protection targets of 5%, 20% and 30%
(models trained at 10%, predictions by D. Silvestro, October 2026).

The raw predictions are read directly from Python/captain_budget_sensitivity_runs.zip
(no need to unzip). For each dimension (IUCN, FUSE, EDGE2) and target, the priority
of each grid cell is the frequency at which it was selected across the 50
predictions, as for the main 10% results ("Analyse CAPTAIN 2 outputs ... 220725.py").

Outputs (R/Data):
  CAPTAIN2_budget_sensitivity_priorities.csv
  One row per grid cell selected at least once in any run (PUID = cell number
  of the 0.5-degree grid, as in the 10% results), one column per run
  (e.g. IUCN_0.05 = priority of the cell for IUCN at a 5% target).

  CAPTAIN2_budget_sensitivity_protected_range.csv
  One row per species (same order as CAPTAIN2_protected_range_fractions_2ndrun.rds),
  with the fraction of its range protected averaged across the 50 predictions
  (e.g. IUCN_0.05) and its class in each dimension (e.g. IUCN_class, 0 = highest
  class, 4 = lowest class).

Usage (from the repository root):
  python "Python/Summarise CAPTAIN 2 budget sensitivity runs.py"
"""
import io
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
zip_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    repo_dir, "Python", "captain_budget_sensitivity_runs.zip")
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(repo_dir, "R", "Data")

dims = {"iucn": "IUCN", "fuse": "FUSE", "edge": "EDGE2"}
targets = {"pt005": "0.05", "pt02": "0.2", "pt03": "0.3"}

with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    columns = {}
    range_fractions = {}
    for dim_code, dim_name in dims.items():
        for tgt_code, tgt in targets.items():
            pattern = re.compile(rf"results/{dim_code}_pred_{tgt_code}_20250720/.*_stats_\d+\.npz$")
            files = sorted(n for n in names if pattern.search(n))
            if len(files) == 0:
                sys.exit(f"No predictions found for {dim_name} at {tgt}")
            matrices = np.array([
                np.load(io.BytesIO(zf.read(f)))["protection_matrix"].flatten() for f in files])
            columns[f"{dim_name}_{tgt}"] = matrices.mean(axis=0)
            stats = [np.load(io.BytesIO(zf.read(f))) for f in files]
            range_fractions[f"{dim_name}_{tgt}"] = np.mean(
                [s["protected_range_fraction"] for s in stats], axis=0)
            range_fractions[f"{dim_name}_class"] = stats[0]["simulated_init_status"]
            print(f"{dim_name} {tgt}: {len(files)} predictions, "
                  f"{matrices.sum(axis=1).mean():.0f} cells selected per prediction, "
                  f"{(columns[f'{dim_name}_{tgt}'] > 0).sum()} cells selected at least once")

df = pd.DataFrame(columns)
df.insert(0, "PUID", np.arange(1, len(df) + 1))
df = df[(df.drop(columns="PUID") > 0).any(axis=1)]
out = os.path.join(out_dir, "CAPTAIN2_budget_sensitivity_priorities.csv")
df.to_csv(out, index=False, float_format="%.4f")
print(f"Saved {len(df)} cells to {out}")

species = pd.DataFrame({"Species": [f"Species_{i + 1}" for i in range(len(range_fractions["IUCN_0.05"]))]})
for key in sorted(range_fractions, key=lambda k: (k.split("_")[0], k.endswith("class"), k)):
    species[key] = range_fractions[key]
out = os.path.join(out_dir, "CAPTAIN2_budget_sensitivity_protected_range.csv")
species.to_csv(out, index=False)  # full precision, so that species with no protection keep a value of exactly 0
print(f"Saved {len(species)} species to {out}")
