import io
import os
import zipfile
import numpy as np

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
zip_path = os.path.join(repo, "Python", "captain3_models.zip")
out_dir = os.path.join(repo, "R", "Data", "CAPTAIN3")
os.makedirs(out_dir, exist_ok=True)

species_table = os.path.join(repo, "R", "Data", "continental_shark_conservation_metrics_10_harmonised_IUCN_categories.csv")
with open(species_table) as f:
    rows = [line.split(",") for line in f.read().splitlines()]
header = [h.strip('"') for h in rows[0]]
table_species = [r[0].strip('"') for r in rows[1:]]
table_classes = {dim: [int(r[header.index(dim)]) for r in rows[1:]] for dim in ["IUCN", "FUSE", "EDGE2"]}

rewards = {"protpop": "protpop", "protpop_worst": "protpop_worst_eq_anneal"}
dimensions = ["IUCN", "FUSE", "EDGE2"]
targets = {"0.1": "predictions", "0.05": "predictions_target0.05", "0.2": "predictions_target0.2", "0.3": "predictions_target0.3"}


def read_predictions(z, names, folder):
    files = sorted(n for n in names if n.startswith(folder + "/replicate_") and n.endswith(".npz"))
    matrices, fracs = [], []
    for n in files:
        d = np.load(io.BytesIO(z.read(n)))
        if [str(s) for s in d["species"]] != table_species:
            raise ValueError(f"Species order in {n} differs from the species table")
        matrices.append(d["protection_matrix"])
        fracs.append(d["protected_range_fraction"])
    return len(files), np.mean(np.stack(matrices), axis=0).flatten(), np.mean(np.stack(fracs), axis=0)


with zipfile.ZipFile(zip_path) as z:
    names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
    for reward, model in rewards.items():
        priorities, fractions = {}, {}
        for dim in dimensions:
            for target, folder in targets.items():
                n, priority, fraction = read_predictions(z, names, f"models/{dim}_{model}/{folder}")
                priorities[(dim, target)] = priority
                fractions[(dim, target)] = fraction
                print(reward, dim, target, n, "replicates,", int((priority > 0).sum()), "cells with priority > 0")
            with open(os.path.join(out_dir, f"CAPTAIN3_{dim}_{reward}_priority_budget0.1_replicates50.csv"), "w") as f:
                f.write("PUID,Priority\n")
                for i, p in enumerate(priorities[(dim, "0.1")], start=1):
                    f.write(f"{i},{p:.4f}\n")
        with open(os.path.join(out_dir, f"CAPTAIN3_{reward}_protected_range_fractions.csv"), "w") as f:
            f.write("Species,EDGE2,FUSE,IUCN\n")
            for i, s in enumerate(table_species):
                f.write(f"{s},{fractions[('EDGE2', '0.1')][i]:.6f},{fractions[('FUSE', '0.1')][i]:.6f},{fractions[('IUCN', '0.1')][i]:.6f}\n")
        other = ["0.05", "0.2", "0.3"]
        cols = [(d, t) for d in dimensions for t in other]
        with open(os.path.join(out_dir, f"CAPTAIN3_{reward}_budget_sensitivity_priorities.csv"), "w") as f:
            f.write("PUID," + ",".join(f"{d}_{t}" for d, t in cols) + "\n")
            for i in range(len(priorities[cols[0]])):
                f.write(f"{i + 1}," + ",".join(f"{priorities[c][i]:.4f}" for c in cols) + "\n")
        with open(os.path.join(out_dir, f"CAPTAIN3_{reward}_budget_sensitivity_protected_range.csv"), "w") as f:
            f.write("Species," + ",".join(f"{d}_{t}" for d, t in cols) + "," + ",".join(f"{d}_class" for d in dimensions) + "\n")
            for i, s in enumerate(table_species):
                f.write(s + "," + ",".join(f"{fractions[c][i]:.6f}" for c in cols) + "," + ",".join(str(table_classes[d][i]) for d in dimensions) + "\n")
