# Shark prioritization with CAPTAIN 3 (captain-ng)

Re-runs the three CAPTAIN 2 prioritizations (IUCN, FUSE, EDGE2; configs in
`captain_script_and_data.zip`, `predict_configs/`) with captain-ng and the corrected
species table (`R/Data/continental_shark_conservation_metrics_10_harmonised_IUCN_categories.csv`).

## Setup

Same inputs and settings as CAPTAIN 2:
- 1,000 continental SDMs (`R/Data/tif files continental`, 0.5°, 323 x 720 grid),
  suitability > 0.9 set to 1 and < 0.5 set to 0. Planning units are the 54,329 cells with at
  least one species; 10% of them (5,432 cells) are protected.
- No costs, no disturbance, static SDMs.
- Each metric is coded 1-5 per species (5 = highest priority), used as a fixed class.

What changed:
- **Episodes.** One time step without population dynamics
  (`EpisodeRunner(env_dynamics=False)`). The target is reached through 20 decisions of
  ~272 cells, and the features are recomputed after each decision (CAPTAIN 2 used 20
  protection steps of 271 cells, which protected 5,149 cells, 9.5%, because of rounding).
- **Reward setups** (`--setup`):
  - `protpop`: for each metric class, the mean protected fraction of the species'
    populations, weighted 1/8/16/32/64 for classes 1-5. This is the CAPTAIN 2
    `species_risk` reward up to a constant. It is a sum over protected cells, so the best plan
    is the cells with the highest value. `predict_sharks.py` reports this greedy plan
    as a reference for how close the trained policy gets.
  - `protpop_worst`: `protpop` plus the mean protected fraction of the 10% worst-served
    species (soft maximin, as in the AF corridor runs; all species count the same in
    this term). This setup needs a reward calibration per metric, and `--worst-weight`
    (default 1) sets the importance of the worst-served term relative to `protpop`.
- **Training.** Evolution strategies with z-scored advantages (the captain-ng default).
- **Features.**
  - Recomputed after each decision: species richness, population, species per metric
    class, protected neighbours, and (for `protpop_worst`) cells holding the worst-served
    species.
  - Static: rarity-weighted richness per class and the `protpop` value of each cell.
- **Replicates.** The weights of the last 50 epochs are each used for one prediction
  replicate, as CAPTAIN 2's `sample_weights_from_last_iter`.

## Running

Use the captain-ng environment, from this folder. The captain-ng version must include
`EpisodeRunner(env_dynamics=...)`.

```bash
CPT="uv run --project ~/Software/CAPTAIN/captain-ng python"
$CPT prepare_shark_data.py                     # once: writes runs/prepared/
$CPT calibrate_sharks.py --metric IUCN --setup protpop_worst
$CPT train_sharks.py     --metric IUCN --setup protpop
$CPT train_sharks.py     --metric IUCN --setup protpop_worst
$CPT predict_sharks.py   --metric IUCN --setup protpop
$CPT predict_sharks.py   --metric IUCN --setup protpop_worst
```

Repeat for `FUSE` and `EDGE2`. Run the full runs on the cluster (below).

### Euler

The scripts and inputs are copied to `~/captain-ng/shark_data` (captain-ng and its
`cpt_venv/` in `~/captain-ng`, as for the AF corridor runs):

```
~/captain-ng/shark_data/captain3/                  these scripts
~/captain-ng/shark_data/data/tif_files_continental/  the 1,000 SDMs
~/captain-ng/shark_data/data/continental_shark_..._categories.csv
~/captain-ng/shark_data/runs/                      outputs
```

1. On the laptop, push captain-ng `main`, then copy scripts and data:
   `bash Python/captain3/cluster/sync_euler.sh push`
2. On Euler, update captain-ng (`git -C ~/captain-ng pull`) and submit the whole pipeline
   (data preparation, 3 calibrations, 6 trainings on GPU, 6 predictions, with job
   dependencies): `bash ~/captain-ng/shark_data/captain3/cluster/submit_all.sh`.
   `cluster/launch_shark.sh` runs a single script (CPU; add `--gpus=a100:1` for training).
3. On the laptop, fetch the results into `runs/`: `bash Python/captain3/cluster/sync_euler.sh pull`

The launcher sets `SHARK_DATA_DIR` and `SHARK_RUNS_DIR`, which point the scripts to this
layout instead of the manuscript repository.

For quick local tests, add `--max-species 60` (a random subset of species) to every
command, plus e.g. `--epochs 5 --perturbations 4 --keep-weights 3` for training and
`--replicates 3` for prediction.

**Memory.** With all species one environment needs ~3 GB at peak. One episode takes
~2.5-3.5 s on a laptop CPU. On CPU each `--workers` process holds its own copy of the
environment, so on a 16 GB laptop use the default of 1 worker. A GPU runs the
perturbations of an epoch in sequence on a single environment.

## Outputs

Everything goes to `runs/` (git-ignored; set `SHARK_RUNS_DIR` to put it elsewhere):
- `prepared/`: preprocessed SDMs.
- `calibration/`: reward multipliers. The file names include a hash of the configuration.
- `models/<metric>_<setup>/`:
  - `settings.json`, `training_log.tsv`, `trained_weights.npy`
  - `weights_history/` (last 50 epochs)
  - `predictions/`:
    - `replicate_<i>.npz`: same keys as the CAPTAIN 2 outputs (`protection_matrix`,
      `protected_range_fraction`), plus `decision` (the decision that protected each cell)
      and `protected_pop_fraction`.
    - `CAPTAIN3_<metric>_<setup>_averaged_budget0.1_replicates50.csv`: PUID and Priority,
      the same columns and cell numbering as the CAPTAIN 2 `.rds` files in `R/Data`.
    - `protection_frequency.tif` and `mean_decision.tif`.
    - `species_protection.csv`: protected range and population fractions per species.
    - `greedy_baseline.npz` and `summary.json`: policy vs. greedy rewards, overlap with
      the greedy plan, species with < 10% of their range protected.
