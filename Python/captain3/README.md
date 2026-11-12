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
    this term). Both terms are protected fractions in [0, 1], so they are added with
    equal weights; `--worst-weight` (default 1) changes the weight of the worst-served
    term. This objective has no simple optimum: `predict_sharks.py` reports a greedy plan
    for it as a reference, not as the best possible plan.

    The first runs (Nov 2026, run names `<metric>_protpop_worst`) scaled the two terms
    with a reward calibration (`calibrate_sharks.py`, still available with
    `--calibrated`). The calibration probes the untrained policy, which leaves the
    worst-served species unprotected. That term then barely varies, so its multiplier
    came out 28-200x larger than for `protpop`. Those runs gave up much of `protpop` and
    reached only 70-81% of a greedy plan for their own objective.

    **Weight annealing.** With equal weights from the start, the FUSE and EDGE2 runs
    (`<metric>_protpop_worst_eq`) converged to the pure `protpop` plan. That plan
    leaves 141-149 species with no protection. The worst-served term averages the 100
    least protected species, so it stayed exactly 0 and gave no training signal. IUCN
    was just below that limit (94 species), so it learned both terms and reached 98% of
    the greedy plan. Training now starts with weight 100 on the worst-served term and
    lowers it geometrically to `--worst-weight` (1) over the first 100 epochs
    (`--worst-weight-start`, `--anneal-epochs`; run names `..._eq_anneal`). The reward
    in `training_log.tsv` follows the annealed weights during that phase, so compare the
    `r_protpop` and `r_worst` columns instead; `worst_weight.tsv` lists the weight per
    epoch.
- **Training.** Evolution strategies with z-scored advantages (the captain-ng default).
  The ES noise sigma is kept at or above 0.05 (`--min-sigma`). With z-scored advantages
  the step size scales as learning rate / sigma. In the first `protpop_worst` runs
  sigma fell to about 0.01, and the weights moved 4-25x further per epoch than the
  perturbations explored.
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
CPT="uv run --project /path/to/captain-ng python"
$CPT prepare_shark_data.py                     # once: writes runs/prepared/
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
2. On Euler, update captain-ng (`git -C ~/captain-ng pull`) and submit the runs (data
   preparation if not done yet, 6 trainings on GPU, 6 predictions, with job
   dependencies): `bash ~/captain-ng/shark_data/captain3/cluster/submit_all.sh`.
   To run only some of them, set `SETUPS` and/or `METRICS`. For example, rerun the
   worst-served setup with `SETUPS=protpop_worst bash .../submit_all.sh`.
   The trainings request any GPU with at least 10 GB of memory (`--gpus=1
   --gres=gpumem:10g`, 8 h), which needs much less queueing than an A100.
   `cluster/launch_shark.sh` runs a single script (CPU; add the GPU options for training).
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
- `calibration/`: reward multipliers, only for `--calibrated` runs. The file names include
  a hash of the configuration.
- `models/<run name>/`, named `<metric>_protpop` or `<metric>_protpop_worst_` +
  `eq` (equal weights) / `cal` (calibrated), + `_anneal` (annealed worst-served weight,
  the default):
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
    - `greedy_combined.npz` (`protpop_worst` only): the greedy plan for the run's own
      weighted objective. `summary.json` also reports the policy's objective as a
      fraction of this plan's (`policy_objective_vs_greedy`).
