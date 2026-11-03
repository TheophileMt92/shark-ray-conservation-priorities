#!/bin/bash
# Submits CAPTAIN 3 shark runs on Euler with job dependencies:
#   data preparation (only if not done yet) -> trainings on GPU -> predictions on CPU
# All jobs are submitted at once; each starts as soon as the job it needs is done.
#
#   bash ~/captain-ng/shark_data/captain3/cluster/submit_all.sh [training options]
#
# Choose runs with the environment variables SETUPS and METRICS (default: all), e.g.
# rerun only the worst-served setup:
#   SETUPS=protpop_worst bash ~/captain-ng/shark_data/captain3/cluster/submit_all.sh
#
# protpop_worst uses equal weights on its two reward terms (run names
# <metric>_protpop_worst_eq). For the calibrated variant run calibrate_sharks.py first and
# pass --calibrated here and to predict_sharks.py.
#
# Extra arguments are passed to every training job, e.g. --epochs 500. Use training-only
# options there (--epochs, --perturbations, --worst-weight, --min-sigma, ...): options that
# change the configuration or run name (--n-decisions, --max-species, --calibrated) must
# also be given to predict_sharks.py. Job logs (shark_c3_<job id>.out/.err) go to the
# current folder.

set -e
L="$(cd "$(dirname "$0")" && pwd)/launch_shark.sh"
SETUPS="${SETUPS:-protpop protpop_worst}"
METRICS="${METRICS:-IUCN FUSE EDGE2}"
RUNS="${SHARK_RUNS_DIR:-${SHARK_DIR:-$HOME/captain-ng/shark_data}/runs}"
sub() { sbatch --parsable "$@"; }
# Training: any GPU with >= 10 GB memory (the model needs ~2-3 GB), instead of waiting
# for an A100. No resume after a timeout: raise --time if runs get close to it.
GPU=(--gpus=1 --gres=gpumem:10g --cpus-per-task=1 --mem-per-cpu=16G --time=08:00:00)

# Check that Slurm accepts the GPU request before submitting anything
sbatch --test-only "${GPU[@]}" "$L" train_sharks.py > /dev/null 2>&1 || {
    echo "Slurm rejects the GPU request: ${GPU[*]}" >&2
    sbatch --test-only "${GPU[@]}" "$L" train_sharks.py
    exit 1
}

if [ -f "$RUNS/prepared/sdm_values.npy" ]; then
    echo "prepared data found in $RUNS/prepared"
    AFTER_PREP=()
else
    PREP=$(sub --time=00:30:00 "$L" prepare_shark_data.py)
    echo "prepare: $PREP"
    AFTER_PREP=(--dependency=afterok:$PREP)
fi

for METRIC in $METRICS; do
    for SETUP in $SETUPS; do
        TRAIN=$(sub "${AFTER_PREP[@]}" "${GPU[@]}" \
            "$L" train_sharks.py --metric $METRIC --setup $SETUP "$@")
        PRED=$(sub --dependency=afterok:$TRAIN --time=02:00:00 \
            "$L" predict_sharks.py --metric $METRIC --setup $SETUP)
        echo "$METRIC $SETUP: train $TRAIN, predict $PRED"
    done
done
