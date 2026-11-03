#!/bin/bash
# Submits the full CAPTAIN 3 shark pipeline on Euler with job dependencies:
#   data preparation -> calibrations (protpop_worst, one per metric)
#   -> 6 trainings on GPU (3 metrics x 2 setups) -> 6 predictions
#
#   bash ~/captain-ng/shark_data/captain3/cluster/submit_all.sh [training options]
#
# Extra arguments are passed to every training job, e.g. --epochs 500. Use training-only
# options there (--epochs, --perturbations, --worst-weight, ...): options that change the
# configuration (--n-decisions, --max-species) must also be given to calibrate_sharks.py
# and predict_sharks.py. Job logs (shark_c3_<job id>.out/.err) go to the current folder.

set -e
L="$(cd "$(dirname "$0")" && pwd)/launch_shark.sh"
sub() { sbatch --parsable "$@"; }

PREP=$(sub --time=00:30:00 "$L" prepare_shark_data.py)
echo "prepare: $PREP"

for METRIC in IUCN FUSE EDGE2; do
    CAL=$(sub --dependency=afterok:$PREP --time=01:00:00 \
        "$L" calibrate_sharks.py --metric $METRIC --setup protpop_worst)
    echo "$METRIC calibration: $CAL"
    for SETUP in protpop protpop_worst; do
        DEP=$PREP
        [ $SETUP = protpop_worst ] && DEP=$CAL
        TRAIN=$(sub --dependency=afterok:$DEP --gpus=a100:1 --cpus-per-task=1 \
            --mem-per-cpu=16G --time=24:00:00 \
            "$L" train_sharks.py --metric $METRIC --setup $SETUP "$@")
        PRED=$(sub --dependency=afterok:$TRAIN --time=02:00:00 \
            "$L" predict_sharks.py --metric $METRIC --setup $SETUP)
        echo "$METRIC $SETUP: train $TRAIN, predict $PRED"
    done
done
