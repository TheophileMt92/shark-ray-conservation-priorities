#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --job-name=shark_c3
#SBATCH --output=shark_c3_%j.out
#SBATCH --error=shark_c3_%j.err

# Runs one CAPTAIN 3 shark script with its arguments. CPU job by default (data preparation,
# calibration, prediction); request a GPU for training:
#   sbatch launch_shark.sh prepare_shark_data.py
#   sbatch launch_shark.sh calibrate_sharks.py --metric IUCN --setup protpop_worst
#   sbatch --gpus=1 --gres=gpumem:10g --time=08:00:00 launch_shark.sh train_sharks.py --metric IUCN --setup protpop
#   sbatch launch_shark.sh predict_sharks.py --metric IUCN --setup protpop
# submit_all.sh submits the whole pipeline with job dependencies.
#
# Layout on Euler (copied by sync_euler.sh; override with SHARK_DIR / CAPTAIN_DIR):
#   ~/captain-ng/                          captain-ng, virtual environment in cpt_venv/
#   ~/captain-ng/shark_data/captain3/      these scripts
#   ~/captain-ng/shark_data/data/          tif_files_continental/ and the species table
#   ~/captain-ng/shark_data/runs/          outputs (prepared data, calibration, models)

if [ $# -lt 1 ]; then
    echo "Usage: sbatch launch_shark.sh <script.py> [arguments]" >&2
    exit 1
fi

module load stack/2024-06 python/3.11.6

CAPTAIN_DIR="${CAPTAIN_DIR:-$HOME/captain-ng}"
SHARK_DIR="${SHARK_DIR:-$CAPTAIN_DIR/shark_data}"
export SHARK_DATA_DIR="${SHARK_DATA_DIR:-$SHARK_DIR/data}"
export SHARK_RUNS_DIR="${SHARK_RUNS_DIR:-$SHARK_DIR/runs}"

source "$CAPTAIN_DIR/cpt_venv/bin/activate"
export PYTHONPATH="${PYTHONPATH}:$CAPTAIN_DIR"
cd "$SHARK_DIR/captain3" || exit 1
echo "data: $SHARK_DATA_DIR | outputs: $SHARK_RUNS_DIR"

# Fail early if the installed captain predates static episodes (EpisodeRunner env_dynamics)
python -c "import inspect, captain as cn; \
assert 'env_dynamics' in inspect.signature(cn.EpisodeRunner).parameters; \
print('captain:', cn.__file__)" || { echo "captain-ng too old: git pull in $CAPTAIN_DIR" >&2; exit 1; }

python "$@"
