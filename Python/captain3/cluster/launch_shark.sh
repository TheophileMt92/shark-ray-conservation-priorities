#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --job-name=shark_c3
#SBATCH --output=shark_c3_%j.out
#SBATCH --error=shark_c3_%j.err

# Runs one CAPTAIN 3 shark script with its arguments, e.g.
#   sbatch launch_shark.sh prepare_shark_data.py
#   sbatch launch_shark.sh calibrate_sharks.py --metric IUCN --setup protpop_worst
#   sbatch launch_shark.sh train_sharks.py --metric IUCN --setup protpop
#   sbatch launch_shark.sh predict_sharks.py --metric IUCN --setup protpop
# submit_all.sh submits the whole pipeline with job dependencies.
#
# Expected layout (override with SHARK_REPO / CAPTAIN_DIR / SHARK_RUNS_DIR):
#   ~/CAPTAIN_elasmo_manuscript/   this repository (SDMs and species table in R/Data)
#   ~/captain-ng/                  captain-ng, with its virtual environment in cpt_venv/
# Outputs go to $SHARK_RUNS_DIR (default: Python/captain3/runs in this repository).

if [ $# -lt 1 ]; then
    echo "Usage: sbatch launch_shark.sh <script.py> [arguments]" >&2
    exit 1
fi

module load stack/2024-06 python/3.11.6

SHARK_REPO="${SHARK_REPO:-$HOME/CAPTAIN_elasmo_manuscript}"
CAPTAIN_DIR="${CAPTAIN_DIR:-$HOME/captain-ng}"
source "$CAPTAIN_DIR/cpt_venv/bin/activate"
export PYTHONPATH="${PYTHONPATH}:$CAPTAIN_DIR"
cd "$SHARK_REPO/Python/captain3" || exit 1

# Fail early if the installed captain predates static episodes (EpisodeRunner env_dynamics)
python -c "import inspect, captain as cn; \
assert 'env_dynamics' in inspect.signature(cn.EpisodeRunner).parameters; \
print('captain:', cn.__file__)" || { echo "captain-ng too old: update it" >&2; exit 1; }

python "$@"
