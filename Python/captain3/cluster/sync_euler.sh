#!/bin/bash
# Copies the CAPTAIN 3 shark scripts and inputs to Euler, or the results back.
# Run on the laptop, from anywhere:
#   bash Python/captain3/cluster/sync_euler.sh push   # scripts, SDMs and species table
#   bash Python/captain3/cluster/sync_euler.sh pull   # calibration and models -> runs/
#
# Remote layout (see launch_shark.sh):
#   $DEST/captain3/                    scripts (without runs/)
#   $DEST/data/tif_files_continental/  the 1,000 SDMs (R/Data/tif files continental)
#   $DEST/data/<species table>         R/Data/continental_shark_..._categories.csv
#   $DEST/runs/                        outputs written on Euler
# Override the host and folder with EULER and DEST.

set -euo pipefail
EULER="${EULER:-dsilvestro@euler.ethz.ch}"
DEST="${DEST:-/cluster/home/dsilvestro/captain-ng/shark_data}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"           # Python/captain3
REPO="$(cd "$HERE/../.." && pwd)"                  # manuscript repository
TABLE="continental_shark_conservation_metrics_10_harmonised_IUCN_categories.csv"
# One SSH connection shared by all transfers, so the login is asked only once
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=10m"

case "${1:-}" in
    push)
        $SSH "$EULER" "mkdir -p '$DEST/captain3' '$DEST/data/tif_files_continental' '$DEST/runs'"
        rsync -av -e "$SSH" --exclude 'runs/' --exclude '__pycache__/' "$HERE/" "$EULER:$DEST/captain3/"
        rsync -av -e "$SSH" "$REPO/R/Data/tif files continental/" "$EULER:$DEST/data/tif_files_continental/"
        rsync -av -e "$SSH" "$REPO/R/Data/$TABLE" "$EULER:$DEST/data/"
        ;;
    pull)
        mkdir -p "$HERE/runs"
        rsync -av -e "$SSH" --exclude 'prepared/' "$EULER:$DEST/runs/" "$HERE/runs/"
        ;;
    *)
        echo "Usage: $0 push|pull" >&2
        exit 1
        ;;
esac
