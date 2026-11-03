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

case "${1:-}" in
    push)
        ssh "$EULER" "mkdir -p '$DEST/captain3' '$DEST/data/tif_files_continental' '$DEST/runs'"
        rsync -av --exclude 'runs/' --exclude '__pycache__/' "$HERE/" "$EULER:$DEST/captain3/"
        rsync -av "$REPO/R/Data/tif files continental/" "$EULER:$DEST/data/tif_files_continental/"
        rsync -av "$REPO/R/Data/$TABLE" "$EULER:$DEST/data/"
        ;;
    pull)
        mkdir -p "$HERE/runs"
        rsync -av --exclude 'prepared/' "$EULER:$DEST/runs/" "$HERE/runs/"
        ;;
    *)
        echo "Usage: $0 push|pull" >&2
        exit 1
        ;;
esac
