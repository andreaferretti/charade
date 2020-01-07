#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../.."

cd "$ROOT_DIR"
DATA_DIR="data/ner"

# DATASET must be defined by the calling script
URL="https://github.com/dice-group/FOX/raw/master/input/Wikiner/${DATASET}.bz2"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -f "$DATASET.bz2" ]; then rm "$DATASET.bz2"; fi
wget "$URL"

if [ -f "$DATASET" ]; then rm "$DATASET"; fi
bunzip2 "$DATASET.bz2"
