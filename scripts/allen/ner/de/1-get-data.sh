#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/ner"
DATASET="aij-wikiner-de-wp2"
URL="https://github.com/dice-group/FOX/blob/master/input/Wikiner/aij-wikiner-de-wp2.bz2?raw=true"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

wget "$URL"
mv "$DATASET.bz2?raw=true" "$DATASET.bz2"
bunzip2 "$DATASET.bz2"