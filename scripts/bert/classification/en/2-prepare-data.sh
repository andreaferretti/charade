#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/classification"
DATASET="newsgroups"

python src/training/bert/classification/convert_newsgroups_dataset.py "$DATASET" "$DATA_DIR"