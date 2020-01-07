#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../.."

cd "$ROOT_DIR"
DATA_DIR="data/ner"

# DATASET must be defined by the calling script
python src/training/pytorch/ner/generate_wikiner_vectors.py "$DATA_DIR/$DATASET"
