#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../.."

cd "$ROOT_DIR"
GUTENBERG_CORPUS="carroll-alice.txt"
DATA_DIR="data/next_sentence_prediction"
DATASET="alice_NSP"

mkdir -p "$DATA_DIR"

python src/training/bert/next_sentence_prediction/generate_dataset.py "$GUTENBERG_CORPUS" "$DATASET" "$DATA_DIR"