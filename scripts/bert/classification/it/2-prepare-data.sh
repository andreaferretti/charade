#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/classification/sentiment"
DATASET="train.csv"

python src/training/bert/classification/convert_sentiment_dataset.py "$DATASET" "$DATA_DIR"