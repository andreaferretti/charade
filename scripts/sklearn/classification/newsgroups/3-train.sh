#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/lda"
MODEL_DIR="models/sklearn/classification"
DATASET="newsgroups"

mkdir -p "$MODEL_DIR"

python src/training/sklearn/classification/train.py \
  --data "$DATA_DIR/$DATASET.csv" \
  --test \
  --bootstrap 400 \
  --topk 20 \
  --column 2 \
  --freq-threshold 6 \
  --language en \
  --model-dir "$MODEL_DIR" \
  --model-name "$DATASET"