#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/lda"
MODEL_DIR="models/gensim/lda"
DATASET="newsgroups"

mkdir -p "$MODEL_DIR"

python src/training/gensim/lda/train.py \
  --data "$DATA_DIR/$DATASET.csv" \
  --num-topics 20 \
  --lang en \
  --model-dir "$MODEL_DIR" \
  --model-name "$DATASET"