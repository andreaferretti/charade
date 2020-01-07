#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../.."

cd "$ROOT_DIR"
DATA_DIR="data/ner"
MODEL_DIR="models/pytorch/ner"

# DATASET and MODEL_LANG must be defined by the calling script

mkdir -p "$MODEL_DIR"

python src/training/pytorch/ner/train.py \
  --train-words "$DATA_DIR/$DATASET-train-words.pth" \
  --train-tags "$DATA_DIR/$DATASET-train-tags.pth" \
  --test-words "$DATA_DIR/$DATASET-test-words.pth" \
  --test-tags "$DATA_DIR/$DATASET-test-tags.pth" \
  --word-index "$DATA_DIR/$DATASET-words.index" \
  --tag-index "$DATA_DIR/$DATASET-tags.index" \
  --results "$MODEL_DIR/$MODEL_LANG-results.json" \
  --model "$MODEL_DIR/$MODEL_LANG.pth" \
  --hidden-dim 300 \
  --num-layers 2 \
  --learning-rate 0.003 \
  --momentum 0.024 \
  --dropout 0.5 \
  --num-epochs 300

# Copy the indices files, which are needed at inference time,
# under the `models` directory
cp "$DATA_DIR/$DATASET-words.index" "$MODEL_DIR/$MODEL_LANG-words.index"
cp "$DATA_DIR/$DATASET-tags.index" "$MODEL_DIR/$MODEL_LANG-tags.index"
