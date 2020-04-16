#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../.."

cd "$ROOT_DIR"
DATA_DIR="data/classification/"
MODEL_DIR="models/bert/classification"


EN="en"
if [ "$LANG" == "$EN" ]; then
    PRETRAINED_MODEL_NAME_OR_PATH="bert-base-uncased"
else
    PRETRAINED_MODEL_NAME_OR_PATH="models/bert/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"
fi

# DATA_NAME, DATASET, MODEL_NAME, LANG, NUM_LABELS, SENTENCE_COLUMN and TARGET_COLUMN must be defined in the calling script

mkdir -p "$MODEL_DIR"

python src/training/bert/classification/train.py \
  --data-dir "$DATA_DIR/" \
  --data "$DATASET" \
  --data-name "$DATA_NAME" \
  --num-labels "$NUM_LABELS" \
  --sentence-column "$SENTENCE_COLUMN" \
  --target-column "$TARGET_COLUMN" \
  --lang "$LANG" \
  --valid-size 0.2 \
  --num-epochs 2 \
  --alpha 0.00001 \
  --batch-size 16 \
  --random-state 0 \
  --do-lower-case True \
  --pretrained-model-name-or-path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --model-name "$MODEL_NAME" \
  --model-dir "$MODEL_DIR"
