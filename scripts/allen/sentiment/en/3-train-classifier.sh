#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
MODEL_DIR="models/allen/sentiment"
ELMO_DIR="models/elmo"
LANG="en"

mkdir -p "$MODEL_DIR"

PYTHONPATH=src allennlp train "$DIR"/sst_classifier_elmo.json -s "$MODEL_DIR/$LANG" \
  --include-package common.allen.sentiment

if [ ! -d "$ELMO_DIR" ]; then
  cp -r data/elmo "$ELMO_DIR"
fi

CONFIG_FILE="$MODEL_DIR/en/config.json"
TMP_FILE="$MODEL_DIR/en/config.json.tmp"

cat "$CONFIG_FILE" | jq '
  .model.word_embeddings.tokens.options_file |= "models/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json" |
  .model.word_embeddings.tokens.weight_file |= "models/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
  ' > "$TMP_FILE"
mv "$TMP_FILE" "$CONFIG_FILE"