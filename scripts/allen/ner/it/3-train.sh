#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/ner"
MODEL_DIR="models/allen/ner"
DATASET="aij-wikiner-it-wp2"

mkdir -p "$MODEL_DIR"

PYTHONPATH=src allennlp train "$DIR"/crf_tagger.json -s "$MODEL_DIR"/it \
  --include-package common.allen.ner