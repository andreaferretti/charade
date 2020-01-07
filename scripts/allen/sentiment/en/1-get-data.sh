#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/sentiment"
DATASET="SST"
URL="https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

wget "$URL" -O "$DATASET.zip"
unzip "$DATASET.zip" -d "$DATASET"