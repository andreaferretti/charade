#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/summarization"
DATASET="cnndm"
URL="https://s3.amazonaws.com/opennmt-models/Summary/"

mkdir -p "$DATA_DIR/$DATASET"
cd "$DATA_DIR/$DATASET"

wget "$URL$DATASET.tar.gz"

tar -xf "$DATASET.tar.gz" 