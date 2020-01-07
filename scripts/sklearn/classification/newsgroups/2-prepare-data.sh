#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/lda"
DATASET="newsgroups"

python src/training/gensim/lda/json_to_tsv.py "$DATASET" "$DATA_DIR"