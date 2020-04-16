#! /bin/bash

set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd "$DIR"

DATA_NAME="sentiment" DATASET="train.csv" MODEL_NAME="sentiment" LANG="it" NUM_LABELS=4 SENTENCE_COLUMN="text" TARGET_COLUMN="sentiment" ../3-train.sh