#! /bin/bash

set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd "$DIR"

DATA_NAME="newsgroups" DATASET="train.csv" MODEL_NAME="newsgroups" LANG="en" NUM_LABELS=20 SENTENCE_COLUMN="sentence" TARGET_COLUMN="category" ../3-train.sh