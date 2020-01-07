#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/ner"
DATASET="aij-wikiner-it-wp2"

# aij-wikiner-it-wp2 has 153250 lines, we split it into 120000 lines for training,
# 15000 lines for validation and the rest (18250) for test.
cat "$DATA_DIR/$DATASET" | head -n 120000 > "$DATA_DIR/$DATASET.train"
cat "$DATA_DIR/$DATASET" | tail -n +120000 | head -n 15000 > "$DATA_DIR/$DATASET.validation"
cat "$DATA_DIR/$DATASET" | tail -n +135000 > "$DATA_DIR/$DATASET.test"