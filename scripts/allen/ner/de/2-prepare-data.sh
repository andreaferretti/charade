#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/ner"
DATASET="aij-wikiner-de-wp2"

# aij-wikiner-de-wp2 has 459110 lines, we split it into 400000 lines for training,
# 30000 lines for validation and the rest (29110) for test.
cat "$DATA_DIR/$DATASET" | head -n 400000 > "$DATA_DIR/$DATASET.train"
cat "$DATA_DIR/$DATASET" | tail -n +400000 | head -n 30000 > "$DATA_DIR/$DATASET.validation"
cat "$DATA_DIR/$DATASET" | tail -n +430000 > "$DATA_DIR/$DATASET.test"