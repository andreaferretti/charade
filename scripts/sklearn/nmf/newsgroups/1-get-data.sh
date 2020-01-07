#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/lda"
URL="https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

wget "$URL"