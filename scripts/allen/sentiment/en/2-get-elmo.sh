#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/elmo"
BASE_URL="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo"
MODEL_SMALL="2x1024_128_2048cnn_1xhighway"
MODEL_MEDIUM="2x2048_256_2048cnn_1xhighway"
MODEL_ORIG="2x4096_512_2048cnn_2xhighway"
MODEL_ORIG_5B="2x4096_512_2048cnn_2xhighway_5.5B"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

wget "$BASE_URL/$MODEL_SMALL/elmo_${MODEL_SMALL}_weights.hdf5"
wget "$BASE_URL/$MODEL_SMALL/elmo_${MODEL_SMALL}_options.json"
wget "$BASE_URL/$MODEL_MEDIUM/elmo_${MODEL_MEDIUM}_weights.hdf5"
wget "$BASE_URL/$MODEL_MEDIUM/elmo_${MODEL_MEDIUM}_options.json"
wget "$BASE_URL/$MODEL_ORIG/elmo_${MODEL_ORIG}_weights.hdf5"
wget "$BASE_URL/$MODEL_ORIG/elmo_${MODEL_ORIG}_options.json"
wget "$BASE_URL/$MODEL_ORIG_5B/elmo_${MODEL_ORIG_5B}_weights.hdf5"
wget "$BASE_URL/$MODEL_ORIG_5B/elmo_${MODEL_ORIG_5B}_options.json"