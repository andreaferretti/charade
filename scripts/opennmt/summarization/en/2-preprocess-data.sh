#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATASET="cnndm"
DATA_DIR="data/summarization/$DATASET"

python $DIR/preprocess.py -train_src $DATA_DIR/train.txt.src \
                     -train_tgt $DATA_DIR/train.txt.tgt.tagged \
                     -valid_src $DATA_DIR/val.txt.src \
                     -valid_tgt $DATA_DIR/val.txt.tgt.tagged \
                     -save_data $DATA_DIR/$DATASET \
                     -src_seq_length 10000 \
                     -tgt_seq_length 10000 \
                     -src_seq_length_trunc 400 \
                     -tgt_seq_length_trunc 100 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 100000
