# FOR THE ITALIAN LANGUAGE
# 1. download model from "https://drive.google.com/uc?export=download&id=1x1pRE7LZilIcPSWgoNpGyci9NwPqZYiL"
# 2. put model into folder "models/bert" with name "bert_uncased_L-12_H-768_A-12_italian_alb3rt0"
# TODO: working scripts

#! /bin/bash
#set -e
#set -u

#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
#ROOT_DIR="$DIR/../.."

#cd "$ROOT_DIR"
#MODEL_DIR="models/bert"
#MODEL_NAME="bert_uncased_L-12_H-768_A-12_italian_alb3rt0"
#URL="https://drive.google.com/uc?export=download&id=1x1pRE7LZilIcPSWgoNpGyci9NwPqZYiL"

#cd "$MODEL_DIR"

#if [[ ! -d "$MODEL_NAME" ]]; then
#  wget "$URL"
#  mv "alberto_uncased_L-12_H-768_A-12_italian_huggingface.co" "$MODEL_NAME"
#fi