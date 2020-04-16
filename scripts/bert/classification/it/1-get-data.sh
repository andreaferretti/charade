#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_DIR="$DIR/../../../.."

cd "$ROOT_DIR"
DATA_DIR="data/classification/sentiment"
URL="http://www.di.unito.it/~tutreeb/sentipolc-evalita16/training_set_sentipolc16.csv.zip"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

wget "$URL"
unzip "training_set_sentipolc16.csv.zip" "training_set_sentipolc16.csv"
rm "training_set_sentipolc16.csv.zip"
mv "training_set_sentipolc16.csv" "train.csv"