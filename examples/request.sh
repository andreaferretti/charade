#! /bin/bash
set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR"

if [ $# -eq 0 ]; then
  JSON_REQUEST="request.json"
else
  JSON_REQUEST="request_$1.json"
fi

curl -XPOST -H 'Content-Type: application/json' -d @"$JSON_REQUEST" http://localhost:9000/