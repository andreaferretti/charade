set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

docker build -f Dockerfile-lock -t charade:prepare .
docker run charade:prepare cat /opt/charade/Pipfile.lock > Pipfile-linux.lock