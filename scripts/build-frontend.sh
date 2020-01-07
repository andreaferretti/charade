set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

rm -rf ./app
cd ui
npm run build
cd ..
cp -r ui/build app