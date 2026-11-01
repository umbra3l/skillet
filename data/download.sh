#!/usr/bin/env bash
# Download the 2025 StackOverflow Developer Survey files.
# The repo stores them via Git LFS, so we pull from the LFS media host.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="https://media.githubusercontent.com/media/StackExchange/Survey/main/packages/archive/2025"
# Skip files already present (results.csv is ~135 MB; `make all` runs this).
fetch() {
  if [ -s "$HERE/$1" ] && [ "$(wc -c < "$HERE/$1")" -ge "$2" ]; then
    echo "$1 already present, skipping"
  else
    curl -fL --retry 3 -o "$HERE/$1" "$BASE/$1"
  fi
}
fetch schema.csv 1000
fetch results.csv 1000000
echo "schema.csv and results.csv are in $HERE"
