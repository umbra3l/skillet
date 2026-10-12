#!/usr/bin/env bash
# Download the 2025 StackOverflow Developer Survey files.
# The repo stores them via Git LFS, so we pull from the LFS media host.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="https://media.githubusercontent.com/media/StackExchange/Survey/main/packages/archive/2025"
curl -fL --retry 3 -o "$HERE/schema.csv"  "$BASE/schema.csv"
curl -fL --retry 3 -o "$HERE/results.csv" "$BASE/results.csv"
echo "Downloaded schema.csv and results.csv to $HERE"
