#!/usr/bin/env bash
set -euo pipefail

remote="${1:-Pro6000-Remote}"
dst="${2:-~/qw3/qw3-v1}"

rsync -az --delete \
  --exclude 'ds4-main' \
  --exclude '.git' \
  --exclude '.DS_Store' \
  --exclude 'build*' \
  ./ "${remote}:${dst}/"
