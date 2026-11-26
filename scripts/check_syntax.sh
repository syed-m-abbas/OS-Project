#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for source_file in scripts/*.sh; do
  bash -n "$source_file"
done
printf '%s\n' 'All Bash scripts parsed successfully.'
