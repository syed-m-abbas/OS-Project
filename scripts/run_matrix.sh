#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]]; then
  printf '%s\n' 'Usage: bash scripts/run_matrix.sh /absolute/path/to/ollama/models results/run-name [additional matrix options]' >&2
  exit 2
fi
models_dir="$1"
output_dir="$2"
shift 2
exec .venv/bin/python -m scheduler_lab matrix --models-dir "$models_dir" --output "$output_dir" "$@"
