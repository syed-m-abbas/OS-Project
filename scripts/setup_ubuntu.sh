#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update
sudo apt-get install -y python3 python3-venv util-linux stress-ng
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[analysis]'
printf '%s\n' 'Environment ready. Install Ollama from https://ollama.com/download/linux if needed.'
printf '%s\n' 'Then follow README.md to pull the exact model and calibrate the prompt.'
