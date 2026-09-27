#!/bin/bash
# Run every model over the test set, in order. Re-runnable: each stage
# skips work it has already done where it can.
#   ./run_pipeline.sh <ssh-host> "<remote .photoslibrary path>"
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
PY=../../.venv/bin/python
$PY -m lab.sample
$PY -m lab.fetch --host "$1" --library "$2"
$PY -m lab.prepare
$PY -m lab.people
$PY -m lab.vlm qwen3-vl:30b-a3b-instruct-q4_K_M
$PY -m lab.vlm qwen3-vl:8b-instruct-q4_K_M
$PY -m lab.siglip
$PY -m lab.whisper_run
.venv-qalign/bin/python lab/qalign.py
$PY -m lab.index --queries
ollama ps
