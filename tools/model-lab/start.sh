#!/bin/bash
# Start the model lab dashboard on http://127.0.0.1:8765 (local only).
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
exec ../../.venv/bin/uvicorn lab.app:app --host 127.0.0.1 --port "${PORT:-8765}" --log-level warning
