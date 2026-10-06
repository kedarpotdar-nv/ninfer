#!/usr/bin/env bash
# Serve the RTX 5090 recipe: usage: serve.sh <artifact.ninfer> [port] [extra ninfer-serve args...]
# NINFER_PDL=0 in the environment disables programmatic dependent launch (A/B control).
set -euo pipefail
artifact=${1:?artifact path}
port=${2:-9932}
shift $(( $# >= 2 ? 2 : $# ))
repo=$(cd "$(dirname "$0")/../.." && pwd)
exec "$repo/build/apps/ninfer-serve" "$artifact" \
  --host 127.0.0.1 --port "$port" --model-id qwen3.8-27b \
  --max-context 65536 --kv-capacity 65536 --max-concurrency 1 \
  --kv-dtype bf16 --prefill-chunk 2048 \
  --device-state-slots 4 --host-context-mib 8192 --preserve-thinking \
  --agent-prompt-cache \
  --spec dflash2 --draft-tokens 7 --lm-head-draft "$@"
