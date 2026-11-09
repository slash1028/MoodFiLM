#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
export NCCL_IB_DISABLE=1 NCCL_NET=Socket PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export HF_HOME="$ROOT/../portable/.cache/huggingface"
export TORCH_HOME="$ROOT/../portable/.cache/torch"
export TRITON_CACHE_DIR="$ROOT/../portable/.cache/triton"
exec "$PYTHON_BIN" "$ROOT/train_stage2.py" --gpu "${GPU:-5}" --run "$@"
