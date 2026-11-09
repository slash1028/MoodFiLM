#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[STOP] Python environment missing: $PYTHON_BIN"
  echo "Wait for the environment copy, or explicitly set PYTHON_BIN to an authorized working environment."
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$ROOT/.cache/huggingface" TORCH_HOME="$ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions" TRITON_CACHE_DIR="$ROOT/.cache/triton"
cd "$ROOT"
exec "$PYTHON_BIN" dialogue_personality/infer.py "$@"
