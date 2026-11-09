#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$ROOT/.cache/huggingface" TORCH_HOME="$ROOT/.cache/torch"
# Node-local, private compiler caches; not NFS. Never delete other users' caches.
TASK_CACHE=$(mktemp -d /tmp/moodfilm-erc.XXXXXXXX)
export TRITON_CACHE_DIR="$TASK_CACHE/triton" TORCH_EXTENSIONS_DIR="$TASK_CACHE/torch_extensions"
cd "$ROOT"
exec "$PYTHON_BIN" meld_dialogue_sweep10.py --gpu "${GPU:-0}" --epochs "${EPOCHS:-5}" "$@"
