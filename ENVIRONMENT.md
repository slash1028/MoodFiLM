# Rebuild the training environment on a new GPU image

The full package inventory from the working lab environment is preserved in `portable/environment-freeze.txt`. It records Python-package versions, **not** the OS image, NVIDIA driver, licensed data, base-model files or custom wheel URLs. The source environment used Python 3.10, `torch==2.11.0+cu128`, `torchaudio==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `transformers==5.5.3`, `peft==0.18.1` and `deepspeed==0.18.9`.

For a Linux NVIDIA GPU node, create a fresh Python 3.10 environment. A conda or micromamba environment is convenient; the following uses conda as an example:

```bash
conda create -n moodfilm python=3.10 -y
conda activate moodfilm
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-app.txt
python -m pip check
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py --require-cuda
```

The PyTorch command is the official CUDA 12.8 wheel recipe for version 2.11.0: https://pytorch.org/get-started/previous-versions/ . Choose another officially supported wheel if the new node's driver or GPU requires it, then rerun the checks and record the changed versions. `nvidia-smi` reports the **host driver**; a CUDA runtime in the Python environment does not install or repair that driver. Do not copy the old `/n/work3/huang/...` conda environment as the only deployment method. Do not blindly `pip install -r portable/environment-freeze.txt`: it includes a locally built `flash_attn` wheel and PyTorch local-version strings that require the right index/build. The archived training and inference configs use SDPA; FlashAttention is not needed for their initial smoke test.

`requirements-app.txt` pins the application packages used by the archived text+audio workflows after PyTorch is installed. The full freeze is for comparison if a result changes after migration. Additional optional packages for old video/analysis utilities appear in the full freeze; add them only when using those utilities.

Before training, restore the weights with `python restore_weights.py`, place Qwen2.5-Omni-7B and the dataset/audio caches as described in `HANDOFF.md` and `DATASETS.md`, and run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py --require-cuda --require-model
python portable/run.py --dataset iemocap --gpus 0 --check-imports
python portable/run.py --dataset meld --gpus 0 --check-imports
```

The final two commands are preflight only. The wrapper scripts accept `PYTHON_BIN=$(command -v python)` to force this environment. On a new node, first smoke-test a short real batch and confirm the audio-cache IDs and device mapping. An import check alone does not prove that NCCL, DeepSpeed, the audio path or the checkpoint load works. If a one-GPU NCCL init fails on an InfiniBand path, test `NCCL_IB_DISABLE=1 NCCL_NET=Socket` as was needed on the source cluster.
