"""Report runtime versions and optionally require usable CUDA/model files."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys

PACKAGES = (
    "torch", "torchaudio", "torchvision", "transformers", "peft",
    "deepspeed", "accelerate", "qwen-omni-utils", "safetensors",
    "numpy", "pandas", "scikit-learn", "librosa", "soundfile",
)
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-model", action="store_true")
    args = parser.parse_args()
    failures = []

    print("Python:", sys.version.split()[0], sys.executable)
    if sys.version_info[:2] != (3, 10):
        print("WARNING: source environment used Python 3.10")
    for name in PACKAGES:
        try:
            print(f"{name}: {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            failures.append(f"Missing package: {name}")

    if "Missing package: torch" not in failures:
        import torch
        print("PyTorch CUDA runtime:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"Visible GPU {index}: {torch.cuda.get_device_name(index)}")
            probe = torch.ones(1, device="cuda:0")
            torch.cuda.synchronize()
            print("CUDA allocation probe:", float(probe.item()))
        elif args.require_cuda:
            failures.append("CUDA is unavailable")

    try:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        print("Qwen2.5-Omni model class:", Qwen2_5OmniThinkerForConditionalGeneration.__name__)
    except Exception as exc:
        failures.append(f"Qwen2.5-Omni import failed: {exc}")

    model = ROOT / "portable/LLM_bases/Qwen2.5-Omni-7B"
    index_file = model / "model.safetensors.index.json"
    if index_file.is_file():
        shards = set(json.loads(index_file.read_text())["weight_map"].values())
        missing = sorted(name for name in shards if not (model / name).is_file())
        print(f"Base-model shards: {len(shards)}, missing: {len(missing)}")
        if missing:
            failures.append(f"Missing model shards: {missing[:5]}")
    elif args.require_model:
        failures.append(f"Missing model index: {index_file}")
    else:
        print("Base model not present; add --require-model to fail on this")

    for failure in failures:
        print("ERROR:", failure)
    if failures:
        raise SystemExit(1)
    print("Environment check passed. Run a real-batch smoke before full training.")


if __name__ == "__main__":
    main()
