#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch


def _load_adapter(path: Path):
    trainable_safetensors = path / "trainable_model.safetensors"
    trainable_bin = path / "trainable_model.bin"
    if trainable_safetensors.exists():
        from safetensors.torch import load_file

        return load_file(str(trainable_safetensors)), "trainable_safetensors"
    if trainable_bin.exists():
        return torch.load(trainable_bin, map_location="cpu"), "trainable_bin"

    safetensors_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path)), "safetensors"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu"), "bin"
    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {path}")


def _save_adapter(state_dict, output_dir: Path, fmt: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt in ("trainable_safetensors", "safetensors"):
        from safetensors.torch import save_file

        filename = "trainable_model.safetensors" if fmt == "trainable_safetensors" else "adapter_model.safetensors"
        save_file(state_dict, str(output_dir / filename))
    else:
        filename = "trainable_model.bin" if fmt == "trainable_bin" else "adapter_model.bin"
        torch.save(state_dict, output_dir / filename)


def main():
    parser = argparse.ArgumentParser(description="Average lightweight trainable or PEFT LoRA adapter weights.")
    parser.add_argument("--adapter_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    adapter_dirs = [Path(p).resolve() for p in args.adapter_dirs]
    output_dir = Path(args.output_dir).resolve()
    if len(adapter_dirs) < 2:
        raise ValueError("Need at least two adapter directories to average.")

    states = []
    fmt = None
    for adapter_dir in adapter_dirs:
        state, this_fmt = _load_adapter(adapter_dir)
        states.append(state)
        fmt = fmt or this_fmt

    keys = list(states[0].keys())
    for state in states[1:]:
        if set(state.keys()) != set(keys):
            missing = sorted(set(keys) - set(state.keys()))[:10]
            extra = sorted(set(state.keys()) - set(keys))[:10]
            raise ValueError(f"Adapter keys differ. missing={missing}, extra={extra}")

    averaged = {}
    for key in keys:
        tensors = [state[key].detach().float() for state in states]
        averaged_tensor = torch.stack(tensors, dim=0).mean(dim=0)
        averaged[key] = averaged_tensor.to(dtype=states[0][key].dtype)

    _save_adapter(averaged, output_dir, fmt)

    for filename in (
        "adapter_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "config.json",
        "model_params.json",
    ):
        src = adapter_dirs[0] / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)

    summary = {
        "adapter_dirs": [str(p) for p in adapter_dirs],
        "num_adapters": len(adapter_dirs),
        "num_tensors": len(averaged),
    }
    with open(output_dir / "average_adapter_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
