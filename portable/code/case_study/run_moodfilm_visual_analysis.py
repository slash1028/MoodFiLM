#!/usr/bin/env python
"""Run evaluation-only MoodFiLM diagnostics on a selected GPU set."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN = PROJECT_ROOT / "code" / "main_new.py"
ANALYZER = PROJECT_ROOT / "code" / "case_study" / "analyze_moodfilm_diagnostics.py"

CHECKPOINTS = {
    "iemocap": {
        "final": "code/experiments/Qwen2.5-Omni-7B/lora/iemocap/runs/audio-text/84_iemo_ffsearch2_0828/stage2_03_p_gdom/adapter_epoch5",
        "dyn_full": "code/experiments/Qwen2.5-Omni-7B/lora/iemocap/runs/audio-text/75_iemo_smf8_0823/stage2_01_pfull/adapter_epoch5",
        "dyn_no": "code/experiments/Qwen2.5-Omni-7B/lora/iemocap/runs/audio-text/75_iemo_smf8_0823/stage2_07_nod/adapter_epoch5",
    },
    "meld": {
        "final": "code/experiments/Qwen2.5-Omni-7B/lora/meld/runs/audio-text/85_meld_ff5_0829/stage2_44_l08/adapter_epoch4",
        "dyn_full": "code/experiments/Qwen2.5-Omni-7B/lora/meld/runs/audio-text/75_meld_smf8_0823/stage2_12_l10/adapter_epoch4",
        "dyn_no": "code/experiments/Qwen2.5-Omni-7B/lora/meld/runs/audio-text/75_meld_smf8_0823/stage2_17_nod/adapter_epoch4",
    },
}

NOISE_SIGMAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CONTROL_ARGS = {
    "output_dir",
    "checkpoint_dir",
    "deepspeed_config",
    "do_train",
    "do_eval",
    "batch_size",
    "gradient_accumulation_steps",
    "eval_batch_size",
    "max_new_tokens",
    "num_train_epochs",
    "eval_every_epochs",
    "save_steps",
    "save_checkpoint_epochs",
    "save_best_checkpoint",
    "disable_deepspeed_checkpoint_save",
    "save_lightweight_adapter_epochs",
    "save_speaker_mood_analysis",
    "speaker_mood_personality_noise_std",
    "speaker_mood_personality_noise_seed",
    "speaker_mood_personality_noise_scale",
    "speaker_mood_personality_corruption",
    "speaker_mood_ordered_num_workers",
    "gradient_checkpointing",
    "offload_optimizer",
    "local_rank",
}


def accepted_options() -> set[str]:
    source = MAIN.read_text(encoding="utf-8")
    return set(re.findall(r"[\"']--([A-Za-z0-9_]+)", source))


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def checkpoint_args(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "model_args.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing lightweight checkpoint metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def make_deepspeed_config(
    model_args: dict[str, Any], output_root: Path, world_size: int
) -> Path:
    source = Path(str(model_args["deepspeed_config"]))
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    config = json.loads(source.read_text(encoding="utf-8"))
    config["train_batch_size"] = int(world_size)
    config["train_micro_batch_size_per_gpu"] = 1
    config["gradient_accumulation_steps"] = 1
    path = output_root / f"deepspeed_eval{world_size}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def job_output(output_root: Path, dataset: str, name: str) -> Path:
    return output_root / "eval" / dataset / name


def configured_checkpoints(dataset: str, args: argparse.Namespace) -> dict[str, Path]:
    checkpoints = {
        key: (PROJECT_ROOT / value).resolve()
        for key, value in CHECKPOINTS[dataset].items()
    }
    override = getattr(args, f"final_checkpoint_{dataset}", None)
    if override is not None:
        checkpoints["final"] = override.resolve()
    return checkpoints


def build_eval_command(
    deepspeed: Path,
    gpus: str,
    port: int,
    checkpoint: Path,
    output_dir: Path,
    noise_std: float,
    noise_seed: int,
    noise_scale: str,
    corruption: str,
    world_size: int,
    ds_config: Path,
) -> list[str]:
    saved = checkpoint_args(checkpoint)
    accepted = accepted_options()
    command = [
        str(deepspeed),
        "--include",
        f"localhost:{gpus}",
        "--master_port",
        str(port),
        str(MAIN),
    ]
    for key, value in saved.items():
        if key in CONTROL_ARGS or key not in accepted or value is None:
            continue
        if isinstance(value, (dict, list)):
            continue
        command.extend([f"--{key}", cli_value(value)])
    command.extend(
        [
            "--output_dir",
            str(output_dir),
            "--run_id",
            f"moodviz_{output_dir.parent.name}_{output_dir.name}",
            "--checkpoint_dir",
            str(checkpoint),
            "--deepspeed_config",
            str(ds_config),
            "--batch_size",
            str(world_size),
            "--gradient_accumulation_steps",
            "1",
            "--eval_batch_size",
            "1",
            "--max_new_tokens",
            "8",
            "--do_train",
            "False",
            "--do_eval",
            "True",
            "--save_speaker_mood_analysis",
            "True",
            "--speaker_mood_personality_noise_std",
            str(noise_std),
            "--speaker_mood_personality_noise_seed",
            str(noise_seed),
            "--speaker_mood_personality_noise_scale",
            str(noise_scale),
            "--speaker_mood_personality_corruption",
            str(corruption),
            "--speaker_mood_ordered_num_workers",
            "1",
            "--save_checkpoint_epochs",
            "none",
            "--save_best_checkpoint",
            "False",
            "--disable_deepspeed_checkpoint_save",
            "True",
            "--save_lightweight_adapter_epochs",
            "none",
        ]
    )
    return command


def run_job(
    dataset: str,
    name: str,
    checkpoint: Path,
    noise_std: float,
    noise_seed: int,
    corruption: str,
    args: argparse.Namespace,
    job_index: int,
) -> Path:
    output_dir = job_output(args.output_root, dataset, name)
    prediction_file = output_dir / "preds_for_eval.text"
    if prediction_file.is_file() and prediction_file.stat().st_size > 0:
        print(f"[SKIP] {dataset}/{name}: {prediction_file} already exists", flush=True)
        return prediction_file
    if output_dir.exists() and any(output_dir.iterdir()):
        partial = output_dir.with_name(
            f"{output_dir.name}.partial_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir.rename(partial)
        print(f"[MOVE] incomplete output -> {partial}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = checkpoint_args(checkpoint)
    ds_config = make_deepspeed_config(saved, args.output_root, args.world_size)
    command = build_eval_command(
        deepspeed=args.deepspeed,
        gpus=args.gpus,
        port=args.port_base + job_index,
        checkpoint=checkpoint,
        output_dir=output_dir,
        noise_std=noise_std,
        noise_seed=noise_seed,
        noise_scale=args.noise_scale,
        corruption=corruption,
        world_size=args.world_size,
        ds_config=ds_config,
    )
    manifest = {
        "dataset": dataset,
        "job": name,
        "checkpoint": str(checkpoint),
        "noise_std": noise_std,
        "noise_seed": noise_seed,
        "noise_scale": args.noise_scale,
        "personality_corruption": corruption,
        "gpus": args.gpus,
        "command": command,
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n===== RUN {dataset}/{name} =====", flush=True)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return prediction_file
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if not prediction_file.is_file():
        raise RuntimeError(f"Evaluation finished without {prediction_file}")
    return prediction_file


def run_analysis(
    dataset: str,
    noise_files: dict[float, list[Path]],
    control_files: dict[str, Path],
    dyn_full_file: Path | None,
    dyn_no_file: Path | None,
    args: argparse.Namespace,
) -> None:
    checkpoints = configured_checkpoints(dataset, args)
    command = [
        str(args.deepspeed.resolve().parent / "python"),
        str(ANALYZER),
        "--dataset",
        dataset,
        "--split",
        "test",
        "--output-dir",
        str(args.output_root / "analysis" / dataset),
        "--max-lag",
        str(args.max_lag),
        "--noise-seed",
        str(args.noise_seeds[0]),
        "--noise-scale",
        args.noise_scale,
        "--final-checkpoint",
        str(checkpoints["final"]),
    ]
    if dyn_full_file is not None and dyn_no_file is not None:
        command.extend(
            [
                "--dyn-full-eval",
                str(dyn_full_file),
                "--dyn-no-eval",
                str(dyn_no_file),
                "--dyn-full-checkpoint",
                str(checkpoints["dyn_full"]),
                "--dyn-no-checkpoint",
                str(checkpoints["dyn_no"]),
            ]
        )
    for sigma, paths in sorted(noise_files.items()):
        for path in paths:
            command.extend(["--noise-eval", f"{sigma:g}={path}"])
    for control, path in sorted(control_files.items()):
        command.extend(["--personality-control", f"{control}={path}"])
    print(f"\n===== ANALYZE {dataset} =====", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,4,6")
    parser.add_argument("--datasets", choices=["all", "iemocap", "meld"], default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "code" / "experiments" / "moodfilm_visual_0830",
    )
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument(
        "--noise-seeds",
        help="Comma-separated perturbation seeds. Sigma 0 is evaluated once.",
    )
    parser.add_argument(
        "--noise-sigmas",
        default=",".join(str(value) for value in NOISE_SIGMAS),
        help="Comma-separated Gaussian noise standard deviations.",
    )
    parser.add_argument(
        "--noise-scale", choices=["raw", "normalized"], default="raw"
    )
    parser.add_argument("--noise-only", action="store_true")
    parser.add_argument(
        "--personality-controls",
        default="",
        help="Comma-separated additional controls: random.",
    )
    parser.add_argument("--final-checkpoint-iemocap", type=Path)
    parser.add_argument("--final-checkpoint-meld", type=Path)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--port-base", type=int, default=61200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--deepspeed",
        type=Path,
        default=Path(
            "/n/work3/huang/tools/miniconda3/envs/InstructERC_cu12/bin/deepspeed"
        ),
    )
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("Provide at least one GPU ID.")
    args.gpus = ",".join(gpu_ids)
    args.world_size = len(gpu_ids)
    seed_values = args.noise_seeds or str(args.noise_seed)
    args.noise_seeds = [
        int(value.strip()) for value in seed_values.split(",") if value.strip()
    ]
    if not args.noise_seeds:
        raise ValueError("Provide at least one personality noise seed.")
    args.noise_sigmas = [
        float(value.strip())
        for value in args.noise_sigmas.split(",")
        if value.strip()
    ]
    if not args.noise_sigmas or 0.0 not in args.noise_sigmas:
        raise ValueError("--noise-sigmas must include 0.")
    if any(value < 0.0 for value in args.noise_sigmas):
        raise ValueError("Personality noise sigmas must be non-negative.")
    if len(set(args.noise_sigmas)) != len(args.noise_sigmas):
        raise ValueError("Personality noise sigmas must be unique.")
    args.personality_controls = [
        value.strip().lower()
        for value in args.personality_controls.split(",")
        if value.strip()
    ]
    invalid_controls = set(args.personality_controls) - {"random"}
    if invalid_controls:
        raise ValueError(f"Unsupported personality controls: {sorted(invalid_controls)}")
    if not args.deepspeed.is_file():
        raise FileNotFoundError(f"DeepSpeed launcher not found: {args.deepspeed}")
    return args


def main() -> None:
    args = parse_args()
    datasets = ("iemocap", "meld") if args.datasets == "all" else (args.datasets,)
    job_index = 0
    for dataset in datasets:
        checkpoints = configured_checkpoints(dataset, args)
        required = ("final",) if args.noise_only else ("final", "dyn_full", "dyn_no")
        for key in required:
            checkpoint = checkpoints[key]
            if not checkpoint.is_dir():
                raise FileNotFoundError(checkpoint)

        noise_files: dict[float, list[Path]] = {}
        for sigma in args.noise_sigmas:
            seeds = args.noise_seeds[:1] if sigma == 0.0 else args.noise_seeds
            noise_files[sigma] = []
            for noise_seed in seeds:
                tag = f"final_s{sigma:g}".replace(".", "p")
                if len(seeds) > 1:
                    tag += f"_n{noise_seed}"
                noise_files[sigma].append(
                    run_job(
                        dataset,
                        tag,
                        checkpoints["final"],
                        sigma,
                        noise_seed,
                        "gaussian",
                        args,
                        job_index,
                    )
                )
                job_index += 1
        control_files: dict[str, Path] = {}
        for control in args.personality_controls:
            control_files[control] = run_job(
                dataset,
                f"control_{control}",
                checkpoints["final"],
                0.0,
                args.noise_seeds[0],
                control,
                args,
                job_index,
            )
            job_index += 1
        dyn_full_file = None
        dyn_no_file = None
        if not args.noise_only:
            dyn_full_file = run_job(
                dataset,
                "dyn_full",
                checkpoints["dyn_full"],
                0.0,
                args.noise_seeds[0],
                "gaussian",
                args,
                job_index,
            )
            job_index += 1
            dyn_no_file = run_job(
                dataset,
                "dyn_no",
                checkpoints["dyn_no"],
                0.0,
                args.noise_seeds[0],
                "gaussian",
                args,
                job_index,
            )
            job_index += 1
        if not args.dry_run:
            run_analysis(
                dataset,
                noise_files,
                control_files,
                dyn_full_file,
                dyn_no_file,
                args,
            )

    print(f"\nAll diagnostics finished: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
