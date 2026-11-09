#!/usr/bin/env python
"""Evaluate GT-prompt MoodFiLM with matched prompt and internal personality noise."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

import torch

from run_moodfilm_visual_analysis import (
    ANALYZER,
    PROJECT_ROOT,
    build_eval_command,
    checkpoint_args,
    make_deepspeed_config,
)

TRAITS = ("O", "C", "E", "A", "N")


def profile(values):
    if isinstance(values, str):
        values = ast.literal_eval(values)
    if isinstance(values, dict):
        values = [values[key] for key in TRAITS]
    if not isinstance(values, (tuple, list)) or len(values) != len(TRAITS):
        raise ValueError(f"Expected a five-factor profile, got {values!r}")
    return torch.tensor([(float(value) - 4.0) / 3.0 for value in values])


def transformed(vector, sigma, seed, mode, noise_key=""):
    key = ",".join(f"{float(item):.8f}" for item in vector.tolist())
    digest = hashlib.sha256(f"{int(seed)}|{noise_key}|{key}".encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
    if mode == "random":
        digest = hashlib.sha256(
            f"random|{int(seed)}|{noise_key}|"
            f"{tuple(round(float(x), 7) for x in vector.tolist())}".encode()
        ).digest()
        generator.manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
        return torch.rand(5, generator=generator) * 2.0 - 1.0
    return (vector + torch.randn(vector.shape, generator=generator) * (sigma / 3.0)).clamp(-1.0, 1.0)


def rewrite_prompt(text, raw):
    ocean = "<" + ", ".join(f"{key}:{raw[key]:.3f}" for key in TRAITS) + ">"
    names = {"O": "openness", "C": "conscientiousness", "E": "extraversion", "A": "agreeableness", "N": "neuroticism"}
    tendencies = []
    for key in TRAITS:
        level = "low" if raw[key] < 3.0 else "high" if raw[key] > 5.0 else "medium"
        tendencies.append(f"{level} {names[key]}")
    pattern = r"Target speaker personality profile \(OCEAN, 1\.0-7\.0 where higher means stronger trait\) is <[^>]+>\. Interpreted tendencies: [^.]+\. Use this profile to disambiguate subtle emotions in tone and wording\."
    replacement = (
        "Target speaker personality profile (OCEAN, 1.0-7.0 where higher means stronger trait) is "
        f"{ocean}. Interpreted tendencies: {', '.join(tendencies)}. "
        "Use this profile to disambiguate subtle emotions in tone and wording."
    )
    updated, count = re.subn(pattern, replacement, text)
    if count != 1:
        raise RuntimeError(f"Expected one personality prompt clause, found {count}.")
    return updated


def raw_like(original, raw):
    """Preserve the dataset's list/dict representation after perturbation."""
    if isinstance(original, dict):
        return {key: raw[key] for key in TRAITS}
    return [raw[key] for key in TRAITS]


def perturb_profile(value, sigma, seed, mode, noise_key=""):
    normalized = profile(value)
    noisy = transformed(normalized, sigma, seed, mode, noise_key)
    raw = {key: round(float(noisy[index] * 3.0 + 4.0), 3) for index, key in enumerate(TRAITS)}
    return raw_like(value, raw), raw


def dialogue_key(row):
    """Return a stable conversation key for MELD/IEMOCAP-style records."""
    for field in ("dialogue_id", "dialog_id", "conversation_id"):
        if row.get(field) is not None:
            return str(row[field])
    sample_id = str(row.get("sample_id") or row.get("utterance_id") or "")
    meld_match = re.match(r"(.+_dia\d+)_utt\d+$", sample_id)
    if meld_match:
        return meld_match.group(1)
    # IEMOCAP utterance IDs normally end in an utterance counter.
    iemocap_match = re.match(r"(.+)_\d+$", sample_id)
    return iemocap_match.group(1) if iemocap_match else sample_id


def scoped_noise_key(row, speaker, noise_scope):
    if noise_scope == "profile":
        return ""
    return f"{dialogue_key(row)}|{speaker or 'unknown-speaker'}"


def make_noisy_data(source, destination, sigma, seed, mode, noise_scope="profile"):
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    perturbed = 0
    unavailable = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            value = row.get("gt_personality")
            target_speaker = row.get("speaker_name") or row.get("speaker_id")
            target_noise_key = scoped_noise_key(row, target_speaker, noise_scope)
            if value is None:
                unavailable += 1
            elif mode != "gaussian" or sigma > 0.0:
                target_noisy, raw = perturb_profile(
                    value, sigma, seed, mode, target_noise_key
                )
                # The prompt and every structured profile consumed by the mood
                # branch must receive the same intervention.
                row["gt_personality"] = target_noisy
                if row.get("personality") is not None:
                    row["personality"] = target_noisy
                row["input"] = rewrite_prompt(row["input"], raw)
                perturbed += 1
            # Context profiles remain available even when the target is unknown.
            # Transform each known profile with the same deterministic mapping
            # used for target profiles, preserving missing values and ordering.
            if mode != "gaussian" or sigma > 0.0:
                context_value = row.get("interlocutor_personality")
                if context_value is not None:
                    context_speakers = row.get("speaker_context_speakers") or []
                    interlocutor = context_speakers[-1] if len(set(context_speakers)) == 1 else "interlocutor"
                    row["interlocutor_personality"], _ = perturb_profile(
                        context_value,
                        sigma,
                        seed,
                        mode,
                        scoped_noise_key(row, interlocutor, noise_scope),
                    )
                if row.get("speaker_context_personalities") is not None:
                    context_speakers = row.get("speaker_context_speakers") or []
                    row["speaker_context_personalities"] = [
                        perturb_profile(
                            value,
                            sigma,
                            seed,
                            mode,
                            scoped_noise_key(
                                row,
                                context_speakers[index]
                                if index < len(context_speakers)
                                else f"context-{index}",
                                noise_scope,
                            ),
                        )[0]
                        if value is not None else None
                        for index, value in enumerate(row["speaker_context_personalities"])
                    ]
            rows.append(row)
    with (destination / "test.json").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[personality-noise] mode={mode} scope={noise_scope} sigma={sigma:g} "
        f"perturbed={perturbed} unavailable={unavailable}",
        flush=True,
    )


def replace_value(command, flag, value):
    index = command.index(flag)
    command[index + 1] = str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dataset", choices=("iemocap", "meld"), default="iemocap")
    parser.add_argument("--port-base", type=int, default=64100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--noise-scope",
        choices=("profile", "dialogue"),
        default="profile",
        help="Reuse noise per original profile, or resample it for each dialogue/speaker.",
    )
    parser.add_argument("--sigmas", default="0,0.4,0.8,1.2,1.6,2.4")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "code/experiments/Qwen2.5-Omni-7B/lora/iemocap/runs/audio-text/88_smf_prompt_strength8_0831/iemo_gtp_s125/adapter_epoch4")
    parser.add_argument("--prompt-data", type=Path, default=PROJECT_ROOT / "code/experiments/Qwen2.5-Omni-7B/lora/iemocap/runs/audio-text/88_smf_prompt_strength8_0831/data_gtprompt_w12/stage_window_personality_continuous")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "code/experiments/gtprompt_noise8_0901")
    parser.add_argument("--deepspeed", type=Path, default=Path("/n/work3/huang/tools/miniconda3/envs/InstructERC_cu12/bin/deepspeed"))
    args = parser.parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("Provide at least one GPU ID.")
    source = args.prompt_data / "test.json"
    if not source.is_file() or not args.checkpoint.is_dir():
        raise FileNotFoundError("Missing prompt test data or checkpoint.")
    saved = checkpoint_args(args.checkpoint)
    ds_config = make_deepspeed_config(saved, args.output_root, len(gpus))
    specs = [(float(value), "gaussian") for value in args.sigmas.split(",")] + [(0.0, "random")]
    noise_args = []
    for index, (sigma, mode) in enumerate(specs):
        name = "random" if mode == "random" else f"sigma_{sigma:g}".replace(".", "p")
        data_dir = args.output_root / "data" / name
        output_dir = args.output_root / "eval" / name
        prediction = output_dir / "preds_for_eval.text"
        if not prediction.is_file():
            make_noisy_data(
                source, data_dir, sigma, args.seed, mode, args.noise_scope
            )
            # Profiles and prompt were already perturbed together in the data.
            # Disable runtime corruption to avoid applying the intervention twice.
            command = build_eval_command(args.deepspeed, ",".join(gpus), args.port_base + index, args.checkpoint, output_dir, 0.0, args.seed, "raw", "gaussian", len(gpus), ds_config)
            replace_value(command, "--data_dir", data_dir)
            replace_value(command, "--max_new_tokens", saved.get("max_new_tokens", 64))
            print("=====", name, "=====", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        noise_args.extend(["--noise-eval", f"{sigma:g}={prediction}"] if mode == "gaussian" else ["--personality-control", f"random={prediction}"])
    analysis = ["/n/work3/huang/tools/miniconda3/envs/InstructERC_cu12/bin/python", str(ANALYZER), "--dataset", args.dataset, "--output-dir", str(args.output_root / "analysis"), "--final-checkpoint", str(args.checkpoint), "--noise-scale", "raw", *noise_args]
    subprocess.run(analysis, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
