#!/usr/bin/env python3
"""Auditable cleanup for legacy experiment states and non-best adapters.

Every apply action writes a JSONL manifest before removing anything.  The
script deliberately preserves metrics, configs, root-level adapters, and any
lightweight files nested beside old DeepSpeed states.
"""

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


ADAPTER_RE = re.compile(r"adapter_epoch(\d+)$")
PRED_RE = re.compile(r"preds_for_eval_(\d+)\.text$")
SELECTION_RE = re.compile(
    r'"selection_metric"\s*:\s*\{.*?"value"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',
    re.DOTALL,
)

# Experiment families that remain loadable for the current paper, final
# ablations, personality chain, and the September multimodal investigations.
KEEP_RUN_FAMILIES = {
    "0.765-personality",
    "89_smf_prompt_upper6_0902",
    "90_smf_components6_0902",
    "94_iemo_s150_journal27_0903",
    "97_iemo_s150_predctx6_0903",
    "98_iemo_personality_utt8_0903",
    "99_iemo_personality_utt15_0904",
    "100_meld_peld_b050abl8_0904",
    "101_meld_smf_upperabl8_0904",
    "102_meld_smf_recover8_0905",
    "103_meld_full_simplefilm10_6gpu_0909",
    "103_meld_interaction12_0906",
    "104_meld_full_simplefilm10_4gpu_0914",
    "104_meld_intergate25_0906",
    "105_meld_nearest26_0907",
    "107_iemo_at_aout100_scopefix_8gpu_0918",
    "109_iemo_at_turnalign_pair20_6gpu_0919",
    "111_iemo_pairclass_mel_2x96gb_5ep_0920",
    "111_iemo_pairclass_token_6x49gb_5ep_0920",
    "112_iemo_utterance_prefix_2x96_0923",
    "112_meld_peld_lab20_2gpu_0921",
    "113_iemo_audio_token10_2x96_0924",
    "113_meld_audiofix_2gpu_0922",
}

KEEP_RUNNERS = {
    "run_iemocap_at_pairwise20_6gpu_0919.sh",
    "run_iemocap_audio_token10_2x96_0924.sh",
    "run_iemocap_pairclass_token_mel_6gpu_5ep_0920.sh",
    "run_iemocap_pairwise_token_mel_6gpu_5ep_0919.sh",
    "run_iemocap_personality_at_fixed_8gpu_0918.sh",
    "run_iemocap_s150_journal27_6gpu_0903.sh",
    "run_iemocap_s150_predctx_6gpu_0903.sh",
    "run_iemocap_s150_predicted_matched_8gpu_0903.sh",
    "run_iemocap_utterance_prefix_2x96_0923.sh",
    "run_meld_dialogue_personality_predict_8gpu_0917.sh",
    "run_meld_dialogue_personality_strict8_0918.sh",
    "run_meld_dialoguepers_full_8gpu_0918.sh",
    "run_meld_full_simplefilm10_4gpu_0914.sh",
    "run_meld_full_simplefilm10_6gpu_0909.sh",
    "run_meld_interaction12_8gpu_0906.sh",
    "run_meld_interaction_gated25_8gpu_0906.sh",
    "run_meld_nearest_module26_8gpu_0907.sh",
    "run_meld_peld_b050_ablation_8gpu_0904.sh",
    "run_meld_smf_recover8_0905.sh",
    "run_meld_smf_upperabl8_0904.sh",
    "run_smf_journal_components_6gpu_0902.sh",
    "run_smf_prompt_upper6_0902.sh",
    "run_stage1_personality_utt_protocol_8gpu_0904.sh",
    "run_stageB_reasoning_from_stage2_label.sh",
}


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def tree_size(path: Path) -> int:
    return sum(file_size(item) for item in path.rglob("*") if item.is_file())


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def deepspeed_records(root: Path) -> list[dict]:
    model_states = sorted(root.rglob("mp_rank_*_model_states.pt"))
    checkpoint_dirs = sorted({path.parent for path in model_states})
    targets = set(model_states)
    for directory in checkpoint_dirs:
        targets.update(directory.glob("*_optim_states.pt"))
    records = []
    for path in sorted(targets):
        records.append(
            {
                "kind": "deepspeed_training_state",
                "path": str(path.resolve()),
                "bytes": file_size(path),
                "checkpoint_dir": str(path.parent.resolve()),
            }
        )
    return records


def scan_adapter_parents(root: Path) -> list[Path]:
    parents = set()
    for current, dirs, _ in os.walk(root):
        adapter_dirs = [name for name in dirs if ADAPTER_RE.fullmatch(name)]
        if adapter_dirs:
            parents.add(Path(current))
            dirs[:] = [name for name in dirs if name not in adapter_dirs]
    return sorted(parents)


def selection_value(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            prefix = handle.read(4 * 1024 * 1024)
    except OSError:
        return None
    match = SELECTION_RE.search(prefix)
    return float(match.group(1)) if match else None


def adapter_records(root: Path):
    records = []
    skipped = []
    for parent in scan_adapter_parents(root):
        adapters = {}
        for path in parent.glob("adapter_epoch*"):
            match = ADAPTER_RE.fullmatch(path.name)
            if match and path.is_dir():
                adapters[int(match.group(1))] = path
        if len(adapters) < 2:
            continue
        predictions = {}
        for path in parent.glob("preds_for_eval_*.text"):
            match = PRED_RE.fullmatch(path.name)
            if not match:
                continue
            value = selection_value(path)
            if value is not None:
                predictions[int(match.group(1)) + 1] = value
        if set(adapters) != set(predictions):
            skipped.append(
                {
                    "setting": str(parent.resolve()),
                    "adapter_epochs": sorted(adapters),
                    "metric_epochs": sorted(predictions),
                    "reason": "adapter/metric epoch sets differ",
                }
            )
            continue
        best_epoch = max(predictions, key=lambda epoch: (predictions[epoch], -epoch))
        for epoch, path in sorted(adapters.items()):
            if epoch == best_epoch:
                continue
            records.append(
                {
                    "kind": "non_best_adapter_epoch",
                    "path": str(path.resolve()),
                    "bytes": tree_size(path),
                    "setting": str(parent.resolve()),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_metric": predictions[best_epoch],
                    "epoch_metric": predictions[epoch],
                }
            )
    return records, skipped


def run_family(path: Path):
    parts = path.parts
    try:
        runs_index = parts.index("runs")
        return parts[runs_index + 2]
    except (ValueError, IndexError):
        return None


def unrelated_adapter_records(root: Path):
    records = []
    skipped = []
    for parent in scan_adapter_parents(root):
        family = run_family(parent)
        if family is None or family in KEEP_RUN_FAMILIES:
            continue
        for path in sorted(parent.glob("adapter_epoch*")):
            if not path.is_dir() or not ADAPTER_RE.fullmatch(path.name):
                continue
            records.append(
                {
                    "kind": "unrelated_best_adapter",
                    "path": str(path.resolve()),
                    "bytes": tree_size(path),
                    "setting": str(parent.resolve()),
                    "run_family": family,
                }
            )
    return records, skipped


def runner_records(project_root: Path):
    records = []
    code_root = project_root / "code"
    for path in sorted(code_root.glob("run_*.sh")):
        if path.name in KEEP_RUNNERS:
            continue
        records.append(
            {
                "kind": "deprecated_experiment_runner",
                "path": str(path.resolve()),
                "bytes": file_size(path),
            }
        )
    return records


def apply_records(records: list[dict]) -> tuple[int, int]:
    removed = 0
    reclaimed = 0
    for record in records:
        path = Path(record["path"])
        if not path.exists():
            continue
        reclaimed += int(record["bytes"])
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed += 1
    return removed, reclaimed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("code/experiments"))
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(".planning/2026-09-22-storage-cleanup-audit/manifests"),
    )
    parser.add_argument(
        "--mode",
        choices=("deepspeed", "adapters", "unrelated-adapters", "runners"),
        required=True,
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.mode == "deepspeed":
        records = deepspeed_records(args.root)
        skipped = []
    elif args.mode == "unrelated-adapters":
        records, skipped = unrelated_adapter_records(args.root)
    elif args.mode == "runners":
        records = runner_records(Path.cwd())
        skipped = []
    else:
        records, skipped = adapter_records(args.root)
    manifest = args.manifest_dir / f"{stamp}_{args.mode}_{'apply' if args.apply else 'dryrun'}.jsonl"
    write_manifest(manifest, records)
    if skipped:
        write_manifest(manifest.with_name(manifest.stem + "_skipped.jsonl"), skipped)
    total = sum(int(record["bytes"]) for record in records)
    print(
        f"[MANIFEST] mode={args.mode} targets={len(records)} "
        f"bytes={total} GiB={total / 1024**3:.2f} skipped={len(skipped)} path={manifest}",
        flush=True,
    )
    if not args.apply:
        return
    removed, reclaimed = apply_records(records)
    print(
        f"[APPLIED] mode={args.mode} removed={removed} "
        f"planned_reclaimed_GiB={reclaimed / 1024**3:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
