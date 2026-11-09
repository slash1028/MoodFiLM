# New-cluster handoff

## What is here

- `portable/code/main_new.py`: common Qwen2.5-Omni trainer and evaluation code for IEMOCAP/MELD. `portable/code/data_utils`, `audio_utils`, `meld_utils`, `speaker_mood_recurrence.py`, and related modules are its dependencies.
- `portable/configs/{iemocap,meld}.json` and `portable/run.py`: archived five-epoch launcher configs. Inspect these before asserting that they match a published/selected historical run.
- `portable/dialogue_personality/infer.py`: apply the existing 0.7648 IEMOCAP Stage1 checkpoint to MELD dialogues; no training.
- `portable/meld_dialogue_sweep10.py`: 10-setting MELD comparison using PELD, predicted dialogue personality, prompt-only, and no-personality variants.
- `msp_improv/audit.py` through `infer_stage1.py`: MSP alignment, provisional dialogue building, native mel extraction and Stage1 inference.
- `msp_improv/train_stage2.py` and `msp_improv/code/`: MSP baseline and predicted-personality MoodFiLM trainer. Its local `code/main_new.py` and `code/data_utils/data_utils.py` carry MSP-specific changes; do not replace them with the portable originals without porting those changes.

## Restore dependencies

Run `python restore_weights.py` to reconstruct the three representative checkpoints from repository parts; it verifies SHA-256. Consult `WEIGHTS.md` for provenance. Rebuild and check the Python/CUDA environment using `ENVIRONMENT.md` before a GPU job. The old `/n/work3/huang/...` Python path is **not** a new-cluster dependency. Use `PYTHON_BIN=/absolute/path/to/your/env/bin/python` for shell wrappers or invoke that Python directly. Set `NCCL_IB_DISABLE=1 NCCL_NET=Socket` if the new node's InfiniBand path is unstable; test the network backend before a long run.

Populate these paths under the clone (or use symlinks to authorized storage):

```text
portable/LLM_bases/Qwen2.5-Omni-7B/          # full model + processor/tokenizer
portable/dialogue_personality/checkpoint_07648/ # IEMOCAP Stage1 adapter
portable/adapters/iemocap/                  # if reusing archived ERC adapter
portable/adapters/meld/                     # if reusing archived ERC adapter
portable/original_data/iemocap/             # iemocap.pkl included; regenerate native mel cache
portable/original_data/meld/                # native mel, 8-24 token cache, PELD map
portable/data/iemocap/{train,valid,test}.json # included for archived ERC evaluation
portable/data/meld/{train,valid,test}.json
dataset/MSP-IMPROV/                          # raw MSP files; see DATASETS.md
```

The scripts reference `checkpoint_07648/trainable_model.safetensors`, including the archived IEMOCAP training config. Check that `restore_weights.py` produced all three files with valid hashes. The Qwen base model needs all safetensor shards and processor files. Keep licensed IEMOCAP/MSP data under the dataset license.

## First-pass commands

Run from this directory after restoring assets:

```bash
PYTHON_BIN=/path/to/env/bin/python
"$PYTHON_BIN" portable/run.py --dataset iemocap --gpus 0 --check-imports
"$PYTHON_BIN" portable/run.py --dataset meld --gpus 0 --check-imports
"$PYTHON_BIN" msp_improv/audit.py --output msp_improv/utterances.jsonl
"$PYTHON_BIN" msp_improv/build_provisional_dialogues.py
GPU=0 PYTHON_BIN="$PYTHON_BIN" bash msp_improv/run_stage1_gpu5.sh
GPU=0 PYTHON_BIN="$PYTHON_BIN" bash msp_improv/run_stage2_gpu5.sh
```

The first two commands perform preflight only; add `--execute` to `portable/run.py` to train. For the 10-setting MELD dialogue experiment, restore the prediction CSV and PELD map, then use `GPU=0 PYTHON_BIN="$PYTHON_BIN" bash portable/run_meld_dialogue10_1gpu.sh --run` (read its dry-run options first). `msp_improv/run_stage1_gpu5.sh` extracts mel if missing, audits a small batch, then performs full prediction; output goes to `msp_improv/results/`. `run_stage2_gpu5.sh` trains both MSP settings and writes `msp_improv/stage2_runs/`. Do not launch these GPU commands until the data and adapters are in place.

## Research boundaries

IEMOCAP uses dialogue-level Stage1 personality prediction and contextual ERC with text plus native mel in the archived config. MELD MoodFiLM uses compressed Qwen audio tokens in the archived config; MELD dialogue-personality inference uses native mel. MSP Stage1 uses native mel and the IEMOCAP checkpoint; Stage2 uses native mel and compares no personality against predicted personality. MSP has no ground-truth OCEAN labels, so do not compute personality CCC there.

The historical IEMOCAP 0.7648 CCC and MELD ~70 WF1 are **references**, not verified by this code archive. `portable/run.py` checks files and CLI but cannot recreate missing caches or adapters. `portable/code/data_process.py` and several older extractors still have lab-specific absolute defaults; override their CLI paths or update them for the new cluster before use. See `DATASETS.md` for what can be rebuilt from raw files and what requires an additional cache/label source.
