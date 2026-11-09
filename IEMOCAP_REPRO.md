# Reproducing the archived IEMOCAP ERC checkpoint

The archived IEMOCAP Stage2 checkpoint was recorded as test WF1 **74.759**. It is a text + native-Qwen-mel experiment with **ground-truth PA-IEMOCAP personality**, no video, and a 12-utterance history window. Use `portable/adapters/iemocap/trainable_model.safetensors` after `python restore_weights.py`. Evaluation loads this final ERC checkpoint; it does not retrain Stage1 or Stage2.

## What was recovered from the old run

The old portable copy contains the exact JSONL prompts and split order: train **5,163**, valid **647**, test **1,623** samples. The original `iemocap.pkl` has the dialogue order, speaker tags and filtering, while `run_manifest.json` records `personality_input_mode=gt`, `personality_condition=True_continuous`, `personality_meta=True`, and `historical_window=12`. The prepared prompts already contain GT OCEAN values; they do not need an online personality prediction. Their text has no `[AUDIO_UTTERANCE]` markers: the old audio path attaches native mel by sample/dialogue ID in the data loader, so absence of that marker alone is not evidence of missing audio.

`scripts/make_iemocap_split_index.py` can create a **local** `repro/iemocap_split_index.json` from the old prepared files. The index records each sample ID, aligned utterance ID, original `iemocap.pkl` speaker tag and sample order, without transcription, emotion targets, OCEAN values or audio. The generated index is not published in this repository. Do not infer the speaker tag from the utterance ID: in six original samples the two disagree; all 7,433 GT personality vectors match the public PA-IEMOCAP file when keyed by the original `iemocap.pkl` tag.

| Local licensed asset (under `portable/`) | Bytes | SHA-256 |
|---|---:|---|
| `data/iemocap/train.json` | 11,070,062 | `1cccecc33934d6caca00129b165bfac0e0070dc78c48939328a3143cc1c9836c` |
| `data/iemocap/valid.json` | 1,357,173 | `9dfeed487a221e4f6ccebace6705eeaa1d86100413766cc76111dbad99216926` |
| `data/iemocap/test.json` | 3,581,457 | `acca6e1760d3e49c4d990f2556a188ccd52b04d5a3a5616a4fc92aa41e56f91c` |
| `original_data/iemocap/iemocap.pkl` | 512,358 | `85f15be1fbd8678e0dd443a11ac609615f0e990cc0ca1551fdf86f98b9e6067f` |
| `original_data/iemocap/iemocap_qwen_native_mel_features.pkl` | 1,747,158,827 | `ca9edbf84d82ade227cb08dded54b1f34d1927882455e383dfb41d856ce75c2e` |

The source prepared files are not in this **public** GitHub repository. The [USC IEMOCAP release agreement](https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf) restricts transfer of the dataset and derivatives. The exact prepared JSONL files contain IEMOCAP transcriptions and labels; the 1.7 GB mel cache derives from audio. Move them to a new machine only through a channel permitted by the relevant license. The code repository contains their fingerprints and split index, not their contents.

## Public GT personality source

PA-IEMOCAP publishes `ocean.csv` at [its official GitHub repository](https://github.com/Kyoto-University-Speech-and-Audio/PA-IEMOCAP), with an MIT license. Our verified source file has 302 data rows and SHA-256 `4772bdc9f013a3d6c4e8b038e559c5633092ce68e5edb3fed0aef99e4f9689ec`. `scripts/check_iemocap_repro.py --download-ocean` fetches it directly into the ignored `portable/datasets/PA-IEMOCAP/` path and checks the hash. The CSV columns `OP, CO, EX, AG, NE` map to `O, C, E, A, N` in the prompts. Cite the PA-IEMOCAP authors when using those labels.

## Recommended migration and evaluation sequence

1. Clone this repository, create the environment in `ENVIRONMENT.md`, run `python restore_weights.py`, and install the Qwen2.5-Omni-7B base model under `portable/LLM_bases/Qwen2.5-Omni-7B/`.
2. Restore the five licensed local assets in the table above if your data access agreement permits. If the source cache cannot be moved, regenerate native mel with the shipped extractor once `iemocap.pkl`, raw IEMOCAP and the Qwen processor are available:

   ```bash
   python portable/code/audio_utils/extract_iemocap_qwen_processor_features.py \
     --iemocap_pkl portable/original_data/iemocap/iemocap.pkl \
     --iemocap_root portable/datasets/IEMOCAP_full_release \
     --qwen_processor_path portable/LLM_bases/Qwen2.5-Omni-7B \
     --output_pkl portable/original_data/iemocap/iemocap_qwen_native_mel_features.pkl
   ```

3. Recreate the local split index from the authorized old files, then download/check public GT and the old split protocol:

   ```bash
   python scripts/make_iemocap_split_index.py --source-root portable
   python scripts/check_iemocap_repro.py --download-ocean --require-data
   python scripts/check_iemocap_repro.py --require-data --check-cache-hash
   ```

4. Preflight, then run a small real-batch checkpoint/audio smoke before a full test evaluation. The archived launcher uses `--mode eval` to load the final IEMOCAP ERC snapshot:

   ```bash
   python portable/run.py --dataset iemocap --gpus 0 --mode eval --check-imports
   python portable/run.py --dataset iemocap --gpus 0 --mode eval --execute
   ```

`--execute` runs the full evaluation and needs an allocated GPU. The CLI `--gpus` takes **physical** GPU IDs. The code has not yet been run on the new image; passing a file/CLI preflight does not establish a 74.759 reproduction. If the prepared prompts or original pkl cannot be restored, regenerate them from licensed raw data and public PA labels. The exact old index cannot be regenerated without the original prepared files; compare any rebuilt output against the counts/hashes above and report it as a **rebuilt-protocol evaluation** unless the prompts and cache hashes match. Do not choose settings by repeatedly checking the test score.
