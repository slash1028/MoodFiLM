# Representative lightweight checkpoints

Three selected `trainable_model.safetensors` snapshots are stored as 4 MiB parts in `weights_parts/`. Run `python restore_weights.py` after cloning; it reassembles them and checks SHA-256. They are **not** Qwen base weights or optimizer states. The trainer loads these files with `model.load_state_dict(..., strict=False)` after constructing the matching base + LoRA model; `adapter_model.safetensors` duplicates the LoRA portion and is unnecessary for this loader.

| Purpose | File | Bytes | SHA-256 | Provenance |
|---|---|---:|---|---|
| IEMOCAP Stage1 personality (reported average CCC 0.7648) | `portable/dialogue_personality/checkpoint_07648/trainable_model.safetensors` | 43,306,016 | `c9dfa86a121d3db6a6c1f4f97448f57637acfa790472b0de62995606a83a303e` | Existing 0.7648 adapter used by MELD/MSP dialogue inference |
| IEMOCAP Stage2 ERC (historical WF1 74.759) | `portable/adapters/iemocap/trainable_model.safetensors` | 24,281,730 | `128042c975ab66b482eb313f20f82ac87f847eed5a226c0c416027a60382c621` | `89_smf_prompt_upper6_0902/iemo_gtp_s150` metadata; copied from `InstructERC-Adapters/iemocap/erc-wf1-74.759` |
| MELD Stage2 simplified MoodFiLM b050 epoch4 | `portable/adapters/meld/trainable_model.safetensors` | 29,811,026 | `d2e624fd68206b8e128f986b8fe0ff798b96acd10b97aadeeff34059d6ce1312` | `89_smf_prompt_upper6_0902/meld_pp_b050`; copied from `InstructERC-Adapters/meld/simplified-moodfilm-b050-epoch4` |

The Stage1 file is all LoRA parameters, including audio encoder LoRA. The IEMOCAP/MELD Stage2 files additionally include MoodFiLM trainable parameters; MELD includes the precomputed-audio adapter. `portable/adapters/{iemocap,meld}/model_args.json` and `adapter_config.json` are retained as provenance/configuration. Older source paths in `model_args.json` are historical and must not be used unchanged on another cluster. The actual trainer uses the remapped configs in `portable/configs/`.

Do not confuse these snapshots with complete model weights. Restore Qwen2.5-Omni-7B separately, then run `sha256sum` on the three files before inference. The IEMOCAP Stage1 inference and archived IEMOCAP training config read the checkpoint path shown in this table. The copy formerly at `portable/adapters/stage1_init` had the same hash and is omitted to avoid duplication.
