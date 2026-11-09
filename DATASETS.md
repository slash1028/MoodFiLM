# Dataset acquisition and local adaptation

Use the root of this checkout as `$PROJECT`. Keep datasets, model weights, audio feature caches and generated JSON outside version control. Relative paths below match the code. If raw data lives elsewhere, symlink its **directory** into the expected path or change the explicit CLI arguments; inspect symlink targets before launching jobs.

## MSP-IMPROV

Obtain the licensed release with `Readme.txt`, `Evaluation.txt`, `Human_transcriptions.zip`, and `Audio/session1.zip` through `Audio/session6.zip`. The video and phonetic alignment archives are not used by this text+audio path. Place files in `$PROJECT/dataset/MSP-IMPROV/`, then:

```bash
cd "$PROJECT/dataset/MSP-IMPROV"
unzip -n Human_transcriptions.zip
for s in 1 2 3 4 5 6; do unzip -n "Audio/session${s}.zip" -d Audio; done
cd "$PROJECT"
python msp_improv/audit.py --output msp_improv/utterances.jsonl
python msp_improv/build_provisional_dialogues.py
```

Expected extracted files include `Audio/session1/S01A/P/*.wav` and `All_human_transcriptions/*.txt`. The audit expects 8,438 audio/transcript/evaluation matched utterances for the source release inspected here; resolve any mismatch before continuing. The `utterances.jsonl` paths are absolute at generation time, so regenerate it after migration. Session 1–4 are train, 5 valid, 6 test. Four-class ERC keeps A/H/S/N; X/O are excluded. The provisional S/T scene grouping contains unresolved order ties and assumes README speaker roles; use only as an exploratory ERC protocol. Stage1 requires the IEMOCAP checkpoint and generates `msp_improv/cache/qwen_native_mel.pkl` from WAVs. Start with `msp_improv/README.md` for its exact CLI.

## MELD

Obtain MELD.Raw and place or link it at `$PROJECT/portable/datasets/MELD.Raw/`. The builder expects `train_splits/train_sent_emo.csv`, `dev_sent_emo.csv`, and `test_sent_emo.csv`; it uses the accompanying audio files for feature extraction. Keep stable split/dialogue/utterance IDs. To build PELD-labelled textual ERC samples, first restore `portable/original_data/meld/peld_personality_map.json`, then run:

```bash
python portable/code/meld_utils/build_meld_gtperson_stage2_data.py \
  --meld-raw-root portable/datasets/MELD.Raw \
  --peld-personality-json portable/original_data/meld/peld_personality_map.json \
  --output-dir portable/data/meld --context-window 6 --include-audio-markers
```

The archived `portable/configs/meld.json` needs `portable/original_data/meld/meld_qwen_audio_dynamic8to24_f6_splitkey.pkl`, a precomputed Qwen 8–24-token audio cache. The MELD dialogue-personality inference instead needs `meld_qwen_native_mel_features_splitkey.pkl`; these are different inputs. The code includes `portable/code/audio_utils/extract_meld_qwen_audio_tokens.py` and `extract_meld_qwen_processor_features.py`, but their CLI defaults reference old lab paths; inspect `--help`, set every source/model/output path explicitly, and audit cache coverage before training. The PELD map is not provided by MELD.Raw and must be restored separately. `portable/dialogue_personality/infer.py` can generate MELD dialogue-level predictions once the 0.7648 checkpoint and native mel cache are present.

## IEMOCAP

Obtain the licensed IEMOCAP full release and place/link it at `$PROJECT/portable/datasets/IEMOCAP_full_release/`, preserving `Session*/dialog/EmoEvaluation`, transcriptions and utterance WAV folders. The exact archived `portable/data/iemocap/{train,valid,test}.json` and `portable/original_data/iemocap/iemocap.pkl` are now included. The public PA-IEMOCAP `ocean.csv` and its MIT license are under `third_party/PA-IEMOCAP/`; run `python scripts/check_iemocap_repro.py --install-ocean` to place a verified local copy at the old expected path. The 1.7 GB native mel cache is not included; regenerate it with the command in `IEMOCAP_REPRO.md`. The source tree includes `portable/code/data_process.py` and `portable/code/audio_utils/extract_iemocap_qwen_processor_features.py`; their old path defaults require explicit CLI paths. See `IEMOCAP_REPRO.md` before evaluating the 74.759 ERC checkpoint.
