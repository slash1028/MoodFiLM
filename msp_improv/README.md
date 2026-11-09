# MSP-IMPROV adaptation

The existing IEMOCAP personality checkpoint is at
`../portable/dialogue_personality/checkpoint_07648`. The intended protocol is
sessions 1–4 train, session 5 validation, session 6 test. ERC uses A/H/S/N;
X/O are excluded. All experiment outputs should remain under this directory.

First run `python msp_improv/audit.py`. Audio, transcription and evaluation
identifiers match after changing evaluation prefix `UTD-IMPROV` to
`MSP-IMPROV`, but this does **not** establish chronological dialogue order.
Run `python msp_improv/audit.py --output msp_improv/utterances.jsonl` to write
the 8,438 aligned per-utterance records, including split, original file paths,
text and four-class label eligibility. `dialogue_id` and `turn_index` remain
null intentionally.
The filename's scene, actor, scenario and FM/MF number are insufficient on
their own to prove which utterance followed which. The supplied README gives
no mapping from these fields to an ordered dialogue. In line with the agreed
protocol, Stage1 full-dialogue personality inference and contextual ERC must
not run on alphabetically sorted or synthetic dialogues. Obtain verified
recording/turn metadata first. The audit reports candidate groups and leaves
`stage1_ready=false` until the chronological mapping can be established.

No MSP training or inference has been launched.

For an exploratory approximation only, run
`python msp_improv/build_provisional_dialogues.py`. This creates
`provisional_dialogues.jsonl` from S/T segments grouped by session, scene and
actor code. It sorts local turn numbers and deterministically resolves ties,
recording every uncertain position. The suffix's second letter is the speaker
according to README, but needs audio validation before use as ground truth.
This provisional file must not be described as recovered chronological ERC.

## Provisional Stage1 personality inference

`prepare_stage1.py` creates one target-speaker prompt for each available
speaker in a candidate scene. Its `audio_turn_sample_ids` match every
`[AUDIO_UTTERANCE]` marker, and both speaking roles are represented by the
README's FM/MF convention. The checkpoint is the existing IEMOCAP 0.7648
adapter; labels in the input file are placeholders, not personality ground
truth. Audio is native Qwen mel extracted from the actual MSP WAVs.

To prepare prompts without GPU: `python msp_improv/infer_stage1.py --gpu 5`.
To extract audio, audit attachment, run a two-sample GPU smoke inference and
then full inference: `bash msp_improv/run_stage1_gpu5.sh`. The script prints
progress and writes outputs under `msp_improv/results/<timestamp>/`.

The export is exploratory. File order has unresolved ties, and README-derived
speaker roles have not been verified by listening to or classifying voices.
Do not treat predicted values as measured personality or use placeholder
targets for CCC. Stage2 training requires a matched no-personality baseline
and validation-set model selection; it is not launched by this script.

## Provisional Stage2 comparison

The completed Stage1 export is checked for exact coverage before preparing
two sequential five-epoch runs: `01_ta_no_personality` and
`02_ta_predicted_moodfilm`. Both use the same 4-class targets, approximate
scene order, w6 textual/audio context, Qwen native mel, seed 42 and session
split. This is a mel-based comparison; it is not directly matched to the
older compressed-token MELD result. Outputs go under `stage2_runs/<run-id>`.

Prepare and inspect without GPU:
`python msp_improv/train_stage2.py --gpu 5 --run-id my_dry_run`.
Train on GPU 5:
`bash msp_improv/run_stage2_gpu5.sh`.

The current trainer evaluates the test split after every epoch and writes
`epoch_metrics.csv`; these are exploratory test metrics. Do not select a
checkpoint by the highest test score and report it as held-out performance.
