# Agent handoff

Read `HANDOFF.md`, `ENVIRONMENT.md`, `DATASETS.md`, `IEMOCAP_REPRO.md` and `WEIGHTS.md` before changing code or launching training. This repository carries three lightweight checkpoints as parts; run `python restore_weights.py`. The exact IEMOCAP JSONL split files and source pkl are included, while the large base model, raw media and audio caches are omitted. Never infer that a preflight succeeded merely because imports work: check all split files, model shards, adapters, audio-cache keys, and a real training/inference batch.

Keep generated data, logs, model weights, and caches out of Git. Preserve original split/session IDs and sample IDs. In MSP-IMPROV, `provisional_dialogues.jsonl` is an approximation with unresolved chronology; do not silently convert it to verified dialogue order. In MELD, predicted dialogue personality is an offline full-dialogue feature and may include future turns. Do not report placeholder personality CCC as ground-truth performance.

When moving to another cluster, first inventory available licensed datasets, model weights, adapters and CUDA software. Rebuild local paths under the cloned project, run static/CPU audits, then a two-sample GPU smoke, then full runs. Record the exact config, cache source, git commit, seed, split and model-selection rule with each result.
