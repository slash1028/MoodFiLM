# MoodFiLM research code archive

This is the portable **code** snapshot for two-stage personality-aware emotion recognition. It contains the IEMOCAP/MELD trainer under `portable/` and the MSP-IMPROV adaptation under `msp_improv/`. Start with [HANDOFF.md](HANDOFF.md) on a new cluster. Environment setup is in [ENVIRONMENT.md](ENVIRONMENT.md); dataset preparation and the expected local directory tree are in [DATASETS.md](DATASETS.md).

Three representative lightweight checkpoints are stored as 4 MiB parts in `weights_parts/`; run `python restore_weights.py` after cloning and verify them against [WEIGHTS.md](WEIGHTS.md). Qwen2.5-Omni-7B base weights, licensed datasets, precomputed audio features, and experiment outputs are absent. Restore these separately before a GPU run. This archive is not a self-contained reproduction of historical scores.

The MSP stage uses a **provisional** dialogue order. Its turn ties and speaker-role assumptions must be audited before treating its ERC scores as chronological dialogue-level results. Stage2 currently computes test metrics every epoch; use validation for model selection in a publication-grade comparison.

`portable/run.py` is a preflight/launcher for the archived IEMOCAP and MELD configurations. `msp_improv/run_stage1_gpu5.sh` and `run_stage2_gpu5.sh` are the MSP entry points; set `PYTHON_BIN` and `GPU` for the destination cluster. Existing code in `portable/code/` also includes older research utilities. Some of those utilities have historical absolute path defaults; pass explicit paths or revise the defaults when using them.
