# MELD dialogue-level personality rerun (prepared, not yet executed)

Inputs are anonymized full conversations, one target character per sample.
Every utterance has one audio insertion marker and an ordered, split-qualified
audio ID. All speakers' audio is requested, not just the target character.
The six main characters are prediction targets; other speakers remain in context.
All three dataset splits are included in inference `test.json`; the original
split is retained in each row. `train.json` and `valid.json` are empty by design.
These are inference inputs, NOT a training dataset.

The target [4,4,4,4,4] is a loader placeholder, NOT a personality annotation.
Do not interpret any automatically produced CCC/MAE against this placeholder.
Never aggregate test/valid predictions into training-time speaker profiles.

Before inference, select a verified post-fix A+T Stage1 checkpoint and retain
its audio representation, projection, token strategy, and context settings.
The old archived CCC 0.7648 checkpoint is not automatically a verified A+T model.
No checkpoint was selected and no GPU inference has been launched by this preparation.

Required audit before a full run:
1. Explicit personality_media_scope=full, audio_enable=True, text_enable=True.
2. Check every requested ID exists in the selected cache; report missing IDs.
3. Inspect an actual collated batch: nonzero audio refs/features, aligned order,
   actual token counts, and truncation. Markers alone do not prove audio reaches LLM.
4. Compare a small real batch against muted/shuffled audio and confirm logits change.
5. Only then run all dialogue-character inputs. Use the checkpoint's training audio
   path; do not silently substitute MELD compressed tokens for native-mel training.
   Current native-mel loader still contains an IEMOCAP-only guard; a native-mel
   checkpoint therefore needs a separately tested MELD extension before this run.

Export with code/meld_utils/export_meld_dialogue_personality.py, then:

    python code/meld_utils/summarize_dialogue_personality.py OUTPUT/dialogue_personality.csv --output OUTPUT/dispersion.json

Inspect within-character across-dialogue std separately from differences between
character averages. Larger variance alone does not prove better personality prediction.

The surrounding portable migration may still be incomplete. Do not start until
base weights, matching audio cache, selected adapter, and independent env are ready.
