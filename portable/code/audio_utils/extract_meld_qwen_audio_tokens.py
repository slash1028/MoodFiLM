#!/usr/bin/env python3
"""Extract compact Qwen2.5-Omni audio tokens for MELD.

The MELD native mel cache uses the same qwen_native_mel_v1 schema as the
IEMOCAP extractor, so the tokenization/compression implementation is shared.
"""

from extract_iemocap_qwen_audio_tokens import main


if __name__ == "__main__":
    raise SystemExit(main())
