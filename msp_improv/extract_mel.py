"""Extract Qwen native mel features for MSP provisional scene turns."""
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORTABLE = ROOT.parent / 'portable'
sys.path.insert(0, str(PORTABLE / 'code/audio_utils'))
from extract_iemocap_qwen_processor_features import extract_qwen_native_mel_features
from transformers import AutoProcessor


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    source = ROOT / 'provisional_dialogues.jsonl'
    output = ROOT / 'cache/qwen_native_mel.pkl'
    if output.exists():
        print(f'Using existing cache: {output}')
        return
    wanted = {}
    for line in source.open(encoding='utf-8'):
        for turn in json.loads(line)['turns']:
            wanted[turn['id']] = turn['audio_path']
    model = PORTABLE / 'LLM_bases/Qwen2.5-Omni-7B'
    processor = AutoProcessor.from_pretrained(str(model), trust_remote_code=True, local_files_only=True)
    extractor = processor.feature_extractor
    sr = int(extractor.sampling_rate)
    features, lengths = {}, {}
    items = sorted(wanted.items())
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start+args.batch_size]
        extracted = extract_qwen_native_mel_features(processor, [path for _, path in batch], sr)
        for (sid, _), (feat, length) in zip(batch, extracted):
            features[sid], lengths[sid] = feat, length
        if start % 400 == 0:
            print(f'Mel {len(features)}/{len(items)}', flush=True)
    payload = {'schema':'qwen_native_mel_v1', 'source':'msp_improv_provisional',
               'sampling_rate':sr, 'hop_length':int(extractor.hop_length),
               'feature_dim':128, 'sample_features':features,
               'sample_feature_lengths':lengths, 'num_sample_features':len(features)}
    output.parent.mkdir(exist_ok=True)
    temp = output.with_suffix('.partial')
    with temp.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(output)
    print(f'Saved {len(features)} mel features: {output}')


if __name__ == '__main__':
    main()
