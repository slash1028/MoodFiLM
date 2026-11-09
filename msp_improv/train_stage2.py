"""Matched single-GPU MSP baseline and predicted-personality MoodFiLM sweep."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PORTABLE = ROOT.parent / 'portable'
sys.path[:0] = [str(PORTABLE), str(PORTABLE/'code')]
from run import training_parser
from meld_utils import build_meld_gtperson_stage2_data as builder
from case_study.add_personality_prompt import inject, prompt_clause

builder.EMOTION_CHOICES = '<angry, happy, sad, neutral>'
SETTINGS = [('01_ta_no_personality', False), ('02_ta_predicted_moodfilm', True)]


def cli_args(cfg):
    import argparse as ap
    parser = training_parser()
    actions = {a.dest:a for a in parser._actions}
    actions['dataset'].choices = ['iemocap','meld','msp_improv','EmoryNLP']
    result=[]
    for key,value in cfg.items():
        if value is None: continue
        if key not in actions:
            if key not in {'model_type','do_sample','preprocess_inputs','open_ended'}:
                raise ValueError(f'Unknown CLI key: {key}')
            continue
        action=actions[key]
        if isinstance(action,ap._StoreTrueAction):
            if value:result.append(action.option_strings[0])
        else:result.extend([action.option_strings[0],str(value)])
    parser.parse_args(result)
    return result


def summarize(output):
    rows=[]
    for name,_ in SETTINGS:
        for path in sorted((output/name).glob('preds_for_eval_[0-9]*.text')):
            with path.open(encoding='utf-8') as f:
                report,_=json.JSONDecoder().raw_decode(f.read(65536))
            score=report['emotion']['score']
            rows.append({'setting':name,'epoch':int(path.stem.rsplit('_',1)[1])+1,
                         'WF1':score['F1_SA'],'ACC':score['Acc_SA']})
    with (output/'epoch_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['setting','epoch','WF1','ACC'])
        writer.writeheader();writer.writerows(rows)
    print('CURRENT METRICS',rows,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=5)
    p.add_argument('--gpus', help='Comma-separated physical GPU IDs; overrides --gpu')
    p.add_argument('--only', choices=['baseline','moodfilm'], help='Run one setting only')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--port', type=int, default=29766)
    p.add_argument('--run-id', default='msp_ta_moodfilm_s42_1104')
    p.add_argument('--prediction-csv', type=Path)
    p.add_argument('--run', action='store_true')
    opt=p.parse_args()
    if opt.gpu<0 or opt.epochs<1 or Path(opt.run_id).name!=opt.run_id:
        p.error('Invalid GPU, epochs or run-id')
    gpus=opt.gpus or str(opt.gpu)
    gpu_ids=gpus.split(',')
    if not all(s.isdigit() for s in gpu_ids) or len(set(gpu_ids))!=len(gpu_ids):
        p.error('Use distinct physical GPU numbers, e.g. --gpus 0,1')
    if 40 % len(gpu_ids):
        p.error('GPU count must divide global batch 40')
    settings=[s for s in SETTINGS if opt.only is None or s[1] == (opt.only=='moodfilm')]
    candidates=sorted((ROOT/'results').glob('*/export/dialogue_personality.csv'))
    pred=opt.prediction_csv or (candidates[-1] if candidates else None)
    if not pred or not pred.is_file(): raise SystemExit('No complete Stage1 export')
    values={}
    with pred.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key=(row['dialogue_id'],row['speaker'])
            if key in values:raise ValueError(f'Duplicate prediction: {key}')
            raw=[float(row[t]) for t in 'OCEAN']
            if not all(1<=v<=7 for v in raw):raise ValueError(f'Invalid OCEAN: {key}')
            values[key]=[(v-1)/6 for v in raw]
    dialogues={'train':{},'valid':{},'test':{}}
    expected=set()
    id_by_index={}
    for did,line in enumerate((ROOT/'provisional_dialogues.jsonl').open(encoding='utf-8')):
        dialogue=json.loads(line);split=dialogue['split']; turns=[]
        id_by_index[did]=dialogue['dialogue_id']
        for turn in dialogue['turns']:
            speaker=turn['speaker_from_readme']
            expected.add((dialogue['dialogue_id'],speaker))
            turns.append({'speaker':speaker,'utterance':turn['text'],
                          'sample_id':turn['id'],'emotion':turn['emotion'],
                          'eligible':turn['eligible_four_class']})
        dialogues[split][did]=turns
    if set(values)!=expected:raise ValueError(f'Prediction coverage mismatch: missing {len(expected-set(values))}, extra {len(set(values)-expected)}')
    base=json.loads((PORTABLE/'configs/meld.json').read_text())
    base.update(dataset='msp_improv',model_name_or_path=str(PORTABLE/'LLM_bases/Qwen2.5-Omni-7B'),
                audio_qwen_processor_path=str(PORTABLE/'LLM_bases/Qwen2.5-Omni-7B'),
                audio_feature_pkl=str(ROOT/'cache/qwen_native_mel.pkl'),
                audio_embedding_source='precomputed_pkl',checkpoint_dir=None,
                audio_context_window=6, batch_size=40,gradient_accumulation_steps=40//len(gpu_ids),
                eval_batch_size=1,num_train_epochs=opt.epochs,seed=42,
                speaker_mood_source='predicted',speaker_mood_dialogue_ordered=False,
                save_best_checkpoint=False,disable_deepspeed_checkpoint_save=True,
                save_lightweight_adapter_epochs='all')
    for path in (Path(base['model_name_or_path']),Path(base['audio_feature_pkl'])):
        if not path.exists():raise FileNotFoundError(path)
    output=ROOT/'stage2_runs'/opt.run_id
    if output.exists():raise FileExistsError(f'Run exists; choose another --run-id: {output}')
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps({'prediction_csv':str(pred.resolve()),
       'splits':{'train':'session1-4','valid':'session5','test':'session6'},
       'ordering':'approximate_filename_order_v1','audio':'native_qwen_mel',
       'labels':['angry','happy','sad','neutral'],'seed':42},indent=2)+'\n')
    runs=[]
    os.chdir(PORTABLE)
    for name,mood in settings:
        data=output/'data'/name;data.mkdir(parents=True)
        for split,ds in dialogues.items():
            profile={(split,did,s):values[(id_by_index[did],s)]
                     for did,turns in ds.items() for s in {t['speaker'] for t in turns}} if mood else {}
            records=builder.build_records(split,ds,{},6,'none',False,{},True,profile)
            eligible={t['sample_id'] for turns in ds.values() for t in turns if t['eligible']}
            records=[r for r in records if r['sample_id'] in eligible]
            for row in records:
                row['personality']=row['gt_personality'] if mood else None
                row['personality_labels_available']=False
                if mood and row['personality'] is not None:
                    row['input']=inject(row['input'],prompt_clause(row['personality']))
            builder.write_jsonl(data/f'{split}.json',records)
            print(name,split,len(records),flush=True)
        folder=output/name;folder.mkdir()
        ds=folder/'deepspeed.json'
        ds.write_text(json.dumps({'train_batch_size':40,'train_micro_batch_size_per_gpu':1,
              'gradient_accumulation_steps':40//len(gpu_ids),'bf16':{'enabled':True},'zero_optimization':{'stage':0}}))
        cfg=dict(base,data_dir=str(data),output_dir=str(folder),run_id=name,
                 deepspeed_config=str(ds),use_speaker_mood_dynamics=mood,
                 use_speaker_mood_prior_loss=mood,use_speaker_mood_dynamics_consistency_loss=mood)
        (folder/'resolved_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        runs.append((name,folder,cli_args(cfg)))
    if not opt.run:
        print(f'Prepared {len(settings)} setting(s) in {output}; add --run to train')
        return
    env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
    env['PYTHONPATH']=str(ROOT/'code')+os.pathsep+str(PORTABLE/'code')+os.pathsep+env.get('PYTHONPATH','')
    for name,folder,argv in runs:
        print('START',name,'GPUS',gpus,flush=True)
        cmd=[sys.executable,'-m','deepspeed.launcher.runner','--include',f'localhost:{gpus}',
             '--master_port',str(opt.port),str(ROOT/'code/main_new.py')]+argv
        with (folder/'terminal.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=PORTABLE,env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            if proc.wait():raise RuntimeError(f'{name} failed; see {folder}/terminal.log')
        summarize(output)
    print('FINISHED',output,flush=True)


if __name__=='__main__':main()
