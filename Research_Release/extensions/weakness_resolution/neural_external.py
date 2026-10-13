                                                                               
from pathlib import Path
import sys, json, shutil, hashlib, os
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'Research_Release/expanded_validation_results'
OUT=ROOT/'Research_Release/weakness_resolution_results/neural'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
common.OUT=OUT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import numpy as np
    import pandas as pd
                                                                                  
    import neural
    selected=[]
    for provider in ['EPA','AURN','NSW']:
        for folder in sorted(SOURCE.iterdir()):
            if not (folder/'dataset_manifest.json').exists():continue
            meta=json.loads((folder/'dataset_manifest.json').read_text())
            if meta['provider']!=provider:continue
            if not (folder/'outage_cases.csv').exists():continue
            dd=np.load(folder/'dataset.npz',allow_pickle=False)
            ds={k:dd[k] for k in dd.files}|{'times':pd.DatetimeIndex(dd['times'])}
            train=neural.cases(ds,0,80);val=neural.cases(ds,1,12)
            if len(train)>=10 and len(val)>=3:
                selected.append({'provider':provider,'cohort':folder.name,'training_cases':len(train),'selection_cases':len(val)})
                break
        else:raise RuntimeError('No training-eligible representative: '+provider)
    OUT.mkdir(parents=True,exist_ok=True)
    identity={'protocol_sha256':sha(HERE/'PROTOCOL.md'),'wrapper_sha256':sha(Path(__file__)),
              'selections':selected,'rule':'First lexicographic training-mask eligible cohort per provider; outcomes do not select cohorts'}
    p=OUT/'SELECTION.json'
    if p.exists() and json.loads(p.read_text())!=identity:raise RuntimeError('Selection identity changed')
    p.write_text(json.dumps(identity,indent=2))
    for item in selected:
        folder=SOURCE/item['cohort'];dest=OUT/item['cohort'];dest.mkdir(exist_ok=True)
        for filename in ['dataset.npz','models.pkl','outage_cases.csv']:
            src=folder/filename;target=dest/filename
            if target.exists() and sha(src)!=sha(target):raise RuntimeError('Frozen input mismatch')
            if not target.exists():shutil.copy2(src,target)
    neural.regions=lambda:[r['cohort'] for r in selected]
    neural.main()
    GRIN,commit=neural.official()
    replay=[]
    for item in selected:
        region=item['cohort'];ds=common.load(region);m=common.models(region);n,p=ds['Z'].shape[1:]
        adj=np.zeros((n,n),np.float32)
        for u,v in m['edges']:adj[u,v]=adj[v,u]=1
        saved=pd.read_csv(OUT/region/'grin_cases.csv',dtype={'station':str})
        choices=json.loads((OUT/region/'grin_choices.json').read_text());cases=neural.cases(ds,2)
        for choice in choices:
            state=neural.torch.load(OUT/region/f"grin_seed{choice['seed']}_width{choice['width']}_training.pt",map_location='cpu',weights_only=False)
            model=GRIN(adj,p,choice['width'],choice['width'],0.,d_emb=8);model.load_state_dict(state['best'])
            actual=pd.DataFrame(neural.evaluate(model,cases))
            joined=actual.merge(saved[saved.seed==choice['seed']],on=['day','station'],suffixes=('_replay','_saved'),validate='one_to_one')
            if len(joined)!=len(cases) or not np.allclose(joined.mse_replay,joined.mse_saved,rtol=1e-7,atol=1e-9):raise AssertionError('Neural checkpoint replay mismatch')
            replay.append({'cohort':region,'seed':choice['seed'],'cases':len(joined),'passed':True})
        print('External neural replay passed',region,flush=True)
    (OUT/'REPLAY.json').write_text(json.dumps({'status':'passed','vendor_commit':commit,'checks':replay},indent=2))

if __name__=='__main__':main()
