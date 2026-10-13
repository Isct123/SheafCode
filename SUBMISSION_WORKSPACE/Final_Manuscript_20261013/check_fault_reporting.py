from pathlib import Path
import os,sys,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];BASE=ROOT/'Research_Release'
sys.path.insert(0,str(BASE/'extensions/recent_data'))
import common,fault_partial,faults
import numpy as np,pandas as pd
checks=[]
for region in ['delhi','epa_la']:
    ds=common.load(region);m=common.models(region);Z=ds['Z'];n,p=Z.shape[1:];D=m['sheaf'].D
    saved=json.loads((BASE/'recent_data_results'/region/'available_fault_status.json').read_text())
    assert saved['source_sha256']==hashlib.sha256(Path(fault_partial.__file__).read_bytes()).hexdigest()
    train=Z[(ds['split']==0)&np.isfinite(Z).all((1,2))];decoder=fault_partial.AvailableDecoder(D,train,n,p);decoder.original=D
    candidates=np.isfinite(Z).all(2);rng=np.random.default_rng(2119);X=[];Y=[]
    times=np.flatnonzero((ds['split']==1)&(candidates.sum(1)>=2))[::3]
    for t in times:
        universe=np.flatnonzero(candidates[t])
        for k in [0,1,2]:
            F=tuple(sorted(rng.choice(universe,k,replace=False)));kind=rng.choice(['same','opposite','random','unequal'])
            y=faults.families(Z[t],Z[max(t-1,0)],ds['mu'],ds['sd'],F,kind,23,rng);X.append(decoder.features(y)[0]);Y.append(k)
    X=np.asarray(X);Y=np.asarray(Y);cut=len(X)//2;best=None
    for ridge in [.1,1,10]:
        trial=fault_partial.AvailableDecoder(D,train,n,p);trial.fit(D,X[:cut],Y[:cut],ridge)
        pred=np.argmax(np.c_[np.ones(len(X)-cut),(X[cut:]-trial.fm)/trial.fs]@trial.B,1);accuracy=float(np.mean(pred==Y[cut:]))
        if best is None or accuracy>best[0]:best=(accuracy,trial,ridge)
    assert best[2]==saved['ridge'];decoder=best[1]
    cal=np.flatnonzero((ds['split']==3)&(candidates.sum(1)>=2));decoder.cutoff=common.quantile([decoder.features(Z[t])[2] for t in cal],.05)
    assert decoder.cutoff==saved['clean_cutoff']
    episodes=[]
    for day,ix in common.blocks(ds,2,history=0,horizon=24):
        universe=np.flatnonzero(candidates[ix].all(0)&candidates[max(ix[0]-1,0)])
        if len(universe)>=2:episodes.append((day,ix,universe))
    expected=pd.read_csv(BASE/'recent_data_results'/region/'available_fault_episode_cases.csv',keep_default_na=False)
    rows=[]
    for episode,(day,ix,universe) in enumerate(common.subset_days(episodes,24)):
        for k in [0,1,2,3]:
            if k>len(universe):continue
            F=tuple(sorted(rng.choice(universe,k,replace=False)))
            for kind in (['clean'] if k==0 else ['same','opposite','random','unequal','drift','stuck','raw_gain','one_channel']):
                for j,t in enumerate(ix):
                    y=faults.families(Z[t],Z[ix[0]-1],ds['mu'],ds['sd'],F,kind,j,rng);pred,alarm,count,norm=decoder.predict(y)
                    rows.append({'episode':episode,'timestamp':str(ds['times'][t]),'kind':kind,'true_count':k,'true_support':','.join(map(str,F)),'predicted_support':','.join(map(str,pred)),'alarm':alarm,'exact':tuple(pred)==F,'estimated_count':len(pred),'norm_score':norm})
    replay=pd.DataFrame(rows);assert len(replay)==len(expected)
    for column in replay:
        a=replay[column];b=expected[column]
        if column=='norm_score':assert np.allclose(a,b,rtol=1e-12,atol=1e-12)
        else:assert a.astype(str).equals(b.astype(str)),(region,column)
    checks.append({'role':region,'status':'passed','episode_rows':len(rows),'ridge':best[2],'calibrated_cutoff':decoder.cutoff})
    print(region,'fault detector and all saved episode outcomes reproduced',len(rows),flush=True)
(OUT/'WHOLE_FAULT_REPLAY.json').write_text(json.dumps({'status':'passed','roles':checks,'scope':'Rebuilds the declared synthetic detector from training/selection/calibration and replays every retained episode; not natural field-fault validation.'},indent=2)+'\n',encoding='utf-8')
