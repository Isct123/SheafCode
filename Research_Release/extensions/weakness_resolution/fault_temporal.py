                                                                          
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'Research_Release/expanded_validation_results'
OUT=ROOT/'Research_Release/weakness_resolution_results/fault_temporal'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
common.OUT=SOURCE
import numpy as np,pandas as pd

def main():
    OUT.mkdir(parents=True,exist_ok=True);summaries=[];statuses=[]
    for folder in sorted(p.parent for p in SOURCE.glob('*/available_fault_episode_cases.csv')):
        name=folder.name;dest=OUT/name;dest.mkdir(exist_ok=True)
        try:frame=pd.read_csv(folder/'available_fault_episode_cases.csv')
        except pd.errors.EmptyDataError:frame=pd.DataFrame()
        if frame.empty:
            statuses.append({'cohort':name,'status':'unavailable: no complete candidate episodes'});continue
        ds=common.load(name);model=common.models(name);D=model['sheaf'].D;Z=ds['Z'];n,p=Z.shape[1:]
        full=np.isfinite(Z).all((1,2));train=Z[(ds['split']==0)&full]
        if len(train)<2:raise AssertionError('Original syndrome whitening lacks training support')
        s=train.reshape(len(train),-1)@D.T;mean=s.mean(0);C=np.cov(s,rowvar=False)
        eig,V=np.linalg.eigh(C+.01*np.eye(len(D)));W=(V/np.sqrt(np.maximum(eig,1e-10)))@V.T
        whitened=(np.nan_to_num(Z).reshape(len(Z),-1)@D.T-mean)@W.T
        clean_score=np.sum(whitened**2,axis=1)
        eligible=np.isfinite(Z).all(2).sum(1)>=2
        cal=np.flatnonzero((ds['split']==3)&eligible)
        qpoint=common.quantile(clean_score[cal],.05)
        stored=json.loads((folder/'available_fault_status.json').read_text())
        if not np.allclose(qpoint,stored['clean_cutoff'],equal_nan=True):raise AssertionError('Original clean cutoff replay mismatch')
        rolling=pd.Series(np.where(eligible,clean_score,np.nan)).rolling(4,min_periods=4).mean().to_numpy()
        calvalid=np.flatnonzero((ds['split']==3)&np.isfinite(rolling))
        calvalid=calvalid[calvalid>=3]
        calvalid=calvalid[np.all(np.stack([ds['split'][calvalid-j]==3 for j in range(4)]),axis=0)]
        qrolling=common.quantile(rolling[calvalid],.05)
        time_index={str(t):i for i,t in enumerate(ds['times'])}
        rows=[]
        for (episode,kind,count),g in frame.groupby(['episode','kind','true_count']):
            g=g.sort_values('timestamp');first=time_index[str(pd.Timestamp(g.iloc[0].timestamp))]
            prior=[float(clean_score[t]) if eligible[t] else np.nan for t in range(max(0,first-3),first)]
            history=prior.copy()
            for r in g.itertuples():
                t=time_index[str(pd.Timestamp(r.timestamp))]
                if kind=='clean' and not np.isclose(clean_score[t],r.norm_score,rtol=1e-7,atol=1e-7):raise AssertionError('Clean test syndrome replay mismatch')
                history.append(float(r.norm_score));last=history[-4:]
                if len(last)!=4 or not np.isfinite(last).all():continue
                                                                              
                                                                            
                temporal=float(np.mean(last))
                for method,alarm in [('original_count_gate',bool(r.alarm)),('point_norm',r.norm_score>qpoint),('four_hour_norm',temporal>qrolling)]:
                    rows.append({'day':r.day,'episode':episode,'timestamp':r.timestamp,'kind':kind,'true_count':count,
                                 'method':method,'alarm':bool(alarm),'point_score':r.norm_score,'temporal_score':temporal})
        f=pd.DataFrame(rows);f.to_csv(dest/'cases.csv',index=False)
        for keys,g in f.groupby(['method','kind','true_count']):
            method,kind,count=keys
            summaries.append({'cohort':name,'method':method,'kind':kind,'true_count':int(count),
                              'hours':len(g),'episodes':g.episode.nunique(),'alarm_rate':g.alarm.mean()})
        statuses.append({'cohort':name,'status':'completed','calibration_point_hours':len(cal),
                         'calibration_temporal_hours':len(calvalid),'point_cutoff':qpoint,'temporal_cutoff':qrolling,
                         'clean_scores_replayed':True,'eligible_test_rows_per_method':len(f)//3,
                         'source_sha256':common.sha(__file__),'protocol_sha256':common.sha(Path(__file__).parent/'FAULT_TEMPORAL_PROTOCOL.md'),
                         'scope':'Four-hour causal detection only, not fault localization; synthetic scenarios; sparse calibration yields full abstention.'})
        print('Temporal fault detector',name,'matched rows per method',len(f)//3,flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'SUMMARY.csv',index=False)
    (OUT/'status.json').write_text(json.dumps(common.clean({'status':'completed','cohorts':statuses}),indent=2))

if __name__=='__main__':main()
