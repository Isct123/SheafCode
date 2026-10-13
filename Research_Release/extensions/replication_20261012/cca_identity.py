                                                                                
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import pickle
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'Research_Release/replication_20261012_results'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
from evaluate import score
common.OUT=OUT

def identity_spatial(model,edges):
    result=copy.deepcopy(model)
    p,n=result.p,result.n
    rows=[]
    for u,v in edges:
        block=np.zeros((p,n*p))
        block[:,u*p:(u+1)*p]=np.eye(p)
        block[:,v*p:(v+1)*p]=-np.eye(p)
        rows.append(block)
    result.D=np.vstack(rows)
    result.S=result.D.T@result.D+result.lam*result.P
    result.precisions={}
    return result

def cohort(name):
    folder=OUT/name
    destination=folder/'cca_identity'
    destination.mkdir(exist_ok=True)
    identity={'protocol_sha256':common.sha(HERE/'CCA_PROTOCOL.md'),'adapter_sha256':common.sha(Path(__file__)),
        'reference_sha256':common.sha(ROOT/'Research_Release/extensions/online_validation/reference.py'),
        'dataset_sha256':common.sha(folder/'dataset.npz'),'cca_model_sha256':common.sha(folder/'models.pkl'),
        'reference_cases_sha256':common.sha(folder/'outage_cases.csv')}
    lock=destination/'identity.json'
    if lock.exists() and json.loads(lock.read_text())!=identity:
        raise RuntimeError('CCA ablation inputs/source changed: '+name)
    lock.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    if (destination/'status.json').exists():
        return json.loads((destination/'status.json').read_text())
    ds=common.load(name)
    models=common.models(name)
    Z=ds['Z'];p=Z.shape[2]
    fixed=identity_spatial(models['sheaf'],models['edges'])
    validation=[]
    for day,ix in common.subset_days(common.blocks(ds,1),12):
        for v in range(Z.shape[1]):
            truth=Z[ix[24:],v];good=np.isfinite(truth)
            if good.sum()>=12*p:validation.append((ix,v,truth,good))
    choices=[];best=None
    for lam in [.1,1.]:
        for scale in [.5,2.]:
            candidate=identity_spatial(common.QuadraticDecoder(Z[ds['split']==0],models['edges'],lam,scale),models['edges'])
            errors=[]
            for ix,v,truth,good in validation:
                errors.append(np.mean((candidate.causal(Z[ix],v)-truth)[good]**2))
            choice={'lambda':lam,'temporal_scale':scale,'selection_mse':float(np.mean(errors)),'selection_cases':len(validation)}
            choices.append(choice)
            if best is None or choice['selection_mse']<best[0]['selection_mse']:best=(choice,candidate)
    pd.DataFrame(choices).to_csv(destination/'selection_grid.csv',index=False)
    (destination/'selection_choice.json').write_text(json.dumps(best[0],indent=2),encoding='utf-8')
    tuned=best[1]
    with (destination/'models.pkl').open('wb') as stream:
        pickle.dump({'identity_spatial_fixed':fixed,'identity_spatial_tuned':tuned},stream)
    rows=[]
    planned=common.blocks(ds,2)
    primary={day for day,_ in planned[::3]}
    for day,ix in planned:
        for v,station in enumerate(ds['stations']):
            truth=Z[ix[24:],v]
            if np.isfinite(truth).sum()<12*p:continue
            for method,model in [('identity_spatial_fixed',fixed),('identity_spatial_tuned',tuned)]:
                prediction=model.causal(Z[ix],v)
                rows.append({'day':day,'station':str(station),'method':method,'primary':day in primary,**score(prediction,truth,ds['sd'][v])})
    frame=pd.DataFrame(rows)
    frame.to_csv(destination/'cases.csv',index=False)
    reference=pd.read_csv(folder/'outage_cases.csv',dtype={'station':str})
    keys=lambda f:set(map(tuple,f[['day','station']].to_numpy()))
    for method in frame.method.unique():assert keys(frame[frame.method==method])==keys(reference[reference.method=='sheaf_causal'])
    joined=pd.concat([frame,reference],ignore_index=True)
    summaries={scope:{method:{str(days):{metric:common.paired(ff,'sheaf_causal',method,value=metric,days=days) for metric in ['mse','pm25_raw_mse']} for days in [7,14]} for method in frame.method.unique()} for scope,ff in [('primary_every_third_day',joined[joined.primary]),('all_eligible_test_days',joined)]}
    result={'status':'completed','cohort':name,'identity':identity,'tuned_identity_choice':best[0],'summary':summaries,
        'temporal_cca_retained':True,'all_outcomes_retained':True,'model_sha256':common.sha(destination/'models.pkl'),
        'cases_sha256':common.sha(destination/'cases.csv')}
    (destination/'status.json').write_text(json.dumps(common.clean(result),indent=2),encoding='utf-8')
    print(name,'CCA gain vs fixed/tuned identity:',summaries['primary_every_third_day']['identity_spatial_fixed']['7']['mse']['gain_percent'],summaries['primary_every_third_day']['identity_spatial_tuned']['7']['mse']['gain_percent'],flush=True)
    return result

def main():
    names=sys.argv[1:] or json.loads((OUT/'cohorts.json').read_text())['eligible']
    names=[name for name in names if (OUT/name/'evaluation_status.json').exists()]
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(cohort,names))
    (OUT/'cca_identity_status.json').write_text(json.dumps({'status':'completed','cohorts':[{'cohort':r['cohort'],'status':r['status']} for r in results]},indent=2),encoding='utf-8')

if __name__=='__main__':main()
