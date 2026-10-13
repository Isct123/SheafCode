                                                                                 
from pathlib import Path
import sys,json,gzip,time,hashlib
import numpy as np
import pandas as pd
from fractions import Fraction as Q
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'extensions/joint_decisions'))
from joint_certified import certify_joint
from run import plain,sha
from joint import joint_envelope
OUT=ROOT/'joint_certification_results';INPUT=ROOT/'math_integration_results'

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'SUMMARY.json').exists():raise RuntimeError('Completed evidence exists; preserve it')
    paths=sorted(INPUT.glob('*_replay.json.gz'));refs={'PM2.5':60.,'PM10':100.,'O3':100.,'O3_ppm':.05,'NO2_ppb':40.}
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};rows=[];clock=time.perf_counter();count=0
    lock={'scope':'Exploratory exact verification after earlier archive/model access; every scheduled previous case retained. No ground truth used by optimizer.','inputs':inputs,'code_sha256':{p.name:sha(p) for p in [Path(__file__),HERE/'joint_certified.py',HERE/'search.py']},'protocol':'Same support, box, residual and injected-fault configuration as math_integration_results; joint max-affine native reference scores. Report conditional truth and every out-of-model error.'}
    (OUT/'PROTOCOL_LOCK.json').write_text(json.dumps(lock,indent=2))
    for path in paths:
        with gzip.open(path,'rt') as f:cases=json.load(f)
        name=cases[0]['case']['cohort'];folder=ROOT/'recent_data_results'/('delhi' if name=='original_india' else 'epa_la') if name in ['original_india','original_usa'] else ROOT/'expanded_validation_results'/name
        ds=np.load(folder/'dataset.npz');inputs[str((folder/'dataset.npz').relative_to(ROOT))]=sha(folder/'dataset.npz')
        channels=list(map(str,ds['channels']));stations=list(map(str,ds['stations']));times=list(map(str,ds['times']));payload=[]
        for case in cases:
            meta=case['case'];v=stations.index(meta['target']);idx=times.index(meta['timestamp']);denom=np.array([refs[c] for c in channels]);H=np.diag(ds['sd'][v]/denom);offsets=ds['mu'][v]/denom
            A=np.array(case['A']);b=np.array(case['b']);blocks=[np.array(B) for B in case['blocks']];lo=np.array(case['lower']);hi=np.array(case['upper']);eps=case['eps'];tic=time.perf_counter()
            exact=certify_joint(A,b,H,offsets,blocks,1,eps,1.,lo,hi);elapsed=time.perf_counter()-tic
            numerical=joint_envelope(A,b,H,offsets,blocks,1,eps,1.,lo,hi)
            if exact['status']=='verified_bounds':
                assert numerical['status']=='bounded'
                assert float(exact['lower'])<=numerical['lower']+1e-6
                assert float(exact['upper'])>=numerical['upper']-1e-6
                if exact['upper_on_minimum'] is not None:assert numerical['lower']<=float(exact['upper_on_minimum'])+1e-6
                if exact['lower_on_maximum'] is not None:assert numerical['upper']>=float(exact['lower_on_maximum'])-1e-6
            else:assert numerical['status']=='inconsistent'
                                                                                                
            rat=lambda x:Q.from_float(float(x));truth=max(sum((rat(a)*rat(z) for a,z in zip(row,ds['Z'][idx,v])),rat(c)) for row,c in zip(H,offsets));positive=truth>1
            valid=meta['exact_truth_assumptions']
            if valid:assert exact['status']=='verified_bounds' and exact['lower']<=truth<=exact['upper']
            issued=exact['decision']!='abstain';wrong=issued and ((exact['decision']=='exceedance')!=positive)
            if valid:assert not wrong
            lower_gap=None if exact.get('upper_on_minimum') is None else float(exact['upper_on_minimum']-exact['lower'])
            upper_gap=None if exact.get('lower_on_maximum') is None else float(exact['upper']-exact['lower_on_maximum'])
            row={'cohort':name,'timestamp':meta['timestamp'],'target':meta['target'],'scenario':meta['scenario'],'status':exact['status'],'decision':exact['decision'],'numerical_decision':numerical['decision'],'positive_truth':bool(positive),'exact_truth_assumptions':bool(valid),'issued':issued,'wrong':bool(wrong),'lower':None if exact.get('lower') is None else str(exact['lower']),'upper':None if exact.get('upper') is None else str(exact['upper']),'lower_endpoint_gap':lower_gap,'upper_endpoint_gap':upper_gap,'seconds':elapsed}
            rows.append(row);payload.append({'metadata':row,'A':A,'b':b,'H':H,'offsets':offsets,'blocks':blocks,'eps':eps,'lower':lo,'upper':hi,'truth_state':ds['Z'][idx,v],'certification':exact});count+=1
        with gzip.open(OUT/(name+'_proofs.json.gz'),'wt',encoding='utf-8') as f:json.dump(plain(payload),f)
        print(name,len(cases),'joint cases',round(time.perf_counter()-clock,1),'seconds',flush=True)
                                                                                       
    A=np.array([[1.,1.]]);b=np.array([-2.]);H=np.eye(2)/.75
    controls=[]
    for epsilon in [0.,.01]:
        result=certify_joint(A,b,H,[0.,0.],[],0,epsilon,1.,[0.,0.],[2.,2.]);assert result['decision']=='exceedance'
        controls.append({'name':'joint_only_positive','eps':epsilon,'A':A,'b':b,'H':H,'offsets':[0.,0.],'blocks':[],'lower':[0.,0.],'upper':[2.,2.],'certification':result})
    (OUT/'CONTROLS.json').write_text(json.dumps(plain(controls),indent=2))
    f=pd.DataFrame(rows);f.to_csv(OUT/'CASES.csv',index=False);summary={'status':'completed','cases':count,'roles':len(paths),'conditional_cases':int(f.exact_truth_assumptions.sum()),'issued':int(f.issued.sum()),'wrong':int(f.wrong.sum()),'conditional_wrong':int(f[f.exact_truth_assumptions].wrong.sum()),'outside_model_wrong':int(f[~f.exact_truth_assumptions].wrong.sum()),'positive_truth_cases':int(f.positive_truth.sum()),'decisions':f.decision.value_counts().to_dict(),'status_counts':f.status.value_counts().to_dict(),'numerical_decision_changes':int((f.decision!=f.numerical_decision).sum()),'max_lower_endpoint_gap':float(f.lower_endpoint_gap.max()),'max_upper_endpoint_gap':float(f.upper_endpoint_gap.max()),'elapsed_seconds':time.perf_counter()-clock,'scope':'Exact stored-coefficient certificates of joint nominal decisions; exploratory artificial-outage/injected-fault archives, not physical model validity.'}
    (OUT/'PROTOCOL_LOCK.json').write_text(json.dumps(lock,indent=2));(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
