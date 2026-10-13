                                                                                   
from pathlib import Path
import sys,subprocess,json,os,concurrent.futures,time,hashlib
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2];OUT=ROOT/'Research_Release/expanded_validation_results'
STAGES=[('fit','models.pkl'),('evaluate','evaluation_status.json'),('faults','fault_status.json'),('fault_partial','available_fault_status.json'),('uncertainty','uncertainty_status.json'),('mathematics','mathematics_status.json'),('extensions','extensions_status.json'),('environment','environment_status.json'),('additional','additional_status.json'),('operator_support','operator_support_status.json'),('final_analysis','final_analysis_status.json')]
def cohort(region):
    folder=OUT/region;logs=folder/'logs';logs.mkdir(exist_ok=True)
    import numpy as np
    ds=np.load(folder/'dataset.npz',allow_pickle=False)
    if not np.isfinite(ds['X'][ds['split']==2]).any():
        if (folder/'models.pkl').exists():
            for task,marker in [('mathematics','mathematics_status.json'),('operator_support','operator_support_status.json')]:
                if (folder/marker).exists():continue
                with (logs/(task+'.log')).open('a',encoding='utf-8') as f:
                    for attempt in range(3):
                        r=subprocess.run([sys.executable,str(HERE/'stage.py'),region,task],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                        if r.returncode!=3221227274:break
                if r.returncode:raise RuntimeError('Applicable mathematical check failed: '+region+'/'+task)
        value={'cohort':region,'status':'unavailable_data','reason':'No recorded pollutant observations in the fixed test period; no empirical success rate can be estimated','test_observed_cells':0,'selection_observed_cells':int(np.isfinite(ds['X'][ds['split']==1]).sum()),'model_fitted':(folder/'models.pkl').exists(),'errors':[]}
        (folder/'run_status.json').write_text(json.dumps(value,indent=2));print(region,'unavailable: no test observations',flush=True);return value
    errors=[]
    for name,marker in STAGES:
        if (folder/marker).exists():continue
        status={'cohort':region,'stage':name,'status':'running','pid':os.getpid(),'updated':time.strftime('%Y-%m-%d %H:%M:%S')}
        (folder/'run_status.json').write_text(json.dumps(status,indent=2))
        print(region,name,flush=True)
        with (logs/(name+'.log')).open('a',encoding='utf-8') as f:
            for attempt in range(3):
                result=subprocess.run([sys.executable,str(HERE/'stage.py'),region,name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                if result.returncode!=3221227274:break
                f.write('\nNative Windows threadpool-handle exception; retry '+str(attempt+1)+' of 2, preserving prior outputs.\n');f.flush()
        if result.returncode:
            errors.append({'stage':name,'exit_code':result.returncode,'log':str(logs/(name+'.log'))})
            if name in ['fit','evaluate']:break
        elif name=='final_analysis':(folder/marker).write_text(json.dumps({'status':'completed'}))
    status={'cohort':region,'status':'failed' if errors else 'completed','errors':errors}
    (folder/'run_status.json').write_text(json.dumps(status,indent=2));return status
def main():
    OUT.mkdir(exist_ok=True)
    if not (OUT/'cohorts.json').exists():(OUT/'cohorts.json').write_text(json.dumps({'eligible':[],'secondary':[]}))
    names=sys.argv[1:] or sorted(p.parent.name for p in OUT.glob('*/dataset.npz'))
    if not names:raise RuntimeError('No acquired cohorts yet')
    workers=max(1,min(3,int(os.environ.get('EXPANDED_WORKERS','3'))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(cohort,names))
    (OUT/'analysis_status.json').write_text(json.dumps({'status':'failed' if any(r['errors'] for r in results) else 'completed','cohorts':results},indent=2))
    if any(r['errors'] for r in results):raise RuntimeError('Some analysis stages failed; see per-cohort logs')
if __name__=='__main__':main()
