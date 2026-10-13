                                                                            

                                                                              
                                                                           
   
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'extensions/functional'))
from certificate import functional_interval
from sheafcode.cohort import select_cohort
from sheafcode.reliable import CachedSpatial

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=int,default=480);ap.add_argument('--output',default='followup_results');args=ap.parse_args()
    if args.hours<2:raise ValueError('At least two sampled hours required')
    out=ROOT/args.output;out.mkdir(exist_ok=True)
    records=[];alarms=[];config={};rng=np.random.default_rng(601)
    for region in ['delhi2025','bengaluru_raw']:
        folder=ROOT/'results_expanded'/region
        m=json.loads((folder/'manifest.json').read_text())
        if m['status']!='complete':raise RuntimeError('Expanded core incomplete: '+region)
        for name,sha in m['code_sha256'].items():
            if digest(ROOT/name)!=sha:raise RuntimeError('Frozen source changed: '+name)
        ds=select_cohort(region);choice=m['choices']['sheaf'];ds.set_transform(choice['transform']);Z=ds.Z()
        model=np.load(folder/'model.npz');D=model['delta'];decoder=CachedSpatial(model['L']+model['lam']*model['P'],ds.n,ds.p)
        q=ds.pollutants.index('PM2.5');config[region]={'model_sha256':digest(folder/'model.npz'),'manifest_sha256':digest(folder/'manifest.json')}
        def raw(x,t,v):return float(ds.raw_from_Z(np.asarray(x).reshape(1,1,ds.p),np.array([t]),[v])[0,0,q])
        for rate in [0.,.2,.5]:
            groups={}
            for part in [1,2]:
                candidates=np.flatnonzero(ds.split==part)
                sample=candidates[np.unique(np.linspace(0,len(candidates)-1,min(args.hours,len(candidates))).astype(int))]
                cases=[]
                for t in sample:
                    for v in range(ds.n):
                        if not np.isfinite(Z[t,v,q]):continue
                        hidden=Z[t].copy();hidden[v]=np.nan
                        mask=rng.random(hidden.shape)<rate;hidden[mask]=np.nan
                        U=np.flatnonzero(~np.isfinite(hidden.reshape(-1)));O=np.flatnonzero(np.isfinite(hidden.reshape(-1)))
                                                                                
                                                                              
                                                                                   
                        A=D[:,U];b=D[:,O]@hidden.reshape(-1)[O]
                        h=np.zeros(len(U));h[np.flatnonzero(U==v*ds.p+q)[0]]=1
                        pred,gain=decoder.predict(hidden[None],[v]);point=raw(pred[0,0],t,v)
                        other_raw=ds.raw_from_Z(hidden[None],np.array([t]),list(range(ds.n)))[0,:,q]
                        fallback=float(np.nanmean(other_raw)) if np.isfinite(other_raw).any() else None
                        truth=float(ds.X[t,v,q]);score=abs(point-truth)
                        cases.append({'t':int(t),'station_index':v,'A':A,'b':b,'h':h,'point':point,'truth':truth,'error':score,'gain':float(gain[0,0,q]),
                                      'city_mean':fallback,'full_residual':float(np.linalg.norm(D@Z[t].reshape(-1))) if np.isfinite(Z[t]).all() else None})
                groups[part]=cases
            val=groups[1]
            if not val:raise RuntimeError('No validation cases')
            k=int(np.ceil((len(val)+1)*.9))
            def quant(values):return float(np.sort(values)[k-1]) if k<=len(values) else float('inf')
            fixed_width=quant([x['error'] for x in val]);scaled_width=quant([x['error']/x['gain'] for x in val])
            complete=[x['full_residual'] for x in val if x['full_residual'] is not None]
            eps=float(np.quantile(complete,.95,method='higher')) if complete else None
                                                                           
            negatives=[x['point'] for x in val if x['truth']<=60]
            cutoff=float(np.quantile(negatives,.95,method='higher')) if negatives else float('inf')
            city_neg=[x['city_mean'] for x in val if x['truth']<=60 and x['city_mean'] is not None]
            city_cut=float(np.quantile(city_neg,.95,method='higher')) if city_neg else float('inf')
            for x in groups[2]:
                bounded=functional_interval(x['A'],x['b'],x['h'],eps) if eps is not None else {'status':'uncalibrated'}
                for method,width in [('fixed_no_feedback',fixed_width),('availability_scaled_no_feedback',scaled_width*x['gain'])]:
                    lo=x['point']-width;hi=x['point']+width
                    records.append({'region':region,'timestamp':str(ds.times[x['t']]),'station':ds.stations[x['station_index']],
                        'extra_cell_missing_rate':rate,'method':method,'truth':x['truth'],'prediction':x['point'],'lower':lo,'upper':hi,
                        'covered':lo<=x['truth']<=hi,'finite_width':float(2*width),'gain':x['gain'],'functional_status':bounded['status'],
                        'alert_decision':'above' if lo>60 else ('below' if hi<=60 else 'abstain')})
                if x['city_mean'] is not None:
                    for method,point,cut in [('sheaf',x['point'],cutoff),('city_mean',x['city_mean'],city_cut)]:
                        alarms.append({'region':region,'timestamp':str(ds.times[x['t']]),'station':ds.stations[x['station_index']],
                            'method':method,'extra_cell_missing_rate':rate,'truth_positive':x['truth']>60,'alert':point>cut,'validation_cutoff':cut,'prediction':point})
            config[region][str(rate)]={'validation_cases':len(val),'test_cases':len(groups[2]),'fixed_halfwidth':fixed_width,
                'scaled_score_quantile':scaled_width,'validation_alert_cutoff':cutoff,'residual_budget':eps}
            print(region,'missing rate',rate,'complete',flush=True)
    f=pd.DataFrame(records);f.to_csv(out/'no_feedback_intervals.csv',index=False)
    a=pd.DataFrame(alarms);a.to_csv(out/'validation_fixed_alerts.csv',index=False)
    summary=[]
    for keys,g in f.groupby(['region','extra_cell_missing_rate','method']):
        decided=g[g.alert_decision!='abstain']
        correct=((decided.alert_decision=='above')==(decided.truth>60))
        summary.append({'region':keys[0],'missing_rate':keys[1],'method':keys[2],'cases':len(g),'coverage':float(g.covered.mean()),
            'mean_width':float(g.finite_width.mean()),'decided_fraction':len(decided)/len(g),'decision_accuracy':float(correct.mean()) if len(decided) else None})
    pd.DataFrame(summary).to_csv(out/'interval_summary.csv',index=False)
    alert_summary=[]
    for keys,g in a.groupby(['region','extra_cell_missing_rate','method']):
        pos=g[g.truth_positive];neg=g[~g.truth_positive]
        alert_summary.append({'region':keys[0],'missing_rate':keys[1],'method':keys[2],'positive_cases':len(pos),'negative_cases':len(neg),
            'sensitivity':float(pos.alert.mean()) if len(pos) else None,'false_positive_rate':float(neg.alert.mean()) if len(neg) else None})
    pd.DataFrame(alert_summary).to_csv(out/'alert_summary.csv',index=False)
    lines=['# Follow-up internal validation','','No test target is revealed to update the predictions, widths or alert cutoffs. All test outcomes are used only for scoring. The data were previously examined; this is not independent validation.','',
        '## No-feedback interval evaluation','',pd.DataFrame(summary).to_string(index=False),'',
        '## Validation-selected alert cutoffs','',pd.DataFrame(alert_summary).to_string(index=False),'',
        'Both methods use the same eligible test cases. Their cutoffs target 5% false positives on validation negatives; their actual test false-positive rates can differ. These are illustrative hourly threshold alerts, not official AQI/GRAP decisions.','',
        'The missingness scenarios use added independent cell masks on top of natural missingness. Dependence-aware confirmatory inference and field outage validation remain necessary. No emissions, exposure or health improvement is established.']
    (out/'FOLLOWUP_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'config':config,'source_sha256':{p.name:digest(p) for p in HERE.glob('*.py')},
        'limitations':['Internal previously examined archives, not independent validation','Hourly concentration threshold is illustrative, not official daily AQI or GRAP',
        'Validation fixed 5% false-alarm target need not equal test false-alarm rate','Cell errors and stations are dependent; descriptive coverage is not a shift guarantee',
        'No hidden-target test feedback; empirical widths use validation labels only','Missingness masks are artificial; no field intervention or emissions reduction established',
        'Residual budget calibrated on complete validation vectors; functional status is diagnostic, not guaranteed outage coverage']},indent=2))

if __name__=='__main__':main()
