                                                                    
from pathlib import Path
import csv,hashlib,json,sys,time
import numpy as np
from fractions import Fraction as Q
sys.path.insert(0,str(Path(__file__).parent))
from decision_repair import conflict_edges,master,plan,infer

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Research_Release/decision_repair_results'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    with (ROOT/'Research_Release/expanded_validation_results/COHORT_RESULTS.csv').open(newline='',encoding='utf-8-sig') as f:
        cohorts=[(r['cohort'],ROOT/'Research_Release/expanded_validation_results'/r['cohort']/'dataset.npz') for r in csv.DictReader(f)]
    cohorts += [('original_india',ROOT/'Research_Release/recent_data_results/delhi/dataset.npz'),
                ('original_usa',ROOT/'Research_Release/recent_data_results/epa_la/dataset.npz')]
    rows=[];banks={};traces=[];input_hashes={};checks=0;start=time.time()
    for name,path in cohorts:
        with np.load(path,allow_pickle=False) as d:
            channels=[str(x) for x in d['channels']];pm=channels.index('PM2.5');X=d['X'][:,:,pm]
            ii=np.flatnonzero((d['split']==2)&np.isfinite(X).all(1))
            selected=ii[np.unique(np.linspace(0,len(ii)-1,min(24,len(ii))).astype(int))] if len(ii) else np.array([],int)
            vectors=X[selected];n=X.shape[1];times=[str(x) for x in d['times'][selected]]
            banks[name]={'stations':[str(x) for x in d['stations']],'timestamps':times,'pm25_vectors':vectors.tolist(),
                         'complete_test_hours':len(ii),'scope':'Offline archived test scenarios, not a training-only prospective hypothesis model'}
        input_hashes[str(path.relative_to(ROOT))]=sha(path)
        if n>10:raise ValueError('Unexpected cohort too large for declared exhaustive solver')
        for threshold in [15.,35.,60.]:
            for eta in [1.,3.,5.]:
                maximum=np.max(vectors,axis=1) if len(vectors) else np.array([])
                keep=(maximum<=threshold-eta)|(maximum>threshold+eta)
                states=vectors[keep].tolist();labels=(maximum[keep]>threshold+eta).astype(int).tolist()
                for faults in [0,1]:
                    row={'cohort':name,'stations':n,'threshold':threshold,'eta':eta,'query_faults':faults,
                         'sampled_scenarios':len(vectors),'margin_excluded':int((~keep).sum()),'scenarios':len(states),
                         'safe':labels.count(0),'exceedance':labels.count(1),'decision_status':'trivial_or_empty',
                         'decision_queries':'','pattern_status':'not_evaluated','pattern_queries':'','refinements':'','checks':0}
                    if len(set(labels))!=2:
                        rows.append(row);continue
                    edges=conflict_edges(states,labels,eta);r=plan(n,edges,faults=faults)
                    exact=master(n,[1.]*n,[e for e,_ in edges],2*faults+1)
                    assert (exact is None)==(r['status']=='infeasible');checks+=1
                    row['decision_status']=r['status'];row['refinements']=r['refinements']
                    if exact is not None:
                        assert r['cost']==exact[1];checks+=1
                        row['decision_queries']=int(r['cost'])
                        lows=[x['lower'] for x in r['trace']];assert lows==sorted(lows);checks+=1
                        for state,label in zip(states,labels):
                            for sign in [-1.,1.]:
                                y={v:Q.from_float(state[v])+Q(int(sign))*Q.from_float(eta) for v in r['stations']}
                                assert infer(states,labels,eta,y,faults)==label;checks+=1
                                if faults:
                                    for bad in r['stations']:
                                        corrupt=y.copy();corrupt[bad]=1e6
                                        assert infer(states,labels,eta,corrupt,faults)==label;checks+=1
                                                                                                
                    all_edges=conflict_edges(states,list(range(len(states))),eta)
                    full=master(n,[1.]*n,[e for e,_ in all_edges],2*faults+1)
                    row['pattern_status']='infeasible' if full is None else 'optimal'
                    if full is not None:
                        row['pattern_queries']=int(full[1]);assert exact is not None and exact[1]<=full[1];checks+=1
                    traces.append({'cohort':name,'threshold':threshold,'eta':eta,'faults':faults,'plan':r,
                                   'decision_edges':[{'mask':e,'pair':list(pair)} for e,pair in edges]})
                    rows.append(row)
        print(name,'complete test hours',len(ii),'sampled',len(vectors),'configs',len(rows),flush=True)
    summary={}
    for f in [0,1]:
        subset=[r for r in rows if r['query_faults']==f]
        active=[r for r in subset if r['safe'] and r['exceedance']]
        feasible=[r for r in active if r['decision_status']=='optimal']
        both=[r for r in active if r['decision_status']=='optimal' and r['pattern_status']=='optimal']
        saved=sum(r['pattern_queries']-r['decision_queries'] for r in both)
        total=sum(r['pattern_queries'] for r in both)
        summary[str(f)]={'configurations':len(subset),'nontrivial_banks':len(active),'decision_feasible':len(feasible),
                        'decision_infeasible':len(active)-len(feasible),'both_feasible':len(both),
                        'decision_strictly_cheaper':sum(r['decision_queries']<r['pattern_queries'] for r in both),
                        'decision_possible_pattern_impossible':sum(r['decision_status']=='optimal' and r['pattern_status']=='infeasible' for r in active),
                        'decision_query_sum_common':sum(r['decision_queries'] for r in both),'pattern_query_sum_common':total,
                        'query_saving_percent_common':100*saved/total if total else None}
    fields=list(rows[0])
    with (OUT/'REPAIR_CONFIGURATIONS.csv').open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=fields);w.writeheader();w.writerows(rows)
    (OUT/'SCENARIO_BANKS.json').write_text(json.dumps(banks,indent=2),encoding='utf-8')
    (OUT/'SEARCH_TRACES.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    report={'status':'completed_and_checked','cohorts':len(cohorts),'configurations':len(rows),'checks':checks,
            'summaries_by_query_fault_budget':summary,'input_sha256':input_hashes,'protocol_sha256':sha(Path(__file__).with_name('PROTOCOL.md')),
            'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'seconds':time.time()-start,
            'scope':'Offline finite archive-scenario combinatorial benchmark; conditional on bank membership and margins. Not prospective prediction, field repair or independent configurations.'}
    (OUT/'SUMMARY.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))

if __name__=='__main__':main()
