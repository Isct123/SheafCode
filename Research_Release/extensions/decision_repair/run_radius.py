import csv,json,sys,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from fractions import Fraction as Q
from radius_repair import radius,frontier,ambiguity_witness
from decision_repair import conflict_edges,plan

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'Research_Release/decision_repair_results'
def encode(x):
    if isinstance(x,Q):return {'exact':str(x),'value':float(x)}
    if isinstance(x,float) and abs(x)==float('inf'):return str(x)
    raise TypeError(type(x).__name__)
def main():
    start=time.time();banks=json.loads((OUT/'SCENARIO_BANKS.json').read_text())
    original=list(csv.DictReader((OUT/'REPAIR_CONFIGURATIONS.csv').open(newline='',encoding='utf-8')))
    rows=[];all_frontiers=[];witnesses=[];checks=0
    for index,row in enumerate(original):
        threshold=float(row['threshold']);eta=float(row['eta']);f=int(row['query_faults']);states=[];labels=[]
        for x in banks[row['cohort']]['pm25_vectors']:
            m=max(x)
            if m<=threshold-eta or m>threshold+eta:
                states.append(x);labels.append(int(m>threshold+eta))
        out={k:row[k] for k in ['cohort','threshold','eta','query_faults','scenarios','safe','exceedance']}
        out.update(status='trivial_or_empty',queries='',baseline_radius='',same_cost_radius='',full_query_radius='',semantic_margin='')
        if len(set(labels))==2:
            n=len(states[0]);p=plan(n,conflict_edges(states,labels,eta),faults=f)
            margins=min(abs(Q.from_float(max(x))-Q.from_float(threshold)) for x in states)
            fr=frontier(states,labels,eta,faults=f)
            for entry in fr:
                if entry['radius']>=0 and entry['radius']!=float('inf'):
                    w=ambiguity_witness(states,entry['witness'],eta,entry['mask'],entry['radius'],faults=f)
                    checks+=1
                    witnesses.append({'configuration':index,'budget':entry['budget'],'radius':entry['radius'],'pair':entry['witness'],'counterexample':w})
            all_frontiers.append({'configuration':index,'cohort':row['cohort'],'faults':f,'semantic_margin':margins,'frontier':fr})
            assert all(a['radius']<=b['radius'] for a,b in zip(fr,fr[1:]));checks+=1
            assert (fr[-1]['radius']<=0)==(p['status']=='infeasible');checks+=1
            if p['status']=='optimal':
                initial=radius(states,labels,eta,p['mask'],f)['radius'];budget=p['cost'];best=next(x for x in fr if x['budget']==budget)
                assert best['radius']>=initial>0;checks+=1
                out.update(status='certified_inside_radius',queries=budget,baseline_radius=float(min(initial,margins)),
                           same_cost_radius=float(min(best['radius'],margins)),full_query_radius=float(min(fr[-1]['radius'],margins)),
                           semantic_margin=float(margins))
            else:out['status']='infeasible'
        rows.append(out)
    summaries={}
    for f in ['0','1']:
        group=[r for r in rows if r['query_faults']==f];good=[r for r in group if r['status']=='certified_inside_radius']
        summaries[f]={'configurations':len(group),'feasible':len(good),'infeasible':sum(r['status']=='infeasible' for r in group),
                      'strict_same_cost_improvements':sum(r['same_cost_radius']>r['baseline_radius'] for r in good),
                      'strict_larger_budget_improvements':sum(r['full_query_radius']>r['same_cost_radius'] for r in good),
                      'semantically_capped_same_cost':sum(r['same_cost_radius']==r['semantic_margin'] for r in good)}
    with (OUT/'RADIUS_CONFIGURATIONS.csv').open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'RADIUS_FRONTIERS.json').write_text(json.dumps(all_frontiers,default=encode),encoding='utf-8')
    (OUT/'RADIUS_BOUNDARY_WITNESSES.json').write_text(json.dumps(witnesses,default=encode),encoding='utf-8')
    report={'status':'completed','checks':checks,'summaries':summaries,'boundary_witnesses':len(witnesses),'seconds':time.time()-start,
            'scope':'Exact labelled coordinate-box transfer and budget frontier; conditional on box membership and threshold stability, not future coverage.',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('radius_repair.py'),Path(__file__).with_name('RADIUS_PROTOCOL.md')]},
            'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'SCENARIO_BANKS.json',OUT/'REPAIR_CONFIGURATIONS.csv']}}
    (OUT/'RADIUS_SUMMARY.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
