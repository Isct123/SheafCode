                                                                            
from pathlib import Path
from itertools import permutations
import sys,csv,json,hashlib,time,math
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from heldout import simulated_readings,greedy
from decision_repair import master
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'Research_Release/operational_validation_results'
OUT=ROOT/'Research_Release/query_policy_results'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cost(high,order,faults):
    ordered=high[:,order];counts=np.cumsum(ordered,axis=1)>faults
    return np.where(counts.any(1),counts.argmax(1)+1,len(order))

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'SUMMARY.json').exists():raise RuntimeError('Completed outputs exist; preserving them')
    banks=json.loads((OLD/'FROZEN_BANKS.json').read_text());choices={}
    for name,bank in banks.items():
        folder=ROOT/'Research_Release/expanded_validation_results'/name
        if name in ['original_india','original_usa']:folder=ROOT/'Research_Release/recent_data_results'/('delhi' if name=='original_india' else 'epa_la')
        with np.load(folder/'dataset.npz',allow_pickle=False) as d:
            X=d['X'][:,:,list(map(str,d['channels'])).index('PM2.5')];val=X[(d['split']==1)&np.isfinite(X).all(1)]
            split=d['split'];times=d['times']
            if not len(val):choices[name]={'status':'no_complete_validation'};continue
            if not times[split==1].max()<times[split==2].min():raise RuntimeError('Chronology violation')
            n=len(bank['stations']);selected={}
            for faults in [0,1]:
                high=[]
                for x in val:
                    trials=[(s,None,'none') for s in [-1,0,1]] if faults==0 else [(s,v,c) for s in [-1,1] for v in range(n) for c in ['low','high']]
                    high.extend(simulated_readings(x,s,v,c)>36 for s,v,c in trials)
                high=np.array(high,dtype=bool)
                best=min((int(cost(high,list(order),faults).sum()),order) for order in permutations(range(n)))
                selected[str(faults)]={'order':list(best[1]),'validation_query_sum':best[0],'validation_trials':len(high)}
            choices[name]={'status':'frozen','dataset_sha256':sha(folder/'dataset.npz'),'validation_hours':len(val),'plans':selected}
                                                                      
    lock={'choices':choices,'protocol_sha256':sha(Path(__file__).with_name('QUERY_ORDER_PROTOCOL.md')),'code_sha256':sha(Path(__file__)),
          'scope':'Validation-only permutation choice; exploratory after prior test access, not preregistered.'}
    (OUT/'CHOICE_LOCK.json').write_text(json.dumps(lock,indent=2))
    high_cache={};datasets={};summary={};checks=0
    for name,b in banks.items():
        folder=ROOT/'Research_Release/expanded_validation_results'/name
        if name in ['original_india','original_usa']:folder=ROOT/'Research_Release/recent_data_results'/('delhi' if name=='original_india' else 'epa_la')
        with np.load(folder/'dataset.npz',allow_pickle=False) as d:
            X=d['X'][:,:,list(map(str,d['channels'])).index('PM2.5')].copy();datasets[name]={str(t):x for t,x,s in zip(d['times'],X,d['split']) if s==2}
    with (OLD/'TRIALS.csv').open(newline='',encoding='utf-8') as f,(OUT/'PAIRED_COSTS.csv').open('w',newline='',encoding='utf-8') as target:
        fields=['cohort','timestamp','faults','error_sign','corrupt_station','corruption','validated_queries','training_queries','box_first_queries','full_queries'];writer=csv.DictWriter(target,fieldnames=fields);writer.writeheader()
        for row in csv.DictReader(f):
            name=row['cohort'];faults=int(row['faults'])
            if choices[name]['status']!='frozen':continue
            bank=banks[name];key=(name,row['timestamp'],int(row['error_sign']),row['corrupt_station'],row['corruption'])
            if key not in high_cache:
                bad=bank['stations'].index(row['corrupt_station']) if row['corrupt_station'] else None
                high_cache[key]=simulated_readings(datasets[name][row['timestamp']],key[2],bad,key[4])>36
            high=high_cache[key];order=choices[name]['plans'][str(faults)]['order'];positions=[k+1 for k,v in enumerate(order) if high[v]]
            q=positions[faults] if len(positions)>faults else len(order)
                                                                      
            if (len(positions)>faults)!=(row['direct_prediction']=='1'):raise RuntimeError('Changed direct certificate')
            checks+=1
            record={k:row[k] for k in fields if k in row};record.update(validated_queries=q,training_queries=int(row['training_order_queries']),box_first_queries=int(row['direct_queries']))
            writer.writerow(record)
            key=(name,str(faults));a=summary.setdefault(key,{'trials':0,'validated':0,'training':0,'box_first':0,'full':0})
            a['trials']+=1;a['validated']+=q;a['training']+=int(row['training_order_queries']);a['box_first']+=int(row['direct_queries']);a['full']+=int(row['full_queries'])
    scaling=[]
    for n in [8,16,32,64,128]:
        for density in ['sparse','dense']:
            for seed in range(3):
                rng=np.random.default_rng(seed+100*n);size=3 if density=='sparse' else n//2
                edges=sorted(set(sum(1<<int(v) for v in rng.choice(n,size,replace=False)) for _ in range(64)))
                for faults in [0,1]:
                    q=2*faults+1;t=time.perf_counter();mask=greedy(n,edges,q);seconds=time.perf_counter()-t
                    if mask is None or not all((mask&e).bit_count()>=q for e in edges):raise RuntimeError('Invalid greedy cover')
                    degree=max(sum(bool(e>>v&1) for e in edges) for v in range(n));forced=0
                    for edge in edges:
                        if edge.bit_count()==q:forced|=edge
                    lower=max(q,forced.bit_count(),math.ceil(q*len(edges)/degree))
                    optimum=None;exact_time=None
                    if n<=16:
                        t=time.perf_counter();optimum=master(n,[1]*n,edges,q)[1];exact_time=time.perf_counter()-t
                        if not lower<=optimum<=mask.bit_count():raise RuntimeError('Invalid optimum bracket')
                    scaling.append({'stations':n,'density':density,'seed':seed,'faults':faults,'constraints':len(edges),'lower_bound':lower,'greedy_mask':mask,'forced_stations':forced.bit_count(),'greedy_queries':mask.bit_count(),'exact_queries':optimum,'greedy_seconds':seconds,'exact_seconds':exact_time});checks+=1
    rows=[{'cohort':k[0],'faults':int(k[1]),**v} for k,v in summary.items()]
    report={'status':'complete_and_checked','cohorts':rows,'scaling':scaling,'checks':checks,'scope':'Retrospective validation-chosen order; synthetic scaling with feasible greedy covers and limited exact comparisons.'}
    (OUT/'SUMMARY.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'checks':checks,'scaling_cases':len(scaling)},indent=2))

if __name__=='__main__':main()
