                                                                                    
from pathlib import Path
from itertools import permutations
from collections import Counter
from fractions import Fraction as Q
import sys,csv,json,math,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from operational_guard import assess
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'Research_Release/operational_validation_results'
OUT=ROOT/'Research_Release/query_policy_results'

def require(ok,message):
    if not ok:raise RuntimeError(message)
def noise(x,sign,bad,corruption):
    y=x+sign
    for v in range(len(x)):
        if abs(Q.from_float(float(y[v]))-Q.from_float(float(x[v])))>abs(sign):y[v]=np.nextafter(y[v],x[v])
    if bad is not None:y[bad]=-65. if corruption=='low' else 135.
    return y
def query_cost(mask,order,faults):
    positions=[i+1 for i,v in enumerate(order) if mask>>v&1]
    return positions[faults] if len(positions)>faults else len(order)

def main():
    lock=json.loads((OUT/'CHOICE_LOCK.json').read_text());banks=json.loads((OLD/'FROZEN_BANKS.json').read_text());checks=0;datasets={}
    for name,record in lock['choices'].items():
        if record['status']!='frozen':continue
        folder=ROOT/'Research_Release/expanded_validation_results'/name
        if name in ['original_india','original_usa']:folder=ROOT/'Research_Release/recent_data_results'/('delhi' if name=='original_india' else 'epa_la')
        require(hashlib.sha256((folder/'dataset.npz').read_bytes()).hexdigest()==record['dataset_sha256'],'Changed dataset');checks+=1
        with np.load(folder/'dataset.npz',allow_pickle=False) as d:
            X=d['X'][:,:,list(map(str,d['channels'])).index('PM2.5')];split=d['split'];times=d['times'];val=X[(split==1)&np.isfinite(X).all(1)]
            datasets[name]={str(t):x.copy() for t,x,s in zip(times,X,split) if s==2}
        n=len(banks[name]['stations'])
        for faults in [0,1]:
            counts=Counter()
            for x in val:
                trials=[(s,None,'none') for s in [-1,0,1]] if faults==0 else [(s,v,c) for s in [-1,1] for v in range(n) for c in ['low','high']]
                for s,v,c in trials:
                    y=noise(x,s,v,c);counts[sum(1<<k for k,z in enumerate(y) if z>36)]+=1
            reference=min((sum(weight*query_cost(mask,order,faults) for mask,weight in counts.items()),order) for order in permutations(range(n)))
            frozen=record['plans'][str(faults)]
            require(reference==(frozen['validation_query_sum'],tuple(frozen['order'])),'Selected order is not the validation minimum');checks+=1
    with (OLD/'TRIALS.csv').open(newline='',encoding='utf-8') as original,(OUT/'PAIRED_COSTS.csv').open(newline='',encoding='utf-8') as output:
        a=csv.DictReader(original);b=csv.DictReader(output);trials=0;api_checks=0
        for row in a:
            name=row['cohort']
            if lock['choices'][name]['status']!='frozen':continue
            saved=next(b);trials+=1
            for key in ['cohort','timestamp','faults','error_sign','corrupt_station','corruption']:require(row[key]==saved[key],'Matched-row mismatch');checks+=1
            bank=banks[name];faults=int(row['faults']);order=lock['choices'][name]['plans'][str(faults)]['order'];bad=bank['stations'].index(row['corrupt_station']) if row['corrupt_station'] else None
            y=noise(datasets[name][row['timestamp']],int(row['error_sign']),bad,row['corruption']);mask=sum(1<<v for v,z in enumerate(y) if z>36)
            q=query_cost(mask,order,faults);require(q==int(saved['validated_queries']),'Selected-order test cost mismatch');checks+=1
            require(int(saved['training_queries'])==int(row['training_order_queries']),'Training comparator mismatch');checks+=1
            require(int(saved['box_first_queries'])==int(row['direct_queries']),'Box comparator mismatch');checks+=1
            if trials%37==0:
                measurements={bank['stations'][v]:float(y[v]) for v in order[:q]}
                result=assess(bank['stations'],measurements,faults=faults)
                expected={'':'abstain','0':'safe','1':'exceedance'}[row['direct_prediction']]
                require(result['label']==expected,'Operational guard integration mismatch');checks+=1;api_checks+=1
        require(next(b,None) is None,'Extra unmatched test costs');checks+=1
    report=json.loads((OUT/'SUMMARY.json').read_text())
    for record in report['scaling']:
        n=record['stations'];rng=np.random.default_rng(record['seed']+100*n);size=3 if record['density']=='sparse' else n//2
        edges=sorted(set(sum(1<<int(v) for v in rng.choice(n,size,replace=False)) for _ in range(64)));q=2*record['faults']+1;mask=record['greedy_mask']
        require(all((mask&e).bit_count()>=q for e in edges),'Invalid scaling cover');checks+=1
        require(mask.bit_count()==record['greedy_queries'],'Query count mismatch');checks+=1
        degree=max(sum(bool(e>>v&1) for e in edges) for v in range(n));forced=0
        for edge in edges:
            if edge.bit_count()==q:forced|=edge
        bound=max(q,forced.bit_count(),math.ceil(q*len(edges)/degree));require(bound==record['lower_bound'],'Lower bound mismatch');checks+=1
        if n<=16:
            best=n
            for candidate in range(1<<n):
                count=candidate.bit_count()
                if count>=best:continue
                if all((candidate&e).bit_count()>=q for e in edges):best=count
            require(best==record['exact_queries'],'Independent exact optimum mismatch');checks+=1
    final={'status':'passed','paired_trials':trials,'validation_orders':sum(r['status']=='frozen' for r in lock['choices'].values())*2,
           'scaling_cases':len(report['scaling']),'api_integration_checks':api_checks,'checks':checks,
           'scope':'Independent histogram-based validation optimization, all paired test query counts, sampled live API checks and exact small synthetic optima.'}
    (OUT/'INDEPENDENT_REPLAY.json').write_text(json.dumps(final,indent=2));print(json.dumps(final,indent=2))

if __name__=='__main__':main()
