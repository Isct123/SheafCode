                                                                           
from pathlib import Path
from fractions import Fraction as Q
from functools import lru_cache
import csv,json,time,hashlib,sys
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'decision_repair_results';OUT=ROOT/'joint_certification_results'
def rat(x):return Q.from_float(float(x))

def main():
    start=time.perf_counter();banks=json.loads((SOURCE/'SCENARIO_BANKS.json').read_text());rows=list(csv.DictReader((SOURCE/'REPAIR_CONFIGURATIONS.csv').open()));fixed=[]
    for row in rows:
        bank=banks[row['cohort']];threshold=rat(row['threshold']);eta=rat(row['eta']);faults=int(row['query_faults']);n=len(bank['stations']);states=[];labels=[]
        for state in bank['pm25_vectors']:
            maximum=max(map(rat,state))
            if maximum<=threshold-eta or maximum>threshold+eta:states.append(state);labels.append(int(maximum>threshold+eta))
        assert len(states)==int(row['scenarios'])
        if len(set(labels))<2:
            assert row['decision_status']=='trivial_or_empty';fixed.append({'cohort':row['cohort'],'status':'trivial_or_empty'});continue
        def optimum(decision):
            edges=[]
            for i in range(len(states)):
                for j in range(i):
                    if decision and labels[i]==labels[j]:continue
                    edges.append(sum(1<<v for v in range(n) if abs(rat(states[i][v])-rat(states[j][v]))>2*eta))
            candidates=[mask.bit_count() for mask in range(1<<n) if all((mask&e).bit_count()>=2*faults+1 for e in edges)]
            return min(candidates) if candidates else None
        a=optimum(True);b=optimum(False)
        assert a==(int(row['decision_queries']) if row['decision_status']=='optimal' else None)
        assert b==(int(row['pattern_queries']) if row['pattern_status']=='optimal' else None)
        fixed.append({'cohort':row['cohort'],'decision_optimum':a,'pattern_optimum':b})
    radius_frontiers=json.loads((SOURCE/'RADIUS_FRONTIERS.json').read_text());radius_points=0
    for saved in radius_frontiers:
        config=rows[saved['configuration']];eta=rat(config['eta']);threshold=rat(config['threshold']);n=len(banks[config['cohort']]['stations']);states=[];labels=[]
        for x in banks[config['cohort']]['pm25_vectors']:
            maximum=max(map(rat,x))
            if maximum<=threshold-eta or maximum>threshold+eta:states.append(list(map(rat,x)));labels.append(int(maximum>threshold+eta))
        pairs=[(i,j) for i in range(len(states)) for j in range(i) if labels[i]!=labels[j]];q=2*saved['faults']+1
        def separation(mask):
            if mask.bit_count()<q:return float('-inf')
            return min(sorted((abs(states[i][v]-states[j][v])/2-eta for v in range(n) if mask>>v&1),reverse=True)[q-1] for i,j in pairs)
        semantic=min(abs(max(x)-threshold) for x in states);assert semantic==Q(saved['semantic_margin']['exact'])
        values={mask:separation(mask) for mask in range(1<<n)}
        for point in saved['frontier']:
            target=Q(point['radius']['exact']) if isinstance(point['radius'],dict) else float(point['radius'])
            assert point['mask'].bit_count()<=point['budget']
            assert target==values[point['mask']]==max(value for mask,value in values.items() if mask.bit_count()<=point['budget']);radius_points+=1
    policies=json.loads((SOURCE/'ADAPTIVE_POLICIES.json').read_text());adaptive=[]
    for number,saved in enumerate(policies):
        states=saved['states'];labels=saved['labels'];eta=rat(saved['eta']);f=saved['faults'];n=len(states[0]);m=len(states);classes=[];count=0
        for v in range(n):
            centres=list(map(rat,[x[v] for x in states]));endpoints=sorted({z+e*eta for z in centres for e in [-1,1]});samples=endpoints+[endpoints[0]-1,endpoints[-1]+1]+[(a+b)/2 for a,b in zip(endpoints,endpoints[1:])]
            classes.append(set(tuple(abs(z-y)<=eta for z in centres) for y in samples))
        @lru_cache(None)
        def optimize(left,budgets):
            nonlocal count
            count+=1
            if count>1000000:raise RuntimeError('Independent adaptive optimization exceeded explicit limit; no optimum certified')
            active_labels={labels[i] for i,b in enumerate(budgets) if b>=0}
            assert active_labels
            if len(active_labels)==1:return 0
            if not left:return float('inf')
            best=float('inf')
            for v in range(n):
                if not left>>v&1:continue
                responses={tuple(-1 if b<0 else b-int(not member[i]) for i,b in enumerate(budgets)) for member in classes[v]}
                responses={tuple(max(-1,b) for b in child) for child in responses if any(b>=0 for b in child)}
                worst=0
                for child in responses:
                    worst=max(worst,optimize(left^(1<<v),child))
                    if 1+worst>=best:break
                best=min(best,1+worst)
            return best
        optimum=optimize((1<<n)-1,(f,)*m);assert optimum==saved['cost']
        adaptive.append({'index':number,'cost':optimum,'states_evaluated':count,'scope':'Independent optimization over all station choices and all real-response classes'})
        if (number+1)%10==0:print(number+1,'independent policy optima checked',round(time.perf_counter()-start,1),'seconds',flush=True)
    report={'status':'passed','fixed_configurations':len(fixed),'radius_frontiers':len(radius_frontiers),'radius_budget_points':radius_points,'adaptive_policies':len(adaptive),'adaptive_states_evaluated':sum(v['states_evaluated'] for v in adaptive),'elapsed_seconds':time.perf_counter()-start,'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'SCENARIO_BANKS.json',SOURCE/'REPAIR_CONFIGURATIONS.csv',SOURCE/'ADAPTIVE_POLICIES.json',SOURCE/'RADIUS_FRONTIERS.json']},'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Independent exact-rational conflict construction/exhaustive fixed optimum, exhaustive radius-budget frontier optimization, and independently optimized minimax policy costs. Small saved banks, no field validation.','adaptive_details':adaptive}
    if '--check-only' not in sys.argv:(OUT/'INDEPENDENT_OPTIMA.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='adaptive_details'}))

if __name__=='__main__':main()
