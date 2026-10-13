                                                                                
import json,sys
from pathlib import Path
from fractions import Fraction as Q
from functools import lru_cache

def main():
    root=Path(__file__).resolve().parents[3]
    directory=Path(sys.argv[1]) if len(sys.argv)>1 else root/'Research_Release/decision_repair_results'
    policies=json.loads((directory/'ADAPTIVE_POLICIES.json').read_text());checks=0
    for saved in policies:
        states=saved['states'];labels=saved['labels'];eta=Q.from_float(saved['eta']);f=saved['faults'];n=len(states[0]);m=len(states)
        nodes={(r['remaining'],tuple(r['budgets'])):r for r in saved['policy']}
        memberships=[]
        for v in range(n):
            xs=[Q.from_float(x[v]) for x in states]
            endpoints=sorted({x+s*eta for x in xs for s in [-1,1]})
            samples=endpoints+[endpoints[0]-1,endpoints[-1]+1]+[(a+b)/2 for a,b in zip(endpoints,endpoints[1:])]
            memberships.append(set(tuple(abs(x-y)<=eta for x in xs) for y in samples))
        reached=set()
        @lru_cache(None)
        def visit(left,budgets):
            nonlocal checks
            reached.add((left,budgets));record=nodes[(left,budgets)]
            labs={labels[i] for i,b in enumerate(budgets) if b>=0}
            assert labs;checks+=1
            if len(labs)==1:
                assert record['label']==next(iter(labs)) and record['query'] is None;checks+=1
                return 0
            v=record['query'];assert left>>v&1 and record['label'] is None;checks+=1
            children={tuple(-1 if b<0 else max(-1,b-int(not inside[i])) for i,b in enumerate(budgets)) for inside in memberships[v]}
            children={x for x in children if any(b>=0 for b in x)}
            return 1+max(visit(left^(1<<v),child) for child in children)
        assert visit((1<<n)-1,(f,)*m)==saved['cost'];checks+=1
        assert reached==set(nodes);checks+=1
    report={'status':'passed','policies':len(policies),'checks':checks,
            'scope':'Solver-independent audit of every reachable all-real response class and exported policy cost; not independent optimization of archive-sized banks.'}
    (directory/'ADAPTIVE_POLICY_REPLAY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
