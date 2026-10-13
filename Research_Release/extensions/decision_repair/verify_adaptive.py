                                                                               
import sys,json,random,hashlib
from pathlib import Path
from fractions import Fraction as Q
sys.path.insert(0,str(Path(__file__).parent))
from adaptive_repair import solve,response_masks
from decision_repair import conflict_edges,plan

def brute(states,labels,eta,f):
                                                                             
    n=len(states[0]);m=len(states);possible=[]
    for v in range(n):
        cuts=sorted({Q(x[v])+s*Q(eta) for x in states for s in [-1,1]})
        ys=cuts+[cuts[0]-1,cuts[-1]+1]+[(a+b)/2 for a,b in zip(cuts,cuts[1:])]
        possible.append(set(tuple(abs(Q(x[v])-y)>Q(eta) for x in states) for y in ys))
    def tree(left,counts):
        active=[i for i,c in enumerate(counts) if c<=f]
        if len({labels[i] for i in active})<=1:return 0
        if not left:return float('inf')
        scores=[]
        for v in left:
            children={tuple(c+int(miss[i]) for i,c in enumerate(counts)) for miss in possible[v]}
            children=[x for x in children if any(c<=f for c in x)]
            scores.append(1+max(tree(tuple(u for u in left if u!=v),x) for x in children))
        return min(scores)
    return tree(tuple(range(n)),(0,)*m)

def main():
    rng=random.Random(1037);checks=0
    assert response_masks([0,2],1)==[0,1,2,3];checks+=1
    for trial in range(80):
        states=[[rng.randrange(5) for _ in range(3)] for _ in range(4)]
        labels=[0,1,rng.randrange(2),rng.randrange(2)]
        eta=Q(rng.randrange(3),2);f=trial%2
        ref=brute(states,labels,eta,f);got=solve(states,labels,eta,faults=f)
        assert (ref==float('inf'))==(got['status']=='infeasible');checks+=1
        if got['status']=='optimal':assert got['cost']==ref;checks+=1
    states=[];labels=[]
    for route in range(3):
        for label in [0,1]:
            x=[10*route,0,0,0];x[route+1]=10*label
            states.append(x);labels.append(label)
    adaptive=solve(states,labels,Q(1,10))
    fixed=plan(4,conflict_edges(states,labels,Q(1,10)))
    assert adaptive['cost']==2 and fixed['cost']==3;checks+=1
    assert solve([[0,0,0],[10,10,10]],[0,1],1,faults=1)['cost']==3;checks+=1
    assert solve([[0,0],[10,10]],[0,1],1,faults=1)['status']=='infeasible';checks+=1
    assert solve(states,labels,Q(1,10),costs=[5,1,1,1])['cost']==3;checks+=1
    assert solve(states,labels,Q(1,10),max_nodes=1)['status']=='search_limit';checks+=1
    path=Path(__file__).resolve().parents[3]/'Research_Release/decision_repair_results/ADAPTIVE_VERIFICATION.json'
    report={'status':'passed','random_banks':80,'checks':checks,'independent_reference':'Unpruned decision trees with mismatch counts; separate all-real response-cell enumeration',
            'worked_example':{'fixed_queries':3,'adaptive_worst_case_queries':2},
            'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('adaptive_repair.py')]}}
    path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
