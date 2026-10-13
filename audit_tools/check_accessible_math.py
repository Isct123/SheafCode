                                                                       
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import json,random

def bound(rows,rhs,weights,score,lo,hi):
    delta=[c-sum(w*row[j] for row,w in zip(rows,weights)) for j,c in enumerate(score)]
    return sum(w*b for w,b in zip(weights,rhs))+sum(max(d*a,d*b) for d,a,b in zip(delta,lo,hi))

def main():
    rng=random.Random(8127);checks=0;points=0
    rows=[[Q(2),Q(1)]];rhs=[Q(5)];lo=[Q(0),Q(0)];hi=[Q(2),Q(3)]
    assert bound(rows,rhs,[Q(3,2)],[Q(3),Q(2)],lo,hi)==9;checks+=1
    assert bound(rows,rhs,[Q(149,100)],[Q(3),Q(2)],lo,hi)==Q(451,50);checks+=1
    weight=Q.from_float(.999999999)
    assert bound([[Q(1)]],[Q(10**9)],[weight],[Q(1)],[Q(0)],[Q(10**9)])==10**9;checks+=1
    for _ in range(200):
        rows=[[Q(rng.randint(-4,4),2) for _ in range(2)] for _ in range(3)]
        rhs=[Q(rng.randint(0,8),2) for _ in rows];weights=[Q(rng.randint(0,5),3) for _ in rows]
        score=[Q(rng.randint(-5,5),3) for _ in range(2)];lo=[Q(-2),Q(-1)];hi=[Q(2),Q(3)]
        upper=bound(rows,rhs,weights,score,lo,hi);lower=-bound(rows,rhs,weights,[-c for c in score],lo,hi)
        for x in product([Q(k,2) for k in range(-4,5)],[Q(k,2) for k in range(-2,7)]):
            if all(sum(a*b for a,b in zip(row,x))<=b for row,b in zip(rows,rhs)):
                q=sum(a*b for a,b in zip(score,x));assert lower<=q<=upper;checks+=1;points+=1
    tree_steps=0
    for _ in range(100):
        n=5;t=2;scores={m:Q(rng.randint(-20,20),3) for m in range(1<<n) if m.bit_count()<=t};opt=max(scores.values())
        active=[(0,0,opt+Q(rng.randint(0,5),2))];inc=None;last_upper=active[0][2];last_lower=None;visited=0
        while active:
            lower=inc if inc is not None else float('-inf');upper=max([b for _,_,b in active]+([inc] if inc is not None else []))
            assert lower<=opt<=upper;checks+=1
            assert upper<=last_upper;checks+=1
            if last_lower is not None:assert lower>=last_lower;checks+=1
            last_upper=upper;last_lower=lower
            depth,mask,parent=active.pop();visited+=1;tree_steps+=1
            if inc is not None and parent<=inc:continue
            if depth==n:
                value=scores[mask];inc=value if inc is None else max(inc,value);continue
            for flag in [0,1]:
                child=mask|(flag<<depth);k=depth+1;prefix=(1<<k)-1
                compatible=[value for m,value in scores.items() if m&prefix==child]
                if not compatible:continue
                upper=min(parent,max(compatible)+Q(rng.randint(0,5),2))
                active.append((k,child,upper))
        assert inc==opt;checks+=1
        assert visited<=2**(n+1)-1;checks+=1
    result={'status':'passed','checks':checks,'feasible_grid_points':points,'weighted_bound_cases':200,'search_trees':100,'tree_steps':tree_steps,
            'scope':'Exact-rational worked examples, finite grid tests and independent finite-tree invariants; supplements written proofs, not a new empirical benchmark or formal proof certification.'}
    out=Path(__file__).resolve().parents[1]/'SUBMISSION_WORKSPACE/Guides/ACCESSIBLE_MATH_CHECK.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
