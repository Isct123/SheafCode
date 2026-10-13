                                                                         

                                                                           
                                                                                 
   
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from decision_repair import rational

class SearchLimit(RuntimeError):
    pass

def response_masks(values, radius):
    values = [rational(x) for x in values]; radius = rational(radius)
    endpoints = sorted({x + s*radius for x in values for s in [-1,1]})
    if not endpoints:
        return [0]
    points = [endpoints[0]-1, endpoints[-1]+1] + endpoints
    points += [(a+b)/2 for a,b in zip(endpoints,endpoints[1:])]
    return sorted({sum(1<<i for i,x in enumerate(values) if abs(x-y)<=radius) for y in points})

def solve(states, labels, eta, faults=0, costs=None, max_nodes=50000):
    if not states or len(states)!=len(labels):
        raise ValueError('A nonempty labelled scalar bank is required')
    n=len(states[0]); m=len(states)
    costs=[1]*n if costs is None else list(costs)
    radii=[rational(eta)]*n if isinstance(eta,(int,float,Fraction)) else [rational(x) for x in eta]
    if any(len(x)!=n for x in states) or len(costs)!=n or len(radii)!=n:
        raise ValueError('Dimensions differ')
    if any(int(c)!=c or c<=0 for c in costs) or any(x<0 for x in radii) or faults<0 or int(faults)!=faults:
        raise ValueError('Invalid costs, radii or fault count')
    costs=list(map(int,costs)); faults=int(faults)
    values=[[rational(x[v]) for x in states] for v in range(n)]
    patterns=[response_masks(x,radii[v]) for v,x in enumerate(values)]
    separators={(i,j):sum(1<<v for v in range(n) if abs(values[v][i]-values[v][j])>2*radii[v])
                for i,j in combinations(range(m),2) if labels[i]!=labels[j]}
    policy={}; nodes=0

    def children(budget,v):
        return sorted({tuple(-1 if b<0 else max(-1,b-(0 if pattern>>i&1 else 1))
                             for i,b in enumerate(budget)) for pattern in patterns[v]}
                      - {(-1,)*m})

    @lru_cache(None)
    def value(remaining,budget):
        nonlocal nodes
        nodes+=1
        if max_nodes is not None and nodes>max_nodes:
            raise SearchLimit('Declared node limit exceeded')
        active=[i for i,b in enumerate(budget) if b>=0]
        if len({labels[i] for i in active})<=1:
            return 0
                                                                          
                                                                               
        for (i,j),edge in separators.items():
            if budget[i]>=0 and budget[j]>=0 and (edge&remaining).bit_count()<=budget[i]+budget[j]:
                return float('inf')
        best=float('inf'); choice=None
        for v in range(n):
            if not remaining>>v&1:
                continue
            worst=0
            for child in children(budget,v):
                worst=max(worst,value(remaining^(1<<v),child))
                if costs[v]+worst>=best:
                    break
            candidate=costs[v]+worst
            if candidate<best:
                best=candidate;choice=v
        if choice is not None:
            policy[(remaining,budget)]=choice
        return best

    root=((1<<n)-1,(faults,)*m)
    try:
        optimum=value(*root)
    except SearchLimit:
        return {'status':'search_limit','nodes':nodes,'max_nodes':max_nodes}
    if optimum==float('inf'):
        return {'status':'infeasible','nodes':nodes}

                                                                               
    checked=set()
    def audit(remaining,budget):
        key=(remaining,budget)
        if key in checked:
            return value(*key)
        checked.add(key)
        active=[i for i,b in enumerate(budget) if b>=0]
        if len({labels[i] for i in active})<=1:
            return 0
        v=policy[key]
        worst=costs[v]+max(audit(remaining^(1<<v),child) for child in children(budget,v))
        assert worst==value(*key)
        return worst
    assert audit(*root)==optimum
    exported=[]
    for remaining,budget in sorted(checked):
        active=[i for i,b in enumerate(budget) if b>=0]
        labels_here={labels[i] for i in active}
        exported.append({'remaining':remaining,'budgets':list(budget),
                         'query':policy.get((remaining,budget)),
                         'label':next(iter(labels_here)) if len(labels_here)==1 else None})
    return {'status':'optimal','cost':optimum,'first_station':policy.get(root),
            'nodes':nodes,'policy_states_checked':len(checked),'policy':exported}
