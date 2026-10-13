                                                                               
from itertools import combinations
from numbers import Real
from decision_repair import rational

def radius(states,labels,eta,mask,faults=0):
    if not states or len(states)!=len(labels):raise ValueError('Nonempty labelled bank required')
    n=len(states[0]);q=2*faults+1
    radii=[rational(eta)]*n if isinstance(eta,Real) else [rational(x) for x in eta]
    if any(len(x)!=n for x in states) or len(radii)!=n or any(x<0 for x in radii):raise ValueError('Invalid state/radius dimensions')
    if int(faults)!=faults or faults<0 or mask<0 or mask>>n:raise ValueError('Invalid fault count or query mask')
    q=int(q)
    pairs=[(i,j) for i,j in combinations(range(len(states)),2) if labels[i]!=labels[j]]
    if not pairs:return {'radius':float('inf'),'witness':None}
    if mask.bit_count()<q:return {'radius':float('-inf'),'witness':list(pairs[0])}
    best=None;witness=None
    for i,j in pairs:
        margins=sorted(((abs(rational(states[i][v])-rational(states[j][v]))/2-radii[v],v)
                        for v in range(n) if mask>>v&1),reverse=True)
        critical=margins[q-1][0]
        if best is None or critical<best:
            best=critical;witness=[i,j]
    return {'radius':best,'witness':witness}

def frontier(states,labels,eta,faults=0,costs=None):
    n=len(states[0]);costs=[1]*n if costs is None else list(costs)
    if len(costs)!=n or any(c<=0 or int(c)!=c for c in costs):raise ValueError('Positive integer costs required')
    choices=[]
    for mask in range(1<<n):
        r=radius(states,labels,eta,mask,faults)
        choices.append((sum(int(costs[v]) for v in range(n) if mask>>v&1),mask,r))
    output=[]
    for budget in sorted({c for c,_,_ in choices}):
        feasible=[(r['radius'],-mask.bit_count(),-mask,mask,r) for c,mask,r in choices if c<=budget]
        _,_,_,mask,r=max(feasible)
        output.append({'budget':budget,'mask':mask,'radius':r['radius'],'witness':r['witness']})
    return output

def ambiguity_witness(states,pair,eta,mask,rho,faults=0):
                                                                                    
    i,j=pair;n=len(states[0]);rho=rational(rho)
    if rho<0:raise ValueError('Perturbation radius cannot be negative')
    radii=[rational(eta)]*n if isinstance(eta,Real) else [rational(x) for x in eta]
    a=[rational(x) for x in states[i]];b=[rational(x) for x in states[j]]
    separate=[v for v in range(n) if mask>>v&1 and abs(a[v]-b[v])>2*(radii[v]+rho)]
    if len(separate)>2*faults:raise ValueError('The pair is still separable')
    first=set(separate[:faults]);second=set(separate[faults:]);response={}
    aa=a.copy();bb=b.copy()
    for v in range(n):
        if not mask>>v&1:continue
        y=b[v] if v in first else a[v] if v in second else (a[v]+b[v])/2
        response[v]=y
        aa[v]=min(a[v]+rho,max(a[v]-rho,y));bb[v]=min(b[v]+rho,max(b[v]-rho,y))
    assert all(abs(x-y)<=rho for x,y in zip(aa,a)) and all(abs(x-y)<=rho for x,y in zip(bb,b))
    assert sum(abs(aa[v]-y)>radii[v] for v,y in response.items())<=faults
    assert sum(abs(bb[v]-y)>radii[v] for v,y in response.items())<=faults
    return {'a':aa,'b':bb,'response':response}
