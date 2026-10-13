                                                                                       
from itertools import combinations
import math
from fractions import Fraction
from numbers import Real

def rational(x):
    return x if isinstance(x,Fraction) else Fraction.from_float(float(x))

def vector(x):
    return [rational(x)] if isinstance(x,Real) else [rational(v) for v in x]

def conflict_edges(states, labels, eta):
                                                                          
    if not states:
        return []
    n = len(states[0]); radii = [rational(eta)]*n if isinstance(eta,Real) else [rational(v) for v in eta]
    if len(labels)!=len(states) or len(radii)!=n or any(v<0 for v in radii):
        raise ValueError('Invalid labels or noise radii')
    edges = {}
    for i,j in combinations(range(len(states)),2):
        if labels[i]==labels[j]:
            continue
        mask = 0
        for v in range(n):
            a,b = vector(states[i][v]),vector(states[j][v])
            if len(a)!=len(b):
                raise ValueError('Station dimensions differ')
            if max((abs(x-y) for x,y in zip(a,b)),default=0)>2*radii[v]:
                mask |= 1<<v
        edges.setdefault(mask,(i,j))
    return [(mask,edges[mask]) for mask in sorted(edges)]

def master(n, costs, constraints, required):
                                                                                     
    if any(int(c)!=c or c<0 for c in costs):
        raise ValueError('Exact small-n solver requires nonnegative integer costs')
    costs = [int(c) for c in costs];best = None
    for mask in range(1<<n):
        if any((mask&edge).bit_count()<required for edge in constraints):
            continue
        cost = sum(costs[v] for v in range(n) if mask>>v&1)
        item = (cost,mask.bit_count(),mask)
        if best is None or item<best:
            best = item
    return None if best is None else (best[2],best[0])

def plan(n, edges, costs=None, faults=0, max_refinements=None):
                                                                                  

                                                                                    
                                                                                     
       
    costs = [1]*n if costs is None else list(costs)
    if len(costs)!=n or any(not math.isfinite(c) or c<0 or int(c)!=c for c in costs) or faults<0 or int(faults)!=faults:
        raise ValueError('Invalid cost/fault specification')
    costs = [int(c) for c in costs]
    required = 2*int(faults)+1; full = (1<<n)-1
    if any(edge&~full for edge,_ in edges):
        raise ValueError('Edge outside repair universe')
    for edge,witness in edges:
        if edge.bit_count()<required:
            return {'status':'infeasible','witness':list(witness),'edge':edge,'required':required,'refinements':0}
    constraints = []; trace = []; upper = sum(costs)
    while True:
        choice = master(n,costs,constraints,required)
        if choice is None:
            raise AssertionError('Full set was feasible but master failed')
        mask,lower = choice
        trace.append({'lower':lower,'upper':upper,'candidate':mask,'constraints':len(constraints)})
        if max_refinements is not None and len(constraints)>=max_refinements:
            return {'status':'bounded','lower':lower,'upper':upper,'refinements':len(constraints),'trace':trace}
        conflict = next(((edge,witness) for edge,witness in edges if (edge&mask).bit_count()<required),None)
        if conflict is None:
            return {'status':'optimal','mask':mask,'cost':lower,'stations':[v for v in range(n) if mask>>v&1],
                    'lower':lower,'upper':lower,'refinements':len(constraints),'trace':trace}
        edge,witness = conflict
        if edge in constraints:
            raise AssertionError('Previously enforced conflict returned')
        constraints.append(edge)

def infer(states, labels, eta, readings, faults=0):
                                                                                   
    n = len(states[0]); radii = [rational(eta)]*n if isinstance(eta,Real) else [rational(v) for v in eta]
    possible = set()
    for state,label in zip(states,labels):
        misses = 0
        for v,y in readings.items():
            x,yy = vector(state[v]),vector(y)
            if max(abs(a-b) for a,b in zip(x,yy))>radii[v]:
                misses += 1
        if misses<=faults:
            possible.add(label)
    return next(iter(possible)) if len(possible)==1 else None
