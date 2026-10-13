                                                                     

                                                                         
                                           
   
from itertools import combinations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'functional'))
from certificate import functional_interval

def supports(blocks,t):
    if not isinstance(t,int) or t<0 or t>len(blocks):raise ValueError('Invalid fault budget')
    return [S for k in range(t+1) for S in combinations(range(len(blocks)),k)]

def project_nuisance(A,b,B):
    if B.shape[1]==0:return A,b
    u,s,_=np.linalg.svd(B,full_matrices=False)
    rank=np.sum(s>1e-12*(s[0] if len(s) else 1));Q=u[:,:rank]
    return A-Q@(Q.T@A),b-Q@(Q.T@b)

def fault_envelope(A,b,h,blocks,t,epsilon):
    A=np.asarray(A,float);b=np.asarray(b,float);h=np.asarray(h,float)
    if A.ndim!=2 or b.shape!=(A.shape[0],) or h.shape!=(A.shape[1],):raise ValueError('Invalid dimensions')
    if any(np.asarray(B).ndim!=2 or np.asarray(B).shape[0]!=len(b) for B in blocks):raise ValueError('Invalid fault block')
    if not all(np.isfinite(x).all() for x in [A,b,h]+[np.asarray(B) for B in blocks]):raise ValueError('Finite inputs required')
    intervals=[]
    for S in supports(blocks,t):
        B=np.concatenate([blocks[i] for i in S],axis=1) if S else np.empty((len(b),0))
        C,d=project_nuisance(A,b,B);r=functional_interval(C,d,h,epsilon)
        intervals.append({'support':list(S),**r})
    feasible=[r for r in intervals if r['status']!='inconsistent']
    if not feasible:return {'status':'inconsistent','supports_checked':len(intervals),'intervals':intervals}
    lo=min(r['lower'] for r in feasible);hi=max(r['upper'] for r in feasible)
    return {'status':'bounded' if np.isfinite([lo,hi]).all() else 'unbounded','lower':lo,'upper':hi,
        'supports_checked':len(intervals),'feasible_supports':len(feasible),'intervals':intervals,
        'scope':'Sharp extrema over union of support-feasible sets; convex hull may contain unattainable gaps'}

def uniformly_identifiable(A,h,blocks,t):
                                                                       
    for J in supports(blocks,min(2*t,len(blocks))):
        B=np.concatenate([blocks[i] for i in J],axis=1) if J else np.empty((A.shape[0],0))
        M=np.concatenate([A,B],axis=1);target=np.concatenate([h,np.zeros(B.shape[1])])
        dual=np.linalg.lstsq(M.T,target,rcond=1e-12)[0]
        if np.linalg.norm(M.T@dual-target)>1e-9*max(1.,np.linalg.norm(target)):
            return {'identifiable':False,'witness_union':list(J)}
    return {'identifiable':True}
