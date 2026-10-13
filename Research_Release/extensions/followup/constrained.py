                                                                              
from itertools import combinations
import numpy as np
from scipy.optimize import linprog

def constrained_envelope(A,b,h,blocks,t,residual_bound,bias_bound,lower=None,upper=None):
    A=np.asarray(A,float);b=np.asarray(b,float);h=np.asarray(h,float)
    if A.ndim!=2 or b.shape!=(len(A),) or h.shape!=(A.shape[1],):raise ValueError('Invalid dimensions')
    p=len(h);eps=np.broadcast_to(np.asarray(residual_bound,float),(len(b),))
    if not np.isfinite(eps).all() or (eps<0).any() or not np.isfinite(bias_bound) or bias_bound<0:raise ValueError('Invalid bounds')
    if not isinstance(t,int) or not 0<=t<=len(blocks):raise ValueError('Invalid fault count budget')
    lo=np.full(p,-np.inf) if lower is None else np.asarray(lower,float)
    hi=np.full(p,np.inf) if upper is None else np.asarray(upper,float)
    if lo.shape!=(p,) or hi.shape!=(p,) or np.isnan(lo).any() or np.isnan(hi).any() or (lo>hi).any():raise ValueError('Invalid state box')
    ranges=[];statuses=[]
    for k in range(t+1):
        for S in combinations(range(len(blocks)),k):
            B=np.concatenate([blocks[j] for j in S],axis=1) if S else np.empty((len(b),0))
            M=np.concatenate([A,B],axis=1);objective=np.r_[h,np.zeros(B.shape[1])]
            bounds=list(zip(lo,hi))+[(-bias_bound,bias_bound)]*B.shape[1]
            G=np.vstack([M,-M]);rhs=np.r_[eps-b,eps+b]
            low=linprog(objective,A_ub=G,b_ub=rhs,bounds=bounds,method='highs')
            high=linprog(-objective,A_ub=G,b_ub=rhs,bounds=bounds,method='highs')
            if low.status==2 or high.status==2:statuses.append('inconsistent');continue
            if low.status==3 or high.status==3:
                ranges.append((-np.inf if low.status==3 else low.fun,np.inf if high.status==3 else -high.fun));statuses.append('unbounded');continue
            if not low.success or not high.success:raise RuntimeError('LP solver failed: '+low.message+' / '+high.message)
                                                                             
            for fit in [low,high]:
                if np.max(G@fit.x-rhs)>1e-6:raise RuntimeError('LP feasibility check failed')
            ranges.append((float(low.fun),float(-high.fun)));statuses.append('bounded')
    if not ranges:return {'status':'inconsistent','supports':len(statuses),'feasible_supports':0}
    L=min(x[0] for x in ranges);U=max(x[1] for x in ranges)
    return {'status':'bounded' if np.isfinite([L,U]).all() else 'unbounded','lower':L,'upper':U,
        'supports':len(statuses),'feasible_supports':len(ranges),'numerical_solver':'HiGHS, checked primal tolerance 1e-6'}
