                                                                         
from itertools import combinations
import numpy as np

def observed_margin(A, p, t):
                                                                          
    A=np.asarray(A,float)
    if A.ndim!=2 or p<1 or A.shape[1]%p or t<1:
        raise ValueError('Invalid operator or block sizes')
    worst=np.inf; count=0
    for k in range(1,min(2*t,A.shape[1]//p)+1):
        for S in combinations(range(A.shape[1]//p),k):
            cols=np.concatenate([np.arange(v*p,(v+1)*p) for v in S])
            s=np.linalg.svd(A[:,cols],compute_uv=False)
            value=float(s[-1]) if len(s)==len(cols) else 0.
            worst=min(worst,value);count+=1
    return {'sigma':worst,'sets':count}

def mismatch_budget(epsilon, eta, state_bound, observed_error=0.):
                                                                      
    x=np.asarray([epsilon,eta,state_bound,observed_error],float)
    if not np.isfinite(x).all() or (x<0).any():raise ValueError('Finite nonnegative bounds required')
    return float(epsilon+eta*state_bound+observed_error)

def minimum_reveals(A,h,max_reveals=None):
                                                                                 

                                                                              
       
    A=np.asarray(A,float);h=np.asarray(h,float)
    if A.ndim!=2 or h.shape!=(A.shape[1],):raise ValueError('Incompatible dimensions')
    limit=A.shape[1] if max_reveals is None else min(max_reveals,A.shape[1])
    for k in range(limit+1):
        for R in combinations(range(A.shape[1]),k):
            B=np.vstack([A,np.eye(A.shape[1])[list(R)]])
            projection=B.T@np.linalg.lstsq(B.T,h,rcond=1e-12)[0]
            if np.linalg.norm(h-projection)<=1e-9*max(1.,np.linalg.norm(h)):
                return {'coordinates':list(R),'cardinality':k,'status':'identified'}
    return {'status':'not_found_within_limit'}
