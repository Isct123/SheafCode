                                                                                

                                                                               
                                                                           
   
import numpy as np
from robust_quantity import functional_interval

def projector(B,tolerance):
    if B.shape[1]==0:return np.zeros((B.shape[0],B.shape[0]))
    u,s,_=np.linalg.svd(B,full_matrices=False)
    rank=np.sum(s>tolerance*(s[0] if len(s) else 1.))
    return u[:,:rank]@u[:,:rank].T

def compressed_envelope(A,b,h,blocks,t,epsilon,tolerance=1e-12):
    if not isinstance(t,int) or t<0 or t>len(blocks):raise ValueError('Invalid budget')
    A=np.asarray(A,float);b=np.asarray(b,float);h=np.asarray(h,float)
    if A.ndim!=2 or b.shape!=(A.shape[0],) or h.shape!=(A.shape[1],):raise ValueError('Invalid shapes')
    if not 0<tolerance<1e-6:raise ValueError('Invalid numerical tolerance')
    if any(np.asarray(B).ndim!=2 or np.asarray(B).shape[0]!=A.shape[0] for B in blocks):raise ValueError('Invalid block')
    states=[{'support':[],'projector':np.zeros((A.shape[0],A.shape[0]))}];cursor=0;transitions=0
    while cursor<len(states):
        state=states[cursor];cursor+=1
        if len(state['support'])>=t:continue
        for j in range(len(blocks)):
            if j in state['support']:continue
            support=state['support']+[j];B=np.concatenate([blocks[i] for i in support],axis=1)
            P=projector(B,tolerance);transitions+=1
            if not any(np.linalg.norm(P-s['projector'],2)<=100*tolerance for s in states):
                states.append({'support':support,'projector':P})
    intervals=[]
    for state in states:
        P=state['projector']
        r=functional_interval(A-P@A,b-P@b,h,epsilon)
        intervals.append(r)
    feasible=[r for r in intervals if r['status']!='inconsistent']
    if not feasible:result={'status':'inconsistent'}
    else:
        lo=min(r['lower'] for r in feasible);hi=max(r['upper'] for r in feasible)
        result={'status':'bounded' if np.isfinite([lo,hi]).all() else 'unbounded','lower':lo,'upper':hi}
    return {**result,'distinct_subspaces':len(states),'transitions':transitions,'representative_supports':[s['support'] for s in states],
        'numerical_span_tolerance':tolerance,'scope':'Exact mathematical quotient; numerical implementation with approximate projector equality; no generic polynomial-time claim'}
