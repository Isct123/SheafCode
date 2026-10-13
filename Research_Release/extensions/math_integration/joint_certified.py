                                                                             
from pathlib import Path
import sys,itertools,math
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from fractions import Fraction as Q
from search import rat,solve_node

def outward(value,upper):
    x=float(value)
    if (upper and rat(x)<value) or (not upper and rat(x)>value):x=float(np.nextafter(x,np.inf if upper else -np.inf))
    return x

def certify_joint(A,b,H,offsets,blocks,t,eps,beta,lo,hi):
    A,b,H,offsets,lo,hi=map(lambda x:np.asarray(x,float),(A,b,H,offsets,lo,hi))
    low_scores=[sum((min(rat(h)*rat(l),rat(h)*rat(u)) for h,l,u in zip(row,lo,hi)),rat(c)) for row,c in zip(H,offsets)]
    high_scores=[sum((max(rat(h)*rat(l),rat(h)*rat(u)) for h,l,u in zip(row,lo,hi)),rat(c)) for row,c in zip(H,offsets)]
                                                                     
    ylo=outward(max(low_scores),False);yhi=outward(max(high_scores),True)
    proofs=[];lower_bounds=[];upper_on_lower=[];upper_bounds=[];lower_on_upper=[]
    for size in range(t+1):
        for S in itertools.combinations(range(len(blocks)),size):
            B=np.concatenate([blocks[v] for v in S],axis=1) if S else np.empty((len(A),0));M=np.c_[A,B];q=B.shape[1]
            G=np.r_[M,-M];rhs=np.r_[eps-b,eps+b];ll=np.r_[lo,np.full(q,-beta)];hh=np.r_[hi,np.full(q,beta)]
            J=np.c_[H,np.zeros((len(H),q)),-np.ones(len(H))]
            GG=np.r_[np.c_[G,np.zeros(len(G))],J];rr=np.r_[rhs,-offsets];lly=np.r_[ll,ylo];hhy=np.r_[hh,yhi];c=np.r_[np.zeros(len(ll)),-1.]
            minimum=solve_node(GG,rr,lly,hhy,c,True);minimum.update({'support':list(S),'kind':'joint_minimum'});proofs.append(minimum)
            if minimum['infeasible']:continue
            lower_bounds.append(-minimum['upper'])
            point=minimum.get('point')
            if point is not None:
                actual=max(sum((rat(a)*z for a,z in zip(row,point[:len(lo)])),rat(offset)) for row,offset in zip(H,offsets));upper_on_lower.append(actual)
            for j,h in enumerate(H):
                c=np.r_[h,np.zeros(q)];maximum=solve_node(G,rhs,ll,hh,c,True);maximum.update({'support':list(S),'kind':'pollutant_maximum','channel':j});proofs.append(maximum)
                if maximum['infeasible']:raise RuntimeError('Exact feasibility disagreement across objectives')
                upper_bounds.append(maximum['upper']+rat(offsets[j]));point=maximum.get('point')
                if point is not None:lower_on_upper.append(sum((rat(a)*z for a,z in zip(h,point[:len(lo)])),rat(offsets[j])))
    if not lower_bounds:return {'status':'certified_infeasible','decision':'abstain','proofs':proofs}
    L=min(lower_bounds);U=max(upper_bounds)
    return {'status':'verified_bounds','decision':'exceedance' if L>1 else ('below_or_equal' if U<=1 else 'abstain'),
        'lower':L,'upper':U,'upper_on_minimum':min(upper_on_lower) if upper_on_lower else None,'lower_on_maximum':max(lower_on_upper) if lower_on_upper else None,'proofs':proofs}
