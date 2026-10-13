                                                                            
from fractions import Fraction as Q
import math
import numpy as np
from scipy.optimize import linprog

def rat(x): return Q.from_float(float(x))

def corrected_bound(G,rhs,c,lo,hi,weights):
                                                                                
    weights=[max(Q(0),rat(x)) for x in weights]
    residual=[rat(c[j])-sum((weights[i]*rat(G[i,j]) for i in range(len(rhs))),Q(0)) for j in range(len(c))]
    base=sum((weights[i]*rat(rhs[i]) for i in range(len(rhs))),Q(0))
    correction=sum((max(x*rat(a),x*rat(b)) for x,a,b in zip(residual,lo,hi)),Q(0))
    return base+correction,base,correction,weights

def feasible(G,rhs,lo,hi,x):
    return all(rat(a)<=y<=rat(b) for a,b,y in zip(lo,hi,x)) and all(
        sum((rat(a)*b for a,b in zip(row,x)),Q(0))<=rat(r) for row,r in zip(G,rhs))

def node_model(A,b,blocks,eps,beta,lower,upper,t,fixed):
    m,p=A.shape;n=len(blocks);sizes=[B.shape[1] for B in blocks];q=sum(sizes)
    M=np.c_[A,*blocks];G=np.c_[np.r_[M,-M],np.zeros((2*m,n))]
    rhs=list(np.r_[eps-b,eps+b]);extra=[];start=p
    for v,k in enumerate(sizes):
        for j in range(start,start+k):
            for sign in [-1,1]:
                row=np.zeros(p+q+n);row[j]=sign;row[p+q+v]=-beta;extra.append(row);rhs.append(0.)
        start+=k
    row=np.zeros(p+q+n);row[p+q:]=1;extra.append(row);rhs.append(float(t))
    G=np.r_[G,np.array(extra)];lo=np.r_[lower,np.full(q,-beta),np.zeros(n)];hi=np.r_[upper,np.full(q,beta),np.ones(n)]
    for v,a in fixed.items():
        lo[p+q+v]=hi[p+q+v]=a
        if a==0:
            j=p+sum(sizes[:v]);lo[j:j+sizes[v]]=0.;hi[j:j+sizes[v]]=0.
    return G,np.array(rhs),lo,hi

def exact_point(G,rhs,lo,hi,c,proposal):
    x=[rat(a) for a in proposal]
    if feasible(G,rhs,lo,hi,x):return x,'endpoint'
                                                                            
    free=np.flatnonzero(lo<hi)
                                                                       
    rows=[np.r_[row,float(np.any(row[free]!=0))] for row in G];rr=list(rhs)
    for j in free:
        for sign,lim in [(1.,hi[j]),(-1.,-lo[j])]:
            r=np.zeros(len(c)+1);r[j]=sign;r[-1]=1.;rows.append(r);rr.append(lim)
    fit=linprog(np.r_[np.zeros(len(c)),-1.],A_ub=rows,b_ub=rr,bounds=list(zip(lo,hi))+[(0.,1.)],method='highs')
    if not fit.success:return None,'no_verified_point'
    interior=[rat(a) for a in fit.x[:-1]]
    for weight in [Q(1,10**8),Q(1,10**6),Q(1,10**4),Q(1,100),Q(1)]:
        mixed=[(1-weight)*a+weight*b for a,b in zip(x,interior)]
        if feasible(G,rhs,lo,hi,mixed):return mixed,'interior_mix'
    return None,'no_verified_point'

def solve_node(G,rhs,lo,hi,c,leaf):
    fits=1;fit=linprog(-c,A_ub=G,b_ub=rhs,bounds=list(zip(lo,hi)),method='highs')
    if fit.status==2:
                                                                                      
        cap=1.+max(0.,float(np.max(np.sum(np.maximum(G*lo,G*hi),axis=1)-rhs)))
        GG=np.c_[G,-np.ones(len(G))];ll=np.r_[lo,0.];hh=np.r_[hi,cap];cc=np.r_[np.zeros(len(c)),-1.]
        phase=linprog(-cc,A_ub=GG,b_ub=rhs,bounds=list(zip(ll,hh)),method='highs');fits+=1
        if phase.success:
            ub,base,corr,w=corrected_bound(GG,rhs,cc,ll,hh,-phase.ineqlin.marginals)
            if ub<0:return {'infeasible':True,'phase':True,'G':GG,'rhs':rhs,'c':cc,'lo':ll,'hi':hh,'weights':w,'upper':ub,'base':base,'correction':corr,'lp_solves':fits}
                                                                                     
        ub,base,corr,w=corrected_bound(G,rhs,c,lo,hi,np.zeros(len(rhs)))
        return {'infeasible':False,'upper':ub,'base':base,'correction':corr,'weights':w,'point':None,'G':G,'rhs':rhs,'c':c,'lo':lo,'hi':hi,'lp_solves':fits,'numerical_status':int(fit.status)}
    if not fit.success:raise RuntimeError(fit.message)
    ub,base,corr,w=corrected_bound(G,rhs,c,lo,hi,-fit.ineqlin.marginals)
    point,kind=exact_point(G,rhs,lo,hi,c,fit.x) if leaf else (None,'relaxation_only')
    if leaf and kind!='endpoint':fits+=1
    return {'infeasible':False,'upper':ub,'base':base,'correction':corr,'weights':w,'point':point,'point_kind':kind,'G':G,'rhs':rhs,'c':c,'lo':lo,'hi':hi,'lp_solves':fits,'numerical_upper':float(-fit.fun),'numerical_status':0}

def search(A,b,h,blocks,t,eps,beta,lower,upper,tolerance=1e-7,max_nodes=255):
    n=len(blocks);width=A.shape[1]+sum(B.shape[1] for B in blocks)+n
    c=np.r_[h,np.zeros(width-len(h))];box=Q(0)
    for v,a,z in zip(h,lower,upper):box+=max(rat(v)*rat(a),rat(v)*rat(z))
    active=[{'fixed':{},'bound':box}];settled=[];inc=None;history=[];records=[];nodes=0;solves=0
    def global_upper():return max([x['bound'] for x in active]+settled+([] if inc is None else [inc])) if active or settled or inc is not None else None
    while active and nodes<max_nodes:
        active.sort(key=lambda x:x['bound']);node=active.pop();fixed=node['fixed'];nodes+=1
        G,rhs,lo,hi=node_model(A,b,blocks,eps,beta,lower,upper,t,fixed)
        leaf=len(fixed)==n
        info=solve_node(G,rhs,lo,hi,c,leaf);solves+=info['lp_solves'];info['fixed']=dict(fixed);info['parent_bound']=node['bound'];records.append(info)
        if not info['infeasible']:
            bound=min(node['bound'],info['upper']);point=info.get('point')
            if point is not None:
                value=sum((rat(a)*z for a,z in zip(c,point)),Q(0));inc=value if inc is None else max(inc,value)
            if inc is None or bound>inc:
                if leaf:settled.append(bound)
                else:
                    v=next(i for i in range(n) if i not in fixed)
                                                                                  
                    for a in [0,1]:
                        if a+sum(fixed.values())<=t:active.append({'fixed':{**fixed,v:a},'bound':bound})
        if inc is not None:
            active=[x for x in active if x['bound']>inc];settled=[u for u in settled if u>inc]
        current_upper=global_upper();history.append({'nodes':nodes,'lower':None if inc is None else str(inc),'upper':None if current_upper is None else str(current_upper),'active':len(active),'unclosed_leaves':len(settled)})
        if inc is not None and current_upper-inc<=rat(tolerance):break
    upper=global_upper()
    return {'status':'certified_infeasible' if upper is None else ('verified_gap' if inc is not None and upper-inc<=rat(tolerance) else 'unresolved'),
        'lower':None if inc is None else str(inc),'upper':None if upper is None else str(upper),'nodes':nodes,'lp_solves':solves,'history':history,'records':records,'unclosed_leaves':len(settled),'active_nodes':len(active)}
