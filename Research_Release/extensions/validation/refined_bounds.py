                                                                            

                                                               
   
import heapq,itertools
import numpy as np
from scipy.optimize import linprog
def maximize(A,b,h,blocks,t,eps,beta,lower,upper,max_nodes=128,tolerance=1e-8):
    A=np.asarray(A,float);b=np.asarray(b,float);h=np.asarray(h,float)
    m,p=A.shape;n=len(blocks);sizes=[B.shape[1] for B in blocks];q=sum(sizes)
    if not 0<=t<=n or max_nodes<1:raise ValueError('Invalid budget')
    eps=np.broadcast_to(eps,(m,));lo=np.asarray(lower,float);hi=np.asarray(upper,float)
    if not np.isfinite(np.r_[A.ravel(),b,h,eps,beta,lo,hi]).all() or beta<0 or (eps<0).any() or (lo>hi).any():raise ValueError('Finite valid bounds required')
    B=np.concatenate(blocks,axis=1) if blocks else np.empty((m,0));M=np.c_[A,B,np.zeros((m,n))]
    G=[*M,*(-M)];rhs=[*(eps-b),*(eps+b)];offset=0
    for j,size in enumerate(sizes):
        for k in range(size):
            for sign in [-1,1]:
                row=np.zeros(p+q+n);row[p+offset+k]=sign;row[p+q+j]=-beta;G.append(row);rhs.append(0.)
        offset+=size
    row=np.zeros(p+q+n);row[p+q:]=1;G.append(row);rhs.append(t)
    G=np.asarray(G);rhs=np.asarray(rhs);cost=np.r_[-h,np.zeros(q+n)];counter=itertools.count();nodes=0
    def solve(fixed):
        nonlocal nodes
        nodes+=1
        bounds=list(zip(lo,hi))+[(None,None)]*q+[((v,v) if j in fixed else (0.,1.)) for j in range(n) for v in [fixed.get(j,0)]]
        r=linprog(cost,A_ub=G,b_ub=rhs,bounds=bounds,method='highs')
        if r.status==2:return None
        if not r.success:raise RuntimeError(r.message)
        if np.max(G@r.x-rhs)>1e-6:raise RuntimeError('Primal violation')
        return float(-r.fun),r.x
    root=solve({})
    if root is None:return {'status':'inconsistent','nodes':nodes}
    queue=[(-root[0],next(counter),{},root[1])];inc=-np.inf;best=None;trace=[]
    while queue:
        ub=-queue[0][0]
        if inc>=ub-tolerance:break
        neg,_,fixed,x=heapq.heappop(queue);alpha=x[p+q:]
        e=x[p:p+q];norms=[];offset=0
        for size in sizes:norms.append(np.max(abs(e[offset:offset+size]),initial=0));offset+=size
                                                                        
        if np.sum(np.asarray(norms)>1e-7)<=t:
            val=float(h@x[:p])
            if val>inc:inc=val;best=x[:p].tolist()
            continue
        free=[j for j in range(n) if j not in fixed and 1e-8<alpha[j]<1-1e-8]
        if not free:
                                                                    
            val=float(h@x[:p])
            if val>inc:inc=val;best=x[:p].tolist()
            continue
        if nodes+2>max_nodes:
            heapq.heappush(queue,(neg,next(counter),fixed,x));break
        j=min(free,key=lambda j:abs(alpha[j]-.5))
        for value in [0,1]:
            child={**fixed,j:value}
            if sum(child.values())>t:continue
            r=solve(child)
            if r is not None and r[0]>inc+tolerance:heapq.heappush(queue,(-r[0],next(counter),child,r[1]))
        trace.append({'nodes':nodes,'upper':max(inc,-queue[0][0] if queue else inc),'incumbent':inc})
    bound=max(inc,-queue[0][0] if queue else inc)
    return {'status':'bounded','upper':bound,'incumbent':inc,'gap':bound-inc,'nodes':nodes,'root_upper':root[0],'attaining_state':best,'trace':trace,'complete':not queue or bound-inc<=tolerance}
def envelope(A,b,h,blocks,t,eps,beta,lower,upper,max_nodes=128):
    U=maximize(A,b,h,blocks,t,eps,beta,lower,upper,max_nodes)
    L=maximize(A,b,-np.asarray(h),blocks,t,eps,beta,lower,upper,max_nodes)
    if U['status']=='inconsistent' or L['status']=='inconsistent':return {'status':'inconsistent'}
    return {'status':'bounded','lower':-L['upper'],'upper':U['upper'],'lower_search':L,'upper_search':U}
