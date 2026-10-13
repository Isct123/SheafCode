                                                                         

                                                                          
   
import numpy as np
from scipy.optimize import linprog

def upper_certificate(A,b,h,blocks,t,epsilon,beta,lower=None,upper=None):
    A=np.asarray(A,float);b=np.asarray(b,float);h=np.asarray(h,float);m,p=A.shape;n=len(blocks)
    if b.shape!=(m,) or h.shape!=(p,) or not isinstance(t,int) or not 0<=t<=n:raise ValueError('Invalid dimensions/budget')
    eps=np.broadcast_to(np.asarray(epsilon,float),(m,))
    if (eps<0).any() or beta<0 or not np.isfinite(eps).all() or not np.isfinite(beta):raise ValueError('Invalid bounds')
    C=[];d=[]
    if lower is not None:
        for j,value in enumerate(lower):
            if np.isfinite(value):C.append(-np.eye(p)[j]);d.append(-value)
    if upper is not None:
        for j,value in enumerate(upper):
            if np.isfinite(value):C.append(np.eye(p)[j]);d.append(value)
    C=np.asarray(C).reshape(-1,p);d=np.asarray(d);sizes=[np.asarray(B).shape[1] for B in blocks];q=sum(sizes);k=len(d)
                                                            
                                                                       
    qa=2*m;tau=qa+q;ua=tau+1;la=ua+n;N=la+k
    cost=np.zeros(N);cost[:m]=-b;cost[m:2*m]=eps;cost[tau]=beta*t;cost[ua:ua+n]=beta;cost[la:]=d
    G=[];rhs=[]
    for j in range(m):
        for sign in [-1,1]:
            row=np.zeros(N);row[j]=sign;row[m+j]=-1;G.append(row);rhs.append(0.)
    offset=0
    for j,B in enumerate(blocks):
        B=np.asarray(B,float)
        if B.shape[0]!=m:raise ValueError('Invalid nuisance block')
        for col in range(B.shape[1]):
            for sign in [-1,1]:
                row=np.zeros(N);row[:m]=sign*B[:,col];row[qa+offset+col]=-1;G.append(row);rhs.append(0.)
        row=np.zeros(N);row[qa+offset:qa+offset+B.shape[1]]=1;row[tau]=-1;row[ua+j]=-1;G.append(row);rhs.append(0.)
        offset+=B.shape[1]
    equality=np.zeros((p,N));equality[:,:m]=A.T;equality[:,la:]=C.T
    fit=linprog(cost,A_ub=np.asarray(G),b_ub=rhs,A_eq=equality,b_eq=h,bounds=[(None,None)]*m+[(0,None)]*(N-m),method='highs')
    if not fit.success:return {'status':'no_finite_certificate','solver_status':fit.status}
    mismatch=np.linalg.norm(equality@fit.x-h,np.inf)
    if mismatch>1e-7:raise RuntimeError('Dual equality feasibility failed')
                                                                        
    w=fit.x[:m];lam=fit.x[la:];leverage=sorted([float(np.sum(abs(np.asarray(B).T@w))) for B in blocks],reverse=True)
    value=float(-b@w+eps@abs(w)+beta*sum(leverage[:t])+d@lam)
    return {'status':'finite','upper':value,'equality_residual':float(mismatch),'w':w.tolist(),'station_leverage_sorted':leverage,
        'scope':'Numerical feasible dual outer bound; not asserted equal to the exact sparse-support envelope'}

def dual_envelope(A,b,h,blocks,t,epsilon,beta,lower=None,upper=None):
    high=upper_certificate(A,b,h,blocks,t,epsilon,beta,lower,upper)
    low=upper_certificate(A,b,-np.asarray(h),blocks,t,epsilon,beta,lower,upper)
    return {'status':'bounded' if high['status']==low['status']=='finite' else 'unbounded',
        'lower':-low['upper'] if low['status']=='finite' else -np.inf,'upper':high['upper'] if high['status']=='finite' else np.inf,
        'lower_certificate':low,'upper_certificate':high}
