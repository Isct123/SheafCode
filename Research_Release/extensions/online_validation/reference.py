                                                                                
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve

def inverse_root(C,ridge=.001):
    w,V=np.linalg.eigh(C+ridge*np.eye(len(C)))
    return (V/np.sqrt(np.maximum(w,1e-12)))@V.T

def fit_pair(a,b,p):
    ok=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    if ok.sum()<200:return np.eye(p),np.eye(p),np.full(p,.3)
    a=a[ok]-a[ok].mean(0);b=b[ok]-b[ok].mean(0)
    wa=inverse_root(a.T@a/(len(a)-1));wb=inverse_root(b.T@b/(len(b)-1))
    U,s,Vt=np.linalg.svd(wa@(a.T@b/(len(a)-1))@wb)
    return U.T@wa,Vt@wb,np.clip(s,0,1)

def graph_correlation(Z,k=3):
    n,p=Z.shape[1:];score=np.zeros((n,n))
    for q in range(p):
        for u in range(n):
            for v in range(u+1,n):
                good=np.isfinite(Z[:,u,q])&np.isfinite(Z[:,v,q])
                c=abs(np.corrcoef(Z[good,u,q],Z[good,v,q])[0,1]) if good.sum()>2 else 0
                score[u,v]+=np.nan_to_num(c)/p;score[v,u]=score[u,v]
    np.fill_diagonal(score,-np.inf)
    return sorted({tuple(sorted((u,int(v)))) for u in range(n) for v in np.argsort(-score[u],kind='stable')[:min(k,n-1)]})

class QuadraticDecoder:
    def __init__(self,Ztrain,edges,lam=1.,temporal_scale=2.):
        _,self.n,self.p=Ztrain.shape;d=self.n*self.p;self.lam=lam
        self.P=np.zeros((d,d));blocks=[];self.temporal={};self.precisions={}
        for u,v in edges:
            a,b,rho=fit_pair(Ztrain[:,u],Ztrain[:,v],self.p)
            r=np.clip(rho,.001,.999);w=np.sqrt(r/(1-r))[:,None]
            row=np.zeros((self.p,d));row[:,u*self.p:(u+1)*self.p]=w*a;row[:,v*self.p:(v+1)*self.p]=-w*b;blocks.append(row)
        self.D=np.vstack(blocks)
        for v in range(self.n):
            full=np.isfinite(Ztrain[:,v]).all(1);a=Ztrain[full,v]
            C=np.cov(a,rowvar=False)+.001*np.eye(self.p) if len(a)>self.p+2 else np.eye(self.p)
            self.P[v*self.p:(v+1)*self.p,v*self.p:(v+1)*self.p]=np.linalg.inv(C)
        self.S=self.D.T@self.D+lam*self.P
        for lag in (1,24):
            self.temporal[lag]=[]
            for v in range(self.n):
                a,b,rho=fit_pair(Ztrain[:-lag,v],Ztrain[lag:,v],self.p)
                                                                                             
                full=np.isfinite(Ztrain[:-lag,v]).all(1)&np.isfinite(Ztrain[lag:,v]).all(1)
                if full.sum()<200:rho=np.full(self.p,.5)
                r=np.clip(rho,.001,.999);w=np.sqrt(temporal_scale*r/(1-r))[:,None]
                self.temporal[lag].append((w*a,-w*b))

    def precision(self,W):
        if W in self.precisions:return self.precisions[W]
        d=self.n*self.p;Q=sp.kron(sp.eye(W),sp.csr_matrix(self.S),format='csr')
        rr=[];cc=[];vv=[]
        for lag,blocks in self.temporal.items():
            for v,(a,b) in enumerate(blocks):
                C=np.block([[a.T@a,a.T@b],[b.T@a,b.T@b]])
                for t in range(max(0,W-lag)):
                    indices=np.r_[np.arange(v*self.p,(v+1)*self.p)+t*d,np.arange(v*self.p,(v+1)*self.p)+(t+lag)*d]
                    rr.extend(np.repeat(indices,len(indices)));cc.extend(np.tile(indices,len(indices)));vv.extend(C.ravel())
        Q+=sp.csr_matrix((vv,(rr,cc)),shape=Q.shape)
        self.precisions[W]=Q
        return Q

    def solve(self,Z):
        shape=Z.shape;y=Z.ravel().copy();U=np.flatnonzero(~np.isfinite(y));O=np.flatnonzero(np.isfinite(y))
        Q=self.precision(len(Z));y[U]=spsolve(Q[U][:,U].tocsc(),-Q[U][:,O]@y[O]) if len(U) else np.array([])
        return y.reshape(shape)

    def causal(self,Z,v,start=24):
        hidden=Z.copy();hidden[start:,v]=np.nan
        return np.array([self.solve(hidden[:j+1])[-1,v] for j in range(start,len(hidden))])

def complete_training(Z,rank=8):
    M=Z.reshape(len(Z),-1);mask=np.isfinite(M);mu=np.nan_to_num(np.nanmean(M,axis=0))
    X=np.where(mask,M,mu);stride=max(1,len(M)//4000);fit=X[::stride]-mu;fm=mask[::stride];original=fit.copy()
    for _ in range(8):
        _,V=np.linalg.eigh(fit.T@fit);basis=V[:,-min(rank,M.shape[1]):];fit=np.where(fm,original,fit@basis@basis.T)
    return np.where(mask,M,(X-mu)@basis@basis.T+mu)

class FullVAR:
    def __init__(self,X,n,p,ridge=1.,noise=.05,shrink=0.):
        self.n,self.p=n,p;d=n*p;self.R=noise
        self.A=np.linalg.solve(X[:-1].T@X[:-1]+ridge*np.eye(d),X[:-1].T@X[1:]).T
        radius=max(abs(np.linalg.eigvals(self.A)))
        if radius>.995:self.A*=.995/radius
        e=X[1:]-X[:-1]@self.A.T;C=np.cov(e,rowvar=False)
        self.Q=(1-shrink)*C+shrink*np.diag(np.diag(C))+.001*np.eye(d)

    def filter(self,Z):
        d=self.n*self.p;x=np.zeros(d);P=np.eye(d);out=[]
        for t,y in enumerate(Z.reshape(len(Z),d)):
            if t:x=self.A@x;P=self.A@P@self.A.T+self.Q
            O=np.flatnonzero(np.isfinite(y))
            if len(O):
                K=solve(P[np.ix_(O,O)]+self.R*np.eye(len(O)),P[O],assume_a='pos').T
                x+=K@(y[O]-x[O]);P-=K@P[O]
                                                                           
            P=(P+P.T)/2;estimate=x.copy();estimate[O]=y[O];out.append(estimate)
        return np.array(out).reshape(Z.shape)

def seasonal(Z,v,start=24):
    out=np.empty((len(Z)-start,Z.shape[2]))
    for j,t in enumerate(range(start,len(Z))):
        for q in range(Z.shape[2]):
            if 0<=t-24<start and np.isfinite(Z[t-24,v,q]):out[j,q]=Z[t-24,v,q]
            else:
                past=np.flatnonzero(np.isfinite(Z[:start,v,q]))
                other=np.delete(Z[t,:,q],v)
                out[j,q]=Z[past[-1],v,q] if len(past) else (np.nanmean(other) if np.isfinite(other).any() else 0.)
    return out
