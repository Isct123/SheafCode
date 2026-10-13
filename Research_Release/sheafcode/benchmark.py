                                                                          

                                                                             
                                                                    
   
import numpy as np
from .kalman import KalmanVAR


class StableVAR(KalmanVAR):
    def __init__(self,Ztrain,ridge=10.,r_obs=.05,rank=8):
        T,n,p=Ztrain.shape
        self.n,self.p,self.d=n,p,n*p
        M=Ztrain.reshape(T,-1); mask=np.isfinite(M)
        mu=np.nan_to_num(np.nanmean(M,axis=0))
        X=np.where(mask,M,mu)
                                                                         
                                                                     
        fit=X[::max(1,T//4000)]-mu
        fitmask=mask[::max(1,T//4000)]
        original=fit.copy()
        for _ in range(8):
            _,V=np.linalg.eigh(fit.T@fit)
            basis=V[:,-min(rank,self.d):]
            fit=np.where(fitmask,original,fit@basis@basis.T)
        completed=(X-mu)@basis@basis.T+mu
        X=np.where(mask,M,completed)
                                                                                
        self.A=np.linalg.solve(X[:-1].T@X[:-1]+ridge*np.eye(self.d),X[:-1].T@X[1:]).T
        radius=np.max(np.abs(np.linalg.eigvals(self.A)))
        if radius>.995:
            self.A*=.995/radius
        resid=X[1:]-X[:-1]@self.A.T
        self.Q=np.diag(np.maximum(resid.var(axis=0),1e-3))
        self.R=r_obs; self.mu=mu


def seasonal_predict(Z,times,v,outage_start):
                                                                      
                                                                           
                                                           
       
    out=np.empty((len(Z)-outage_start,Z.shape[2]))
    observed=Z[:outage_start,v]
    for h,t in enumerate(range(outage_start,len(Z))):
        neighbours=np.delete(Z[t],v,axis=0)
        fallback=np.nan_to_num(np.nanmean(neighbours,axis=0))
        for q in range(Z.shape[2]):
            lag=t-24
            if 0<=lag<outage_start and np.isfinite(Z[lag,v,q]):
                out[h,q]=Z[lag,v,q]
            else:
                ix=np.flatnonzero(np.isfinite(observed[:,q]))
                out[h,q]=observed[ix[-1],q] if len(ix) else fallback[q]
    return out
