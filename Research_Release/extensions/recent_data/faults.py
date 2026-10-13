import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from scipy.optimize import minimize
def families(x,raw,mu,sd,F,kind,j,rng):
    y=x.copy();p=x.shape[1]
    for k,v in enumerate(F):
        if kind=='stuck':y[v]=raw[v]
        elif kind=='raw_gain':y[v]=1.5*x[v]+.5*mu[v]/sd[v]
        elif kind=='one_channel':y[v,0]+=2
        else:
            direction=np.ones(p)
            if kind=='opposite' and k%2:direction=-direction
            if kind=='random':direction=rng.choice([-1.,1.],p)
            amp=2*(.1 if kind=='unequal' and k==len(F)-1 and len(F)>1 else 1.)
            if kind=='drift':amp*=j/23
            y[v]+=amp*direction
    return y
class Decoder:
    def __init__(self,D,train,n,p):
        self.n=n;self.p=p;s=train.reshape(len(train),-1)@D.T;self.mean=s.mean(0);C=np.cov(s,rowvar=False);w,V=np.linalg.eigh(C+.01*np.eye(len(D)));self.W=(V/np.sqrt(np.maximum(w,1e-10)))@V.T;self.D=self.W@D;self.supports=[]
        for k in [0,1,2]:
            for F in itertools.combinations(range(n),k):
                A=self.D[:,np.concatenate([np.arange(v*p,(v+1)*p) for v in F]).astype(int)] if F else np.zeros((len(D),0))
                R=np.eye(len(D))-A@np.linalg.pinv(A);self.supports.append((F,R))
    def features(self,x):
        s=self.W@(x.ravel()@self.original.T-self.mean);base=float(s@s);by=[];best=[]
        for k in [0,1,2]:
            vals=[(float((R@s)@(R@s)),F) for F,R in self.supports if len(F)==k];val,F=min(vals);by.append(val);best.append(F)
        a=np.array([np.log1p(base),np.log1p(by[1]),np.log1p(by[2]),(base-by[1])/(1+base),(by[1]-by[2])/(1+base)])
        return a,best,base
    def fit(self,D,X,Y,ridge):
        self.original=D;self.fm=X.mean(0);self.fs=np.maximum(X.std(0),.01);A=np.c_[np.ones(len(X)),(X-self.fm)/self.fs];one=np.eye(3)[Y]
        def obj(flat):
            B=flat.reshape(6,3);scores=A@B;scores-=scores.max(1,keepdims=True);P=np.exp(scores);P/=P.sum(1,keepdims=True);loss=-np.sum(one*np.log(P+1e-15))/len(A)+ridge*np.sum(B[1:]**2)/(2*len(A));grad=A.T@(P-one)/len(A);grad[1:]+=ridge*B[1:]/len(A);return loss,grad.ravel()
        self.B=minimize(obj,np.zeros(18),jac=True,method='L-BFGS-B').x.reshape(6,3)
    def predict(self,x):
        a,F,base=self.features(x);score=np.r_[1,(a-self.fm)/self.fs]@self.B;k=int(np.argmax(score));alarm=base>self.cutoff and k>0
        return F[k] if alarm else (),alarm,k,base
def main():
    for region in regions():
        ds=load(region);m=models(region);Z=ds['Z'];n,p=Z.shape[1:];D=m['sheaf'].D;full=np.isfinite(Z).all((1,2));train=Z[(ds['split']==0)&full]
        if len(train)<50:save(region+'/fault_status.json',{'status':'insufficient_full_training_vectors','count':len(train)});continue
        decoder=Decoder(D,train,n,p);decoder.original=D;rng=np.random.default_rng(1381);valid=np.flatnonzero((ds['split']==1)&full);valid=valid[::max(1,len(valid)//400)];X=[];Y=[]
        for t in valid:
            for k in [0,1,2]:
                F=tuple(sorted(rng.choice(n,k,replace=False)));kind=rng.choice(['same','opposite','random','unequal']);y=families(Z[t],Z[t-1],ds['mu'],ds['sd'],F,kind,23,rng);X.append(decoder.features(y)[0]);Y.append(k)
        X=np.asarray(X);Y=np.asarray(Y);cut=len(X)//2;choices=[];best=None
        for ridge in [.1,1.,10.]:
            candidate=Decoder(D,train,n,p);candidate.fit(D,X[:cut],Y[:cut],ridge);pred=np.argmax(np.c_[np.ones(len(X)-cut),(X[cut:]-candidate.fm)/candidate.fs]@candidate.B,1);accuracy=float(np.mean(pred==Y[cut:]));choices.append({'ridge':ridge,'selection_accuracy':accuracy})
            if best is None or accuracy>best[0]:best=(accuracy,candidate,ridge)
        decoder=best[1];cal=np.flatnonzero((ds['split']==3)&full);decoder.cutoff=quantile([decoder.features(Z[t])[2] for t in cal],.05)
        save(region+'/fault_choices.json',{'ridge':best[2],'grid':choices,'clean_calibration_cases':len(cal),'clean_cutoff':decoder.cutoff,'maximum_count':2,'source_sha256':sha(__file__),'scope':'Train-whitened syndrome; count learned on first validation half, clean cutoff on second half'})
        rows=[];starts=[]
        for day,ix in blocks(ds,2,history=0,horizon=24):
            if np.all(full[ix]) and ix[0]>0 and full[ix[0]-1]:starts.append((day,ix))
        for episode,(day,ix) in enumerate(subset_days(starts,24)):
            for k in [0,1,2,3]:
                if k>n:continue
                F=tuple(sorted(rng.choice(n,k,replace=False)))
                for kind in (['clean'] if k==0 else ['same','opposite','random','unequal','drift','stuck','raw_gain','one_channel']):
                    for j,t in enumerate(ix):
                        y=families(Z[t],Z[ix[0]-1],ds['mu'],ds['sd'],F,kind,j,rng);pred,alarm,count,base=decoder.predict(y)
                        rows.append({'day':day,'episode':episode,'timestamp':str(ds['times'][t]),'kind':kind,'true_count':k,'true_support':','.join(map(str,F)),'predicted_support':','.join(map(str,pred)),'alarm':alarm,'exact':tuple(pred)==F,'estimated_count':len(pred),'norm_score':base})
            frame_save(region+'/fault_episode_cases.csv',pd.DataFrame(rows));print(region,'fault episode',episode+1,'/',min(24,len(starts)),flush=True)
        f=pd.DataFrame(rows)
        if len(f):frame_save(region+'/fault_summary.csv',f.groupby(['kind','true_count']).agg(cases=('exact','size'),alarms=('alarm','mean'),exact_support=('exact','mean'),count_accuracy=('estimated_count',lambda x:0.)).reset_index().drop(columns='count_accuracy'))
        save(region+'/fault_status.json',{'status':'completed' if rows else 'no_fully_observed_test_episodes','eligible_episodes':len(starts),'sampled_episodes':min(24,len(starts)),'case_rows':len(rows),'scope':'Synthetic faults on recorded observations; no verified field-fault ground truth'})
if __name__=='__main__':main()
