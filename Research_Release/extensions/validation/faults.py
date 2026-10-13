                                                                                  
import sys,itertools,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from scipy.optimize import minimize
from scipy.special import softmax
def fit_classifier(X,y,ridge):
    mu=X.mean(0);sd=np.maximum(X.std(0),1e-6);V=np.c_[np.ones(len(X)),(X-mu)/sd]
    def objective(w):
        W=w.reshape(V.shape[1],3);P=softmax(V@W,axis=1);loss=-np.log(np.maximum(P[np.arange(len(y)),y],1e-12)).mean()+ridge*np.sum(W[1:]**2)/len(y)
        P[np.arange(len(y)),y]-=1;grad=V.T@P/len(y);grad[1:]+=2*ridge*W[1:]/len(y)
        return loss,grad.ravel()
    fit=minimize(objective,np.zeros(V.shape[1]*3),jac=True,method='L-BFGS-B',options={'maxiter':500})
    if not fit.success and np.linalg.norm(fit.jac)>1e-4:raise RuntimeError(fit.message)
    return mu,sd,fit.x.reshape(V.shape[1],3)
def probabilities(X,model):mu,sd,W=model;return softmax(np.c_[np.ones(len(X)),(X-mu)/sd]@W,axis=1)
def main():
    rng=np.random.default_rng(850);allrows=[];streamrows=[];settings={};start=time.time()
    for region in REGIONS:
        ds,m,folder=cohort(region);ds.set_transform(m['choices']['sheaf']['transform']);Z=ds.Z();D=np.load(folder/'model.npz')['delta'];n,p=ds.n,ds.p
        complete=np.isfinite(Z).all(axis=(1,2));tr=np.flatnonzero((ds.split==0)&complete)
        residual=Z[tr].reshape(len(tr),-1)@D.T;mu=residual.mean(0);C=np.cov(residual,rowvar=False);C=.8*C+.2*np.diag(np.diag(C))+1e-3*np.eye(len(D));cache={}
        support=(abs(D)>1e-14).astype(int)
        def decode(x):
            observed=np.isfinite(x.ravel());valid=support@observed.astype(int)==support.sum(1);cand=np.flatnonzero(np.isfinite(x).all(1));key=valid.tobytes()+cand.tobytes()
            if key not in cache:
                L=np.linalg.cholesky(C[np.ix_(valid,valid)]);M=np.linalg.solve(L,D[valid]);baseline=np.linalg.solve(L,mu[valid]);G=M.T@M;groups={}
                for k in [1,2]:
                    S=list(itertools.combinations(cand.tolist(),k))
                    if not S:continue
                    cols=np.array([np.concatenate([np.arange(v*p,(v+1)*p) for v in s]) for s in S]);inv=np.linalg.pinv(G[cols[:,:,None],cols[:,None,:]],rcond=1e-10);groups[k]=(S,cols,inv)
                cache[key]=(M,baseline,groups)
            M,baseline,groups=cache[key];s=M@np.nan_to_num(x.ravel())-baseline;b=M.T@s;gains=[0.,0.];best={0:()}
            for k,(S,cols,inv) in groups.items():
                B=b[cols];g=np.einsum('bi,bij,bj->b',B,inv,B);j=int(np.argmax(g));gains[k-1]=max(float(g[j]),0);best[k]=tuple(S[j])
            feature=np.log1p([gains[0],gains[1],max(0,gains[1]-gains[0]),float(s@s),len(M),len(groups.get(1,([],))[0])])
            return feature,best,gains
        def sample(rows,count):return rows[np.unique(np.linspace(0,len(rows)-1,min(count,len(rows))).astype(int))]
        va=ds.rows(1);parts=np.array_split(va,3)
        def examples(rows,count):
            ans=[]
            for t in sample(rows,count):
                fully=np.flatnonzero(np.isfinite(Z[t]).all(1))
                if len(fully)<2:continue
                for k in [0,1,2]:
                    configs=[('clean',0)] if k==0 else [(d,a) for d in ['positive','negative','random','unequal','gradual'] for a in [.5,1.,2.,3.]]
                    for direction,a in configs:
                        F=tuple(sorted(rng.choice(fully,k,replace=False).tolist()));y=Z[t].copy()
                        for j,v in enumerate(F):
                            shift=rng.normal(size=p) if direction=='random' else np.ones(p)*(-1 if direction=='negative' else 1)
                            shift/=np.linalg.norm(shift);scale=a*(.15 if direction=='unequal' and j==1 else 1)
                            if direction=='gradual':scale*=((int(t)%12)+1)/12
                            y[v]+=shift*np.sqrt(p)*scale
                        X,best,gains=decode(y);ans.append((X,k,F,best,gains,int(t),direction,a))
            return ans
        train=examples(parts[0],100);val=examples(parts[1],80);X=np.array([c[0] for c in train]);labels=np.array([c[1] for c in train]);VX=np.array([c[0] for c in val]);vy=np.array([c[1] for c in val])
        candidates=[]
        for ridge in [.01,1.,10.]:
            model=fit_classifier(X,labels,ridge);P=probabilities(VX,model);loss=-np.log(np.maximum(P[np.arange(len(vy)),vy],1e-12)).mean();candidates.append((loss,ridge,model))
        loss,ridge,model=min(candidates,key=lambda c:c[0]);cal=[];lscal=[]
        for t in sample(parts[2],240):
            if np.isfinite(Z[t]).all(1).sum()<2:continue
            x,best,g=decode(Z[t]);cal.append(1-probabilities(x[None],model)[0,0]);lscal.append(max(g[0],g[1]/2))
        k=int(np.ceil((len(cal)+1)*.95));cut=sorted(cal)[min(k-1,len(cal)-1)];penalty=sorted(lscal)[min(k-1,len(cal)-1)]
        settings[region]={'ridge':ridge,'validation_logloss':loss,'calibration_hours':len(cal),'posterior_cutoff':cut,'penalty':penalty,'training_cases':len(train),'mu':model[0].tolist(),'scale':model[1].tolist(),'weights':model[2].tolist(),'model_sha256':digest(folder/'model.npz')};save('fault_choices.json',settings)
        for X,true,F,best,g,t,d,a in examples(ds.rows(2),360):
            P=probabilities(X[None],model)[0];pred=0 if 1-P[0]<=cut else 1+int(P[2]>P[1]);base=max([0,1,2],key=lambda k:0 if k==0 else g[k-1]-k*penalty)
            for method,count in [('calibrated_classifier',pred),('full_covariance_penalized_ls',base)]:
                found=best.get(count,());allrows.append({'region':region,'day':str(ds.times[t].date()),'timestamp':str(ds.times[t]),'direction':d,'magnitude':a,'fault_count':true,'method':method,'alarm':bool(count),'exact':found==F,'count_correct':count==true,'fault_support':str(F),'predicted_support':str(found)})
                                                                                
                                                                    
        test=ds.rows(2)
        for episode,t0 in enumerate(sample(test[:-12],60)):
            ids=np.arange(t0,t0+12);fully=np.flatnonzero(np.isfinite(Z[ids]).all(axis=(0,2)))
            if len(fully)<2 or not (ds.split[ids]==2).all():continue
            F=tuple(sorted(rng.choice(fully,2,replace=False).tolist()));directions=rng.normal(size=(2,p));directions/=np.linalg.norm(directions,axis=1)[:,None]
            for j,t in enumerate(ids):
                magnitude=3*j/11;y=Z[t].copy()
                for v,shift in zip(F,directions):y[v]+=shift*np.sqrt(p)*magnitude
                X,best,g=decode(y);P=probabilities(X[None],model)[0];k=0 if 1-P[0]<=cut else 1+int(P[2]>P[1]);found=best.get(k,())
                streamrows.append({'region':region,'episode':episode,'timestamp':str(ds.times[t]),'ramp_hour':j,'magnitude':magnitude,'fault_count':0 if j==0 else 2,'alarm':bool(k),'exact':found==(() if j==0 else F),'fault_support':str(F),'predicted_support':str(found)})
        print(region,'fault extension complete',round(time.time()-start),flush=True)
    f=pd.DataFrame(allrows);f.to_csv(OUT/'fault_cases.csv',index=False)
    pd.DataFrame(streamrows).to_csv(OUT/'fault_ramp_episodes.csv',index=False)
    summary=f.groupby(['region','fault_count','direction','magnitude','method']).agg(cases=('exact','size'),alarm_rate=('alarm','mean'),exact_end_to_end=('exact','mean'),count_accuracy=('count_correct','mean')).reset_index();summary.to_csv(OUT/'fault_summary.csv',index=False)
    comparisons=[]
    for key,g in f.groupby(['region','fault_count','direction','magnitude']):
        pvt=g.pivot(index=['timestamp','day'],columns='method',values='exact').dropna().reset_index();pvt['week']=pd.to_datetime(pvt.day).dt.to_period('W-SUN').astype(str)
        groups=[np.array([len(w),w.calibrated_classifier.sum(),w.full_covariance_penalized_ls.sum()],float) for _,w in pvt.groupby('week')];arr=np.array(groups);tot=arr.sum(0);draw=[]
        for _ in range(500):
            v=arr[rng.integers(0,len(arr),len(arr))].sum(0);draw.append((v[1]-v[2])/v[0])
        ci=np.quantile(draw,[.025,.975]);comparisons.append({'region':key[0],'fault_count':key[1],'direction':key[2],'magnitude':key[3],'weeks':len(arr),'exact_accuracy_difference':(tot[1]-tot[2])/tot[0],'ci_low':ci[0],'ci_high':ci[1]})
    pd.DataFrame(comparisons).to_csv(OUT/'fault_paired_week_intervals.csv',index=False)
    save('fault_scope.json',{'source_sha256':digest(__file__),'scope':'Unknown-count classifier trained on synthetic validation faults; ridge selected on middle validation segment; clean alarm cutoff calibrated on final segment; test targets used only to score; full-covariance syndrome whitening fitted on complete training vectors. All fully observed stations searched. Separate contiguous 12h ramps use fixed supports/directions with no label feedback; detector remains stateless, not a trained change-point model. Natural clean recordings may contain unidentified actual sensor faults. Examined archives, artificial faults, exploratory intervals not familywise adjusted; no field validation.'})
if __name__=='__main__':main()
