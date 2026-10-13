                                                                                
import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from scipy.special import softmax

class Decoder:
    def __init__(self,Ztrain,D,n,p,choice):
        full=np.isfinite(Ztrain).all((1,2));R=Ztrain[full].reshape(full.sum(),-1)@D.T
        self.mu=R.mean(0);C=np.cov(R,rowvar=False)
        self.C=.8*C+.2*np.diag(np.diag(C))+.001*np.eye(len(D))
        self.D=D;self.n=n;self.p=p;self.choice=choice;self.cache={};self.support=(abs(D)>1e-14).astype(int)

    def predict(self,y):
        observed=np.isfinite(y.ravel());valid=self.support@observed.astype(int)==self.support.sum(1)
        candidates=np.flatnonzero(np.isfinite(y).all(1));key=valid.tobytes()+candidates.tobytes()
        if key not in self.cache:
            L=np.linalg.cholesky(self.C[np.ix_(valid,valid)]);M=np.linalg.solve(L,self.D[valid]);mu=np.linalg.solve(L,self.mu[valid]);G=M.T@M;groups={}
            for k in [1,2]:
                supports=list(itertools.combinations(candidates.tolist(),k))
                if not supports:continue
                cols=np.array([np.concatenate([np.arange(v*self.p,(v+1)*self.p) for v in s]) for s in supports])
                inv=np.linalg.pinv(G[cols[:,:,None],cols[:,None,:]],rcond=1e-10)
                groups[k]=(supports,cols,inv)
            self.cache[key]=(M,mu,groups)
        M,mu,groups=self.cache[key];s=M@np.nan_to_num(y.ravel())-mu;b=M.T@s;gains=[0.,0.];best={0:()}
        for k,(supports,cols,inv) in groups.items():
            B=b[cols];g=np.einsum('bi,bij,bj->b',B,inv,B);j=int(np.argmax(g));gains[k-1]=max(float(g[j]),0);best[k]=tuple(supports[j])
        x=np.log1p([gains[0],gains[1],max(0,gains[1]-gains[0]),float(s@s),len(M),len(groups.get(1,([],))[0])])
        c=self.choice;P=softmax(np.r_[1,(x-np.array(c['mu']))/np.array(c['scale'])]@np.array(c['weights']))
        predicted=0 if 1-P[0]<=c['posterior_cutoff'] else 1+int(P[2]>P[1])
        ls=max([0,1,2],key=lambda k:0 if k==0 else gains[k-1]-k*c['penalty'])
        return {'classifier':best.get(predicted,()),'penalized_ls':best.get(ls,())}

def week_ci(frame):
    groups=[g for _,g in frame.groupby('week')];rng=np.random.default_rng(744)
    arr=np.array([[len(g),g.exact.sum(),g.alarm.sum()] for g in groups])
    boot=[]
    for _ in range(1000):
        a=arr[rng.integers(0,len(arr),len(arr))].sum(0);boot.append(a[1]/a[0])
    return np.quantile(boot,[.025,.975])

def main():
    choices_path=ROOT/'validation_results/fault_choices.json';choices=json.loads(choices_path.read_text())
    lock('persistent_fault_lock.json',{'choices_sha256':sha(choices_path),'source_sha256':sha(__file__),'seed':9105,
         'model_sha256':{r:sha(ROOT/'results_expanded'/r/'model.npz') for r in ['delhi2025','bengaluru_raw']}})
    path=OUT/'persistent_fault_cases.csv';existing=pd.read_csv(path) if path.exists() else pd.DataFrame();rows=existing.to_dict('records')
    done=set(zip(existing.region,existing.episode)) if len(existing) else set();rng=np.random.default_rng(9105)
    for region in ['delhi2025','bengaluru_raw']:
        ds,m,folder=local(region);Z=ds.Z();D=np.load(folder/'model.npz')['delta'];decoder=Decoder(Z[ds.split==0],D,ds.n,ds.p,choices[region])
        blocks=day_blocks(ds.times,ds.split,2)
        blocks=[blocks[i] for i in np.unique(np.linspace(0,len(blocks)-1,min(48,len(blocks))).astype(int))]
        for episode,(day,ix) in enumerate(blocks):
            test=ix[24:];eligible=np.flatnonzero(np.isfinite(Z[test]).all((0,2)))
            if len(eligible)<2:continue
            for count in [0,1,2,3]:
                if count>len(eligible):continue
                F=tuple(sorted(rng.choice(eligible,count,replace=False).tolist()))
                kinds=['clean'] if count==0 else ['same_sign','opposite_sign','unequal','drift','stuck','shared_additive','shared_scale','partial_channel']
                for kind in kinds:
                    shift=rng.normal(size=ds.p);shift/=np.linalg.norm(shift)
                    for j,t in enumerate(test):
                        y=Z[t].copy()
                        for i,v in enumerate(F):
                            if kind=='stuck':y[v]=Z[test[0]-1,v]
                            elif kind=='shared_scale':y[v]=1.5*y[v]
                            elif kind=='shared_additive':y[v]+=2*np.sqrt(ds.p)*shift
                            elif kind=='partial_channel':y[v,0]+=2
                            else:
                                scale=2*(-1 if kind=='opposite_sign' and i%2 else 1)
                                if kind=='unequal' and i:scale*=.15
                                if kind=='drift':scale*=((j+1)/24)
                                y[v]+=scale
                        if not np.isfinite(y[list(F)]).all():continue
                        found=decoder.predict(y)
                        if (region,episode) in done:continue
                        for method,support in found.items():
                            rows.append({'region':region,'episode':episode,'day':day,'timestamp':str(ds.times[t]),'outage_hour':j+1,
                                'fault_type':kind,'fault_count':count,'method':method,'true_support':str(F),'estimated_support':str(support),
                                'alarm':bool(support),'count_correct':len(support)==count,'exact':support==F,
                                'fault_rms_standardized':float(np.sqrt(np.mean((y[list(F)]-Z[t,list(F)])**2))) if F else 0.,
                                'week':str(ds.times[t].to_period('W-SUN'))})
            pd.DataFrame(rows).to_csv(path,index=False)
            print(region,'persistent fault episode',episode,flush=True)
    frame=pd.DataFrame(rows);summary=[]
    for key,g in frame.groupby(['region','fault_type','fault_count','method']):
        ci=week_ci(g);summary.append(dict(zip(['region','fault_type','fault_count','method'],key))|{
            'cases':len(g),'episodes':g.episode.nunique(),'weeks':g.week.nunique(),'alarm_rate':g.alarm.mean(),
            'count_accuracy':g.count_correct.mean(),'exact_end_to_end':g.exact.mean(),'exact_ci_low':ci[0],'exact_ci_high':ci[1]})
    pd.DataFrame(summary).to_csv(OUT/'persistent_fault_summary.csv',index=False)
    save('persistent_fault_scope.json',{'status':'completed','case_rows':len(frame),'detector_maximum_fault_count':2,
        'scope':'Frozen validation-trained classifier and penalized LS; no new tuning. Artificial persistent faults in standardized coordinates, all eligible observed station candidates. Three faults deliberately exceed decoder capacity. Conditional weekly intervals exploratory; natural recordings are not verified fault-free.'})

if __name__=='__main__':main()
