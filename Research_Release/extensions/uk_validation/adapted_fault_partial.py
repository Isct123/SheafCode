                                                                        

                                                                        
   
import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from faults import Decoder,families
class AvailableDecoder(Decoder):
    def features(self,x):
        eligible=tuple(np.flatnonzero(np.isfinite(x).all(1)));filled=np.nan_to_num(x);s=self.W@(filled.ravel()@self.original.T-self.mean);base=float(s@s);by=[];best=[]
        for k in [0,1,2]:
            vals=[(float((R@s)@(R@s)),F) for F,R in self.supports if len(F)==k and set(F).issubset(eligible)]
            val,F=min(vals) if vals else (base,());by.append(val);best.append(F)
        return np.array([np.log1p(base),np.log1p(by[1]),np.log1p(by[2]),(base-by[1])/(1+base),(by[1]-by[2])/(1+base)]),best,base
def main():
    for region in regions():
        ds=load(region);m=models(region);Z=ds['Z'];n,p=Z.shape[1:];D=m['sheaf'].D;tr=Z[(ds['split']==0)&np.isfinite(Z).all((1,2))];decoder=AvailableDecoder(D,tr,n,p);decoder.original=D;rng=np.random.default_rng(2119);X=[];Y=[]
        candidates=np.isfinite(Z).all(2)
        times=np.flatnonzero((ds['split']==1)&(candidates.sum(1)>=2))[::3]
        for t in times:
            universe=np.flatnonzero(candidates[t])
            for k in [0,1,2]:
                F=tuple(sorted(rng.choice(universe,k,replace=False)));kind=rng.choice(['same','opposite','random','unequal']);y=families(Z[t],Z[max(t-1,0)],ds['mu'],ds['sd'],F,kind,23,rng);X.append(decoder.features(y)[0]);Y.append(k)
        X=np.asarray(X);Y=np.asarray(Y);cut=len(X)//2;best=None;grid=[]
        for ridge in [.1,1,10]:
            trial=AvailableDecoder(D,tr,n,p);trial.fit(D,X[:cut],Y[:cut],ridge);pred=np.argmax(np.c_[np.ones(len(X)-cut),(X[cut:]-trial.fm)/trial.fs]@trial.B,1);acc=float(np.mean(pred==Y[cut:]));grid.append({'ridge':ridge,'count_accuracy':acc})
            if best is None or acc>best[0]:best=(acc,trial,ridge)
        decoder=best[1];cal=np.flatnonzero((ds['split']==3)&(candidates.sum(1)>=2));decoder.cutoff=quantile([decoder.features(Z[t])[2] for t in cal],.05)
        episodes=[]
        for day,ix in blocks(ds,2,history=0,horizon=24):
            always=np.flatnonzero(candidates[ix].all(0)&candidates[max(ix[0]-1,0)])
            if len(always)>=2:episodes.append((day,ix,always))
        rows=[]
        for episode,(day,ix,universe) in enumerate(subset_days(episodes,24)):
            for k in [0,1,2,3]:
                if k>len(universe):continue
                F=tuple(sorted(rng.choice(universe,k,replace=False)))
                for kind in (['clean'] if k==0 else ['same','opposite','random','unequal','drift','stuck','raw_gain','one_channel']):
                    for j,t in enumerate(ix):
                        y=families(Z[t],Z[ix[0]-1],ds['mu'],ds['sd'],F,kind,j,rng);pred,alarm,count,norm=decoder.predict(y)
                        rows.append({'day':day,'episode':episode,'timestamp':str(ds['times'][t]),'kind':kind,'true_count':k,'true_support':','.join(map(str,F)),'predicted_support':','.join(map(str,pred)),'eligible_candidates':int(candidates[t].sum()),'alarm':alarm,'exact':tuple(pred)==F,'count_correct':len(pred)==k,'estimated_count':len(pred),'norm_score':norm})
            frame_save(region+'/available_fault_episode_cases.csv',pd.DataFrame(rows))
                                                                                           
        hours=np.flatnonzero((ds['split']==2)&(candidates.sum(1)>=2))[::6];hourly=[]
        for t in hours:
            universe=np.flatnonzero(candidates[t])
            for k in [0,1,2]:
                F=tuple(sorted(rng.choice(universe,k,replace=False)))
                for kind in (['clean'] if k==0 else ['same','opposite','random','unequal','raw_gain','one_channel']):
                    y=families(Z[t],Z[t],ds['mu'],ds['sd'],F,kind,23,rng);pred,alarm,count,norm=decoder.predict(y);hourly.append({'timestamp':str(ds['times'][t]),'day':str(ds['times'][t].date()),'kind':kind,'true_count':k,'alarm':alarm,'exact':tuple(pred)==F,'count_correct':len(pred)==k,'true_support':','.join(map(str,F)),'predicted_support':','.join(map(str,pred))})
        frame_save(region+'/available_fault_hour_cases.csv',pd.DataFrame(hourly))
        if rows:frame_save(region+'/available_fault_summary.csv',pd.DataFrame(rows).groupby(['kind','true_count']).agg(cases=('exact','size'),alarm_rate=('alarm','mean'),exact_support=('exact','mean'),count_accuracy=('count_correct','mean')).reset_index())
        frame_save(region+'/available_fault_hour_summary.csv',pd.DataFrame(hourly).groupby(['kind','true_count']).agg(cases=('exact','size'),alarm_rate=('alarm','mean'),exact_support=('exact','mean'),count_accuracy=('count_correct','mean')).reset_index())
        save(region+'/available_fault_status.json',{'status':'completed','source_sha256':sha(__file__),'eligible_episodes':len(episodes),'sampled_episodes':min(24,len(episodes)),'episode_rows':len(rows),'hour_rows':len(hourly),'grid':grid,'ridge':best[2],'clean_calibration_cases':len(cal),'clean_cutoff':decoder.cutoff,'candidate_rule':'Every fully observed station at each hour; stable fully observed injected supports in episodes','natural_missing_context':'Training-mean imputation before syndrome; empirical missing-pattern limitations remain','scope':'Available-candidate evaluation, separate from the strict whole-network episode study; no result-driven tuning. Synthetic faults, maximum reported count two.'})
        print(region,'available fault evaluation',len(rows),'episode rows',len(hourly),'hour rows',flush=True)
if __name__=='__main__':main()
