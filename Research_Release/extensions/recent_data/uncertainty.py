import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def snapshot_stream(ds,m,indices,loss,seed):
    Z=ds['Z'];n,p=Z.shape[1:];rng=np.random.default_rng(seed);rows=[]
    for t in indices:
        for v,station in enumerate(ds['stations']):
            if not np.isfinite(Z[t,v,0]):continue
            x=Z[t:t+1].copy();x[:,v]=np.nan;drop=rng.random((n,p))<loss;drop[v]=False;x[:,drop]=np.nan
            pred=static_solve(x,m['sheaf'].S)[0,v];city=raw_city(x,v,ds['mu'],ds['sd'])[0];raw=pred[0]*ds['sd'][v,0]+ds['mu'][v,0];true=ds['X'][t,v,0];available=np.isfinite(x).sum()/((n-1)*p)
            U=np.flatnonzero(~np.isfinite(x.ravel()));O=np.flatnonzero(np.isfinite(x.ravel()));Q=m['sheaf'].S;target=v*p;position=int(np.flatnonzero(U==target)[0]);gain=np.linalg.norm(np.linalg.solve(Q[np.ix_(U,U)],Q[np.ix_(U,O)])[position])*ds['sd'][v,0]
            rows.append({'t':t,'station_index':v,'station':str(station),'timestamp':str(ds['times'][t]),'day':str(ds['times'][t].date()),'truth':true,'prediction':raw,'city_prediction':city[0]*ds['sd'][v,0]+ds['mu'][v,0],'availability':available,'pre_outcome_scale':max(gain,1e-3)})
    return pd.DataFrame(rows)
def main():
    for region in regions():
        ds=load(region);m=models(region);lock(region+'/uncertainty_lock.json',{'source_sha256':sha(__file__),'model_sha256':sha(OUT/region/'models.pkl'),'calibration':'second validation half only','feedback':'no test-label updates'})
        threshold=np.nanquantile(ds['X'][ds['split']==0,:,0],.95,axis=0);summaries=[];case_rows=[];calframes=[]
        for loss in [0.,.2,.5]:
            cal=snapshot_stream(ds,m,np.flatnonzero(ds['split']==3),loss,7501);cal['loss']=loss;calframes.append(cal)
            test=snapshot_stream(ds,m,np.flatnonzero(ds['split']==2),loss,7601);test['extreme']=[r.truth>threshold[int(r.station_index)] for r in test.itertuples()]
            cal['abs_error']=abs(cal.truth-cal.prediction);pooled=quantile(cal.abs_error)
            for method in ['pooled_fixed','station_fixed','availability_scaled']:
                f=test.copy();widths=[]
                for r in f.itertuples():
                    cc=cal[cal.station==r.station]
                    radius=pooled if method=='pooled_fixed' else quantile(cc.abs_error/(cc.pre_outcome_scale if method=='availability_scaled' else 1.))*(r.pre_outcome_scale if method=='availability_scaled' else 1.)
                    widths.append(radius)
                f['method']=method;f['context_loss']=loss;f['lower']=f.prediction-widths;f['upper']=f.prediction+widths;f['covered']=(f.truth>=f.lower)&(f.truth<=f.upper);f['width']=2*np.array(widths);f['full_set']=~np.isfinite(f.width);f['decided']=(f.lower>60)|(f.upper<=60);f['correct_when_decided']=((f.lower>60)==(f.truth>60));case_rows.append(f)
                for group,g in [('all',f),('extreme',f[f.extreme]),('nonextreme',f[~f.extreme])]+[(str(q),g) for q,g in f.groupby(pd.to_datetime(f.timestamp).dt.to_period('Q'))]:
                    summaries.append({'region':region,'context_loss':loss,'method':method,'group':group,'cases':len(g),'coverage':g.covered.mean() if len(g) else None,'finite_mean_width':g.loc[~g.full_set,'width'].mean() if len(g) else None,'full_set_fraction':g.full_set.mean() if len(g) else None,'decided_fraction':g.decided.mean() if len(g) else None,'accuracy_when_decided':g.loc[g.decided,'correct_when_decided'].mean() if g.decided.any() else None})
            print(region,'no-feedback intervals',loss,'cal',len(cal),'test',len(test),flush=True)
        f=pd.concat(case_rows,ignore_index=True);frame_save(region+'/uncertainty_cases.csv',f);frame_save(region+'/interval_calibration_predictions.csv',pd.concat(calframes,ignore_index=True));frame_save(region+'/uncertainty_summary.csv',pd.DataFrame(summaries))
                                                                                     
        adaptive=[];base=pd.concat(case_rows);base=base[(base.method=='station_fixed')&(base.context_loss==0)]
        cal0=calframes[0]
        for station,g in base.groupby('station',sort=False):
            hist=list(cal0[cal0.station==station].abs_error);level=.1
            for r in g.sort_values('timestamp').itertuples():
                radius=float('inf') if level<=0 else (-float('inf') if level>=1 else quantile(hist[-500:],level));covered=radius>=0 and abs(r.truth-r.prediction)<=radius;err=int(not covered)
                adaptive.append({'station':station,'timestamp':r.timestamp,'level_before':level,'radius':radius,'covered':covered,'full_set':radius==float('inf'),'empty_set':radius==-float('inf'),'feedback':'target revealed only after interval construction'});level+=.01*(.1-err);hist.append(abs(r.truth-r.prediction))
        frame_save(region+'/revealed_adaptive_cases.csv',pd.DataFrame(adaptive));save(region+'/uncertainty_status.json',{'status':'completed','no_feedback_rows':len(f),'revealed_label_rows':len(adaptive),'no_feedback_calibration_disjoint_from_model_selection':True,'scope':'Empirical coverage; time dependence/shift not certified exchangeable. Revealed adaptive labels unavailable operationally during dark outages.'})
if __name__=='__main__':main()
