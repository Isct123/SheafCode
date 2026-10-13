                                                                             
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def objective(model,W):
                                                                                    
    n,p=model.n,model.p;d=n*p;rows=[sp.block_diag([sp.csr_matrix(model.D)]*W,format='csr')];ri=[];ci=[];vv=[];base=0
    for lag,pairs in model.temporal.items():
        for t in range(max(0,W-lag)):
            for v,(A,B) in enumerate(pairs):
                for j in range(p):
                    for q in range(p):
                        ri.extend([base+j,base+j]);ci.extend([t*d+v*p+q,(t+lag)*d+v*p+q]);vv.extend([A[j,q],B[j,q]])
                base+=p
    if base:rows.append(sp.coo_matrix((vv,(ri,ci)),shape=(base,W*d)).tocsr())
    D=sp.vstack(rows,format='csr');return (D.T@D+model.lam*sp.kron(sp.eye(W),model.P,format='csr')).tocsr()

def solve(Q,x):
    y=x.ravel().copy();U=np.flatnonzero(~np.isfinite(y));O=np.flatnonzero(np.isfinite(y));y[U]=spsolve(Q[U][:,U].tocsc(),-Q[U][:,O]@y[O]);return y.reshape(x.shape)

def filter_var(model,x):
    n,p=model.n,model.p;d=n*p;mean=np.zeros(d);C=np.eye(d);out=[]
    for t,row in enumerate(x.reshape(len(x),d)):
        if t:mean=model.A@mean;C=model.A@C@model.A.T+model.Q
        O=np.flatnonzero(np.isfinite(row))
        if len(O):
            H=np.eye(d)[O];innovation=H@C@H.T+model.R*np.eye(len(O));K=C@H.T@np.linalg.inv(innovation);mean=mean+K@(row[O]-H@mean);C=(np.eye(d)-K@H)@C
        C=(C+C.T)/2;pr=mean.copy();pr[O]=row[O];out.append(pr)
    return np.array(out).reshape(x.shape)

def controls(x,v,ds,m):
    ans={};n,p=x.shape[1:]
    for name,Q in [('spatial_sheaf',m['sheaf'].D.T@m['sheaf'].D+m['sheaf'].lam*m['sheaf'].P),('gaussian',m['gaussian'])]:
        pred=[]
        for state in x[24:]:
            y=state.ravel().copy();U=np.flatnonzero(~np.isfinite(y));O=np.flatnonzero(np.isfinite(y));y[U]=np.linalg.lstsq(Q[np.ix_(U,U)],-Q[np.ix_(U,O)]@y[O],rcond=None)[0];pred.append(y.reshape(n,p)[v])
        ans[name]=np.array(pred)
    city=[];season=[]
    for t in range(24,48):
        raw=x[t]*ds['sd']+ds['mu'];other=raw[np.arange(n)!=v];count=np.isfinite(other).sum(0);average=np.divide(np.nansum(other,0),count,out=ds['mu'][v].copy(),where=count>0);city.append((average-ds['mu'][v])/ds['sd'][v]);pr=[]
        for j in range(p):
            if np.isfinite(x[t-24,v,j]):pr.append(x[t-24,v,j])
            else:
                previous=np.flatnonzero(np.isfinite(x[:24,v,j]));survivors=x[t,np.arange(n)!=v,j];pr.append(x[previous[-1],v,j] if len(previous) else np.nanmean(survivors) if np.isfinite(survivors).any() else 0.)
        season.append(pr)
    ans.update(city_mean=np.array(city),seasonal=np.array(season));return ans

def main():
    started=time.time();checks=[];rows=[]
    for region in regions()+['epa_la']:
        ds=load(region);m=models(region);ref=pd.read_csv(OUT/region/'outage_cases.csv',dtype={'station':str});selected=ref;Q={W:objective(m['sheaf'],W) for W in range(25,49)};Qsp=m['sheaf'].D.T@m['sheaf'].D+m['sheaf'].lam*m['sheaf'].P;ixmap=dict(blocks(ds,2))
        for (day,station),g in selected.groupby(['day','station']):
            ix=ixmap[day];v=int(np.flatnonzero(ds['stations']==station)[0]);x=ds['Z'][ix].copy();x[24:,v]=np.nan;truth=ds['Z'][ix[24:],v];good=np.isfinite(truth)
            causal=np.array([solve(Q[j+1],x[:j+1])[-1,v] for j in range(24,48)]);retro=solve(Q[48],x)[24:,v];var=filter_var(m['var'],x)[24:,v]
            predictions={'sheaf_causal':causal,'sheaf_retrospective':retro,'var_causal':var}|controls(x,v,ds,m)
            for method,pred in predictions.items():
                mse=float(np.mean((pred-truth)[good]**2));saved=float(g.loc[g.method==method,'mse'].iloc[0]);passed=abs(mse-saved)<=1e-9*(1+abs(saved));checks.append({'check':region+'/'+day+'/'+station+'/'+method,'passed':passed});rows.append({'region':region,'day':day,'station':station,'method':method,'independent_mse':mse,'saved_mse':saved,'absolute_difference':abs(mse-saved)})
                if not passed:raise AssertionError(checks[-1])
        check={'check':region+'/independent_static_objective','passed':np.allclose(Qsp,m['sheaf'].S,rtol=1e-12,atol=1e-12)};checks.append(check)
        if not check['passed']:raise AssertionError(check)
        joint=pd.read_csv(OUT/region/'functional_joint_bounded_cases.csv',dtype={'station':str});references={'PM2.5':60.,'PM10':100.,'O3':100.,'O3_ppm':.05,'NO2_ppb':40.}
        for i,r in joint.iterrows():
            if not r.bounded_assumptions_met:continue
            t=ds['times'].get_loc(pd.Timestamp(r.timestamp));v=int(np.flatnonzero(ds['stations']==r.station)[0]);actual=max(ds['X'][t,v,j]/references[c] for j,c in enumerate(ds['channels']));passed=r.joint_status=='bounded' and r.joint_lower-1e-7<=actual<=r.joint_upper+1e-7;checks.append({'check':region+'/conditional_joint_truth/'+str(i),'passed':passed})
            if not passed:raise AssertionError(checks[-1])
        print(region,'independent all-day inference replay complete',len(selected)//7,'cases',flush=True)
    frame_save('INDEPENDENT_FINAL_REPLAY.csv',pd.DataFrame(rows))
                                                                           
    old=ROOT/'validation_results/external';mapping=json.loads((old/'station_mapping.json').read_text());before=pd.read_csv(old/'observations.csv',parse_dates=['timestamp']);reverse={x['station']:str(x['location_id']) for x in mapping.values()};before['station_id']=before.station.map(reverse);new=pd.read_csv(DATA/'observations.csv',dtype={'station':str},parse_dates=['timestamp']);merged=before.merge(new,left_on=['station_id','timestamp','pollutant'],right_on=['station','timestamp','pollutant'],suffixes=('_prior','_recent'))
                                                                                              
    a=before.groupby(['station_id','timestamp','pollutant']).value.mean().reset_index();shared=a.merge(new,left_on=['station_id','timestamp','pollutant'],right_on=['station','timestamp','pollutant'],suffixes=('_prior','_recent'));shared['value_agrees']=np.isclose(shared.value_prior,shared.value_recent,rtol=1e-10,atol=1e-10);frame_save('PRIOR_RECENT_OBSERVATION_OVERLAP.csv',shared)
    save('PRIOR_RECENT_OVERLAP.json',{'shared_station_ids':sorted(shared.station_id.unique()),'shared_hour_channel_rows':len(shared),'equal_aggregated_values':int(shared.value_agrees.sum()),'shared_period':[str(shared.timestamp.min()),str(shared.timestamp.max())],'prior_observations_sha256':sha(old/'observations.csv'),'scope':'Exact ID/time/channel match against preserved prior supplementary OpenAQ observations. Shared observations make recent Delhi a refitted exploratory holdout, not a wholly new blinded validation. Broader overlap with historical cohorts or prior user access is not excluded.'})
    save('FINAL_CONFIRMATION_REPLAY.json',{'status':'passed','checks':checks,'inference_checks':len(rows)+2,'conditional_joint_checks':len(checks)-len(rows)-2,'all_eligible_station_days':len(rows)//7,'methods_replayed':7,'source_sha256':sha(__file__),'elapsed_seconds':time.time()-started,'scope':'Independent incidence-row quadratic, observation-matrix Kalman and separate simple-control inference replay for every eligible station-day, plus conditional joint-score containment, using frozen learned parameters. Confirms inference/numerical reporting, not independent model fitting, physical validity or unrestricted theorem originality.'})
if __name__=='__main__':main()
