                                                                                 

                                                                         
                                                                            
   
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from sheafcode.cohort import select_cohort
from sheafcode.kalman import KalmanVAR
from sheafcode.reliable import paired_bootstrap

def completed_training(Z,rank=8):
    T,n,p=Z.shape;M=Z.reshape(T,-1);mask=np.isfinite(M);mu=np.nan_to_num(np.nanmean(M,axis=0))
    X=np.where(mask,M,mu);stride=max(1,T//4000);fit=X[::stride]-mu;fm=mask[::stride];original=fit.copy()
    for _ in range(8):
        _,V=np.linalg.eigh(fit.T@fit);basis=V[:,-min(rank,n*p):];fit=np.where(fm,original,fit@basis@basis.T)
    return np.where(mask,M,(X-mu)@basis@basis.T+mu)

def fit_model(X,n,p,ridge,noise,shrink):
    model=KalmanVAR.__new__(KalmanVAR);model.n=n;model.p=p;model.d=n*p
    model.A=np.linalg.solve(X[:-1].T@X[:-1]+ridge*np.eye(n*p),X[:-1].T@X[1:]).T
    radius=np.max(abs(np.linalg.eigvals(model.A)))
    if radius>.995:model.A*=.995/radius
    residual=X[1:]-X[:-1]@model.A.T;C=np.cov(residual,rowvar=False)
    model.Q=(1-shrink)*C+shrink*np.diag(np.diag(C))+1e-3*np.eye(n*p)
    model.R=noise;return model

def main():
    out=ROOT/'extension_results/strong_baselines';out.mkdir(parents=True,exist_ok=True)
    allcomparisons={};config={}
    for region in ['delhi2025','bengaluru_raw']:
        folder=ROOT/'results_expanded'/region;m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete'
        saved=pd.read_csv(folder/'prolonged.csv');ds=select_cohort(region);ds.set_transform('none');std=ds.std.copy()
        tr=ds.rows(0);va=ds.rows(1);choices={};models={};spaces={};validation=[]
        blocks=[]
        for day in pd.unique(ds.times[va].floor('D')):
            ix=va[ds.times[va].floor('D')==day]
            if len(ix)==24 and ix[0]>=24 and ds.split[ix[0]-24]==1:blocks.append(np.arange(ix[0]-24,ix[-1]+1))
        blocks=[blocks[i] for i in np.unique(np.linspace(0,len(blocks)-1,min(8,len(blocks))).astype(int))]
        valstations=list(range(min(6,ds.n)))
        for transform in ['none','diurnal']:
            ds.set_transform(transform);Z=ds.Z();spaces[transform]=Z.copy();X=completed_training(Z[tr])
            for ridge in [1.,100.]:
                for noise in [.05,.5]:
                    for shrink in [0.,.2,1.]:
                        model=fit_model(X,ds.n,ds.p,ridge,noise,shrink)
                        for mode in ['causal','retrospective']:
                            losses=[]
                            for ix in blocks:
                                for v in valstations:
                                    truth=ds.X[ix[24:],v];good=np.isfinite(truth)
                                    if good.sum()<12*ds.p:continue
                                    hidden=Z[ix].copy();hidden[24:,v]=np.nan
                                    pred=model.smooth(hidden,causal=mode=='causal')[24:,v][:,None,:]
                                    raw=ds.raw_from_Z(pred,ix[24:],[v])[:,0]
                                    assert np.isfinite(raw[good]).all()
                                    losses.append(float(np.mean(((raw-truth)/std[v])[good]**2)))
                            loss=float(np.mean(losses));rec={'transform':transform,'ridge':ridge,'observation_noise':noise,'innovation_shrink':shrink,'validation_mse':loss,'mode':mode}
                            validation.append(rec)
                            if mode not in choices or loss<choices[mode]['validation_mse']:choices[mode]=rec;models[mode]=model
        pd.DataFrame(validation).to_csv(out/(region+'_validation.csv'),index=False)
                                                                           
        (out/(region+'_choices.json')).write_text(json.dumps(choices,indent=2));records=[]
        for (day,station),g in saved.groupby(['day','station']):
            v=ds.stations.index(station);ix=np.flatnonzero((ds.split==2)&(ds.times.floor('D')==pd.Timestamp(day)))
            assert len(ix)==24 and ds.split[ix[0]-24]==2
            rows=np.arange(ix[0]-24,ix[-1]+1);truth=ds.X[ix,v];good=np.isfinite(truth)
            for mode in ['causal','retrospective']:
                t=choices[mode]['transform'];ds.set_transform(t);hidden=spaces[t][rows].copy();hidden[24:,v]=np.nan
                pred=models[mode].smooth(hidden,causal=mode=='causal')[24:,v][:,None,:];raw=ds.raw_from_Z(pred,ix,[v])[:,0]
                assert np.isfinite(raw[good]).all()
                records.append({'day':day,'station':station,'mode':mode,'method':'full_covariance_var','mse':float(np.mean(((raw-truth)/std[v])[good]**2))})
        frame=pd.concat([saved,pd.DataFrame(records)],ignore_index=True);frame.to_csv(out/(region+'_cases.csv'),index=False)
        allcomparisons[region]={mode:paired_bootstrap(frame[frame['mode']==mode],'temporal_sheaf','full_covariance_var') for mode in ['causal','retrospective']}
        config[region]={'choices':choices,'validation_days':len(blocks),'validation_stations':len(valstations),'reference_model_manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()}
        print(region,allcomparisons[region],flush=True)
    (out/'summary.json').write_text(json.dumps({'comparisons':allcomparisons,'config':config,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'exploratory stronger classical benchmark on raw concentration cohorts; hyperparameters/transforms chosen on validation; test archive previously examined; no neural SOTA claim'},indent=2))
if __name__=='__main__':main()
