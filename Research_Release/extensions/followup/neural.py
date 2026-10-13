                                                                  

                                                                             
                                                                            
                                                               
   
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from sheafcode.cohort import select_cohort
from sheafcode.reliable import CachedSpatial,paired_bootstrap

def features(z,v,adj):
    hidden=np.asarray(z).copy();hidden[:,v]=np.nan
    mask=np.isfinite(hidden);values=np.nan_to_num(hidden)
    neighbor=values[:, :, :]*adj[v][None,:,None]
    count=np.sum(mask*adj[v][None,:,None],axis=1)
    aggregate=neighbor.sum(axis=1)/np.maximum(count,1)
    return np.concatenate([values.reshape(len(z),-1),mask.reshape(len(z),-1).astype(float),aggregate],axis=1)

def fit(X,Y,width,penalty,seed,steps=300):
    rng=np.random.default_rng(seed);W=rng.normal(0,1/np.sqrt(X.shape[1]),(X.shape[1],width));b=np.zeros(width)
    V=rng.normal(0,1/np.sqrt(width),(width,Y.shape[1]));c=np.zeros(Y.shape[1])
    params=[W,b,V,c];first=[np.zeros_like(x) for x in params];second=[np.zeros_like(x) for x in params]
    for step in range(1,steps+1):
        ix=rng.choice(len(X),min(256,len(X)),replace=False);x=X[ix];y=Y[ix];h=np.tanh(x@W+b);pred=h@V+c
        e=2*(pred-y)/(len(ix)*Y.shape[1]);dh=(e@V.T)*(1-h*h)
        gradients=[x.T@dh+penalty*W,dh.sum(0),h.T@e+penalty*V,e.sum(0)]
        for k,g in enumerate(gradients):
            first[k]=.9*first[k]+.1*g;second[k]=.999*second[k]+.001*g*g
            params[k]-=.003*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
    return params

def predict(X,model):W,b,V,c=model;return np.tanh(X@W+b)@V+c

def main():
    out=ROOT/'followup_results/neural_control';out.mkdir(parents=True,exist_ok=True);summaries={}
    for region in ['delhi2025','bengaluru_raw']:
        folder=ROOT/'results_expanded'/region;m=json.loads((folder/'manifest.json').read_text())
        ds=select_cohort(region);ds.set_transform('none');Z=ds.Z();n,p=ds.n,ds.p
        coords=ds.coords.to_numpy();dist=np.linalg.norm(coords[:,None]-coords[None,:],axis=2);adj=np.zeros((n,n))
        for v in range(n):adj[v,np.argsort(dist[v])[1:min(n,4)]]=1
        records=[];choices=[]
        saved=pd.read_csv(folder/'snapshots.csv');saved=saved[saved.pattern=='single_station'];allowed=set(zip(saved.day.astype(str),saved.station.astype(str)))
        for v in range(n):
            train=np.flatnonzero((ds.split==0)&np.isfinite(Z[:,v]).all(axis=1));train=train[::max(1,len(train)//6000)]
            validation=np.flatnonzero((ds.split==1)&np.isfinite(Z[:,v]).all(axis=1))
            validation=validation[::max(1,len(validation)//1500)]
            if len(train)<30 or len(validation)<10:raise ValueError('Insufficient neural training/validation')
            X=features(Z[train],v,adj);Y=Z[train,v];VX=features(Z[validation],v,adj)
            best=None
            for width in [16,64]:
                for penalty in [0.,.001]:
                    model=fit(X,Y,width,penalty,701+v);loss=float(np.mean((predict(VX,model)-Z[validation,v])**2))
                    if best is None or loss<best[0]:best=(loss,model,width,penalty)
            loss,model,width,penalty=best;choices.append({'station':ds.stations[v],'width':width,'penalty':penalty,'validation_mse':loss})
            (out/(region+'_choices.json')).write_text(json.dumps(choices,indent=2))
            np.savez_compressed(out/(region+'_station_'+str(v)+'.npz'),W=model[0],b=model[1],V=model[2],c=model[3])
                                                                              
            for day,station in sorted(allowed):
                if station!=ds.stations[v]:continue
                ix=np.flatnonzero((ds.split==2)&(ds.times.floor('D')==pd.Timestamp(day)))
                truth=ds.X[ix,v];good=np.isfinite(truth)
                if not good.any():continue
                raw=ds.raw_from_Z(predict(features(Z[ix],v,adj),model)[:,None,:],ix,[v])[:,0]
                records.append({'day':day,'station':station,'pattern':'single_station','method':'graph_nonlinear_control','mse':float(np.mean(((raw-truth)/ds.std[v])[good]**2))})
            print(region,ds.stations[v],'nonlinear control complete',flush=True)
        frame=pd.concat([saved,pd.DataFrame(records)],ignore_index=True);frame.to_csv(out/(region+'_cases.csv'),index=False)
        summaries[region]={'choices':choices,'comparison':paired_bootstrap(frame,'sheaf','graph_nonlinear_control'),
            'model_manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps({'results':summaries,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Project-built graph-conditioned MLP, snapshot only, no temporal history; NOT GRIN or SOTA; single optimization seed; internally examined archives'},indent=2))

if __name__=='__main__':main()
