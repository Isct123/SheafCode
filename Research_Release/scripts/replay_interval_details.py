                                                                      
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sheafcode.cohort import select_cohort
from sheafcode.reliable import CachedSpatial,AdaptiveQuantile
checks=[]
for folder in sorted((ROOT/'results').iterdir()):
    if not (folder/'manifest.json').exists():continue
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['status']!='complete':continue
    ds=select_cohort(folder.name);ds.set_transform('none');std=ds.std.copy()
    ds.set_transform(manifest['choices']['sheaf']['transform']);Z=ds.Z()
    model=np.load(folder/'model.npz');decoder=CachedSpatial(model['L']+float(model['lam'])*model['P'],ds.n,ds.p)
    va=ds.rows(1);te=ds.rows(2);records=[]
    for station in manifest['target_stations']:
        v=ds.stations.index(station)
        pv,gv=decoder.predict(Z[va],[v]);pt,gt=decoder.predict(Z[te],[v])
        pv=ds.raw_from_Z(pv,va,[v])[:,0];pt=ds.raw_from_Z(pt,te,[v])[:,0]
        ev=abs(pv-ds.X[va,v])/std[v];et=abs(pt-ds.X[te,v])/std[v]
        gv=gv[:,0];gt=gt[:,0]
        streams=[]
        for q in range(ds.p):
            ok=np.isfinite(ev[:,q]);hist=ev[ok,q]
            streams.append(None if len(hist)<50 else (AdaptiveQuantile(hist),AdaptiveQuantile(hist/gv[ok,q]),AdaptiveQuantile(hist/7),AdaptiveQuantile(hist).quantile(False)))
        data={(q,name):[] for q in range(ds.p) for name in ['fixed_no_labels','adaptive_revealed','adaptive_mask_scale','constant_scale_control']}
        for t in range(len(te)):
            thresholds=[]
            for q,s in enumerate(streams):
                thresholds.append(None if s is None else (s[0].quantile(),s[1].quantile(),s[2].quantile(),s[3]))
            for q,s in enumerate(streams):
                if s is None or not np.isfinite(et[t,q]):continue
                u,g,c,f=thresholds[q]
                for name,score,threshold,width in [('fixed_no_labels',et[t,q],f,f),('adaptive_revealed',et[t,q],u,u),('adaptive_mask_scale',et[t,q]/gt[t,q],g,g*gt[t,q]),('constant_scale_control',et[t,q]/7,c,c*7)]:
                    data[q,name].append((t,int(score<=threshold),width))
                s[0].update(et[t,q],u);s[1].update(et[t,q]/gt[t,q],g);s[2].update(et[t,q]/7,c)
        for (q,name),values in data.items():
            if not values:continue
            a=np.array(values);t=a[:,0].astype(int);h=a[:,1];w=a[:,2];finite=np.isfinite(w)&(w>=0)
            quarter=np.minimum(3,t*4//len(te));calendar=ds.times[te[t]].to_period('Q').astype(str)
            cal={k:float(h[calendar==k].mean()) for k in np.unique(calendar)}
            records.append({'station':station,'pollutant':ds.pollutants[q],'method':name,'n':len(h),'coverage':float(h.mean()),
                'full_sets':int(np.isposinf(w).sum()),'empty_sets':int(np.isneginf(w).sum()),
                'mean_finite_halfwidth_training_sigma':float(w[finite].mean()) if finite.any() else None,
                'mean_finite_halfwidth_concentration_units':float(w[finite].mean()*std[v,q]) if finite.any() else None,
                'lowest_equal_duration_test_segment':min(float(h[quarter==j].mean()) for j in np.unique(quarter)),
                'lowest_actual_calendar_quarter':min(cal.values()),'calendar_quarter_counts':json.dumps({k:int((calendar==k).sum()) for k in cal}),
                'calendar_quarter_coverage':json.dumps(cal)})
    details=pd.DataFrame(records);details.to_csv(folder/'intervals_by_pollutant.csv',index=False)
    saved=pd.read_csv(folder/'intervals.csv')
    for (station,name),g in details.groupby(['station','method']):
        old=saved[(saved.station==station)&(saved.method==name)].iloc[0]
        assert g.n.sum()==old.n and abs(np.average(g.coverage,weights=g.n)-old.coverage)<1e-12
        assert g.full_sets.sum()==old.full_sets and g.empty_sets.sum()==old.empty_sets
        checks.append({'region':folder.name,'station':station,'method':name,'passed':True})
    print(folder.name,'pollutant interval replay verified',flush=True)
(ROOT/'docs/interval_replay_audit.json').write_text(json.dumps({'checks':checks,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'per-pollutant replay reproduces pooled coverage/full/empty counts; actual calendar quarters added separately'},indent=2))
