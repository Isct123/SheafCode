                                                                              
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sheafcode.cohort import select_cohort
from sheafcode.baselines2 import GaussianConditional
from sheafcode.reliable import CachedSpatial,paired_bootstrap
base=ROOT/'results_expanded';out=ROOT/'expanded_docs';out.mkdir(exist_ok=True)
comparisons=[];checks=[];statuses={}
for folder in sorted(base.iterdir()):
    if not (folder/'manifest.json').exists():continue
    m=json.loads((folder/'manifest.json').read_text());statuses[folder.name]=m['status']
    if m['status']!='complete':continue
    for name,h in m['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h, 'Source drift: '+name
    P=pd.read_csv(folder/'prolonged.csv');ds=select_cohort(folder.name)
    ds.set_transform('none');std=ds.std.copy();choice=m['choices']['gaussian']
    ds.set_transform(choice['transform']);Z=ds.Z();g=GaussianConditional(Z[ds.split==0],.1)
    C=(1-choice['shrink'])*g.S+choice['shrink']*np.diag(np.diag(g.S))
    decoder=CachedSpatial(np.linalg.inv(C),ds.n,ds.p,g.mu);extra=[]
    for (day,station),group in P.groupby(['day','station']):
        ix=np.flatnonzero((ds.split==2)&(ds.times.floor('D')==pd.Timestamp(day)));assert len(ix)==24
        v=ds.stations.index(station);pr=ds.raw_from_Z(decoder.predict(Z[ix],[v])[0],ix,[v])[:,0]
        truth=ds.X[ix,v];good=np.isfinite(truth);assert np.isfinite(pr[good]).all()
        mse=float(np.mean(((pr-truth)/std[v])[good]**2))
        for mode in group['mode'].unique():extra.append({'day':day,'station':station,'mode':mode,'method':'gaussian','mse':mse})
    P=pd.concat([P,pd.DataFrame(extra)],ignore_index=True);assert np.isfinite(P.mse).all()
    P.to_csv(folder/'prolonged_with_gaussian.csv',index=False)
    for mode in ['causal','retrospective']:
        sub=P[P['mode']==mode]
        for ref in ['stable_var','gaussian','spatial_sheaf','seasonal_persistence']:
            r=paired_bootstrap(sub,'temporal_sheaf',ref);r['station_cluster_ci95']=paired_bootstrap(sub,'temporal_sheaf',ref,unit='station')['difference_ci95']
            t=sub.pivot_table(index=['day','station'],columns='method',values='mse').dropna(subset=['temporal_sheaf',ref]).reset_index()
            t['delta']=t[ref]-t.temporal_sheaf
                                                                               
            t['week']=(pd.to_datetime(t.day).astype('int64')//(7*86400*10**9)).astype(int)
            w=t.groupby('week').delta.agg(['sum','count']);rng=np.random.default_rng(907)
            ids=rng.integers(0,len(w),(2000,len(w)))
            draws=w['sum'].to_numpy()[ids].sum(1)/w['count'].to_numpy()[ids].sum(1)
            r['seven_calendar_day_cluster_ci95']=np.quantile(draws,[.025,.975]).tolist();r['seven_day_clusters']=len(w)
            comparisons.append({'region':folder.name,'mode':mode,'reference':ref,**r})
    checks.append({'region':folder.name,'source_hashes':len(m['code_sha256']),'finite_primary_errors':True,'gaussian_cases':len(extra)})
    print(folder.name,'expanded audit complete',flush=True)
T=pd.DataFrame(comparisons);T.to_csv(out/'comparisons.csv',index=False)
(out/'audit.json').write_text(json.dumps({'statuses':statuses,'checks':checks,'scope':'expanded internal benchmark, all eligible target stations; no untouched external validation'},indent=2))
lines=['# Expanded benchmark audit','', 'All eligible target stations; up to 120 snapshot days, 60 prolonged outage days and 360 fault hours. This extends internal coverage of previously examined archives, not independent external replication. Raw-concentration and inverted-sub-index cohorts remain distinct. Seven-day intervals resample fixed seven-calendar-day clusters, not individual hours.','',str(statuses),'',
 '| Network | Causal comparator | Cases / days | RMSE gain | Day-cluster MSE difference CI | Seven-calendar-day CI |', '|---|---|---:|---:|---|---|']
for _,r in T[T['mode']=='causal'].iterrows():
    lines.append(f"| {r.region} | {r.reference} | {r.n} / {r.clusters} | {100*r.rmse_relative_gain:.1f}% | {r.difference_ci95} | {r.seven_calendar_day_cluster_ci95} |")
lines+=['','Intervals alone are not multiplicity-adjusted significance tests. Inspect every comparator, including seasonal persistence. Larger samples cannot turn prior test access into a blinded holdout or establish pollution reduction.','']
(out/'EXPANDED_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
