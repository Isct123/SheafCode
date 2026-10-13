from pathlib import Path
import sys, json, warnings
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'evidence/sheafcode_iris_2026_final/sheafcode_iris_2026_final'
TABLES=BASE/'evidence/sheafcode_iris_2026_final_result_tables/sheafcode_iris_2026_final/results'
sys.path.insert(0,str(SOURCE))
from sheafcode.frozen import Frozen
from sheafcode.data import load_b10,load_b4
warnings.filterwarnings('ignore')
record=[]
for reg in ['delhi2025','bengaluru_raw']:
    F=Frozen(reg,temporal=True);ds=F.ds
    ds.set_transform('none');common=ds.Z();mu=ds.mean.copy();std=ds.std.copy();ds.set_transform(F.c['transform'])
    def to_common(pred,rows,v):
        raw=ds.raw_from_Z(pred[:,None,:],rows,[v])[:,0,:]
        return (raw-mu[v])/std[v]
    snap=pd.read_csv(TABLES/f'v3_{reg}'/'test_results.csv')
    rows=snap[(snap.method=='vector_sheaf')&(snap['size']==1)].groupby('pattern').head(2)
    for _,row in rows.iterrows():
        v=ds.stations.index(row.pattern)
        idx=np.where((ds.split==2)&(ds.times.floor('D')==pd.Timestamp(row.day)))[0]
        pred=F.sheaf.reconstruct(F.Z[idx],[v],F.lam)[:,0,:]
        y=common[idx,v,:];m=np.isfinite(y)
        err=float(np.sqrt(np.mean((to_common(pred,idx,v)[m]-y[m])**2)))
        record.append(dict(network=reg,regime='single_snapshot',target=row.pattern,day=row.day,expected=row.rmse_z,replayed=err,passed=bool(np.isclose(err,row.rmse_z,rtol=1e-7,atol=1e-8))))
    pro=pd.read_csv(TABLES/f'v3_{reg}'/'prolonged_results.csv')
    sh=pro[pro.method=='st_sheaf'];units=sh[['run','station']].drop_duplicates().iloc[[0,-1]]
    for _,unit in units.iterrows():
        v=ds.stations.index(unit.station);t=pd.Timestamp(unit.run)
        target_rows=sh[(sh.run==unit.run)&(sh.station==unit.station)].sort_values('hours_elapsed')
        L=int(target_rows.hours_elapsed.max());start=int(np.where(ds.times==t)[0][0])
        idx=np.arange(start-24,start+L);hide=np.zeros((len(idx),F.n),bool);hide[24:,v]=True
        pred=F.sheaf_t.reconstruct_mask(F.Z[idx],hide,F.lam)[24:,v,:]
        score=np.mean((to_common(pred,np.arange(start,start+L),v)-common[start:start+L,v,:])**2,axis=1)
        err=float(np.max(np.abs(score-target_rows.sq_err.to_numpy())))
        record.append(dict(network=reg,regime='prolonged',target=unit.station,run=unit.run,hours=L,max_abs_squared_error_difference=err,passed=bool(np.allclose(score,target_rows.sq_err,rtol=1e-7,atol=1e-8))))
    print(reg,'snapshot rows',len(rows),'prolonged units',len(units),'all passed',all(r['passed'] for r in record),flush=True)

                                                                          
idx=load_b10();raw=load_b4();stats={}
for station in ['BTM','Jayanagar','Peenya','SilkBoard']:
    for pol in ['PM2.5','PM10','NO2','O3']:
        pair=pd.concat([idx[(station,pol)],raw[(station,pol)]],axis=1,join='inner').dropna()
        if len(pair):
            diff=np.abs(pair.iloc[:,0]-pair.iloc[:,1]);stats[f'{station}/{pol}']=dict(n=len(pair),correlation=float(pair.corr().iloc[0,1]),median_abs_difference=float(diff.median()),p95_abs_difference=float(diff.quantile(.95)))
out=dict(replays=record,overlapping_index_raw_exports=stats,all_replays_pass=all(r['passed'] for r in record))
(BASE/'Final_Manuscript/audit/reconstruction_replay.json').write_text(json.dumps(out,indent=2))
print('replay checks',len(record),'passed',sum(r['passed'] for r in record));print(json.dumps(stats,indent=1))
