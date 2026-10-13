from common import *

def snapshot_stream(ds, m, indices, loss, seed):
    Z = ds['Z']
    n, p = Z.shape[1:]
    rng = np.random.default_rng(seed)
    rows = []
    for t in indices:
        for v, station in enumerate(ds['stations']):
            if not np.isfinite(Z[t, v, 0]):
                continue
            x = Z[t:t + 1].copy()
            x[:, v] = np.nan
            drop = rng.random((n, p)) < loss
            drop[v] = False
            x[:, drop] = np.nan
            pred = static_solve(x, m['sheaf'].S)[0, v]
            city = raw_city(x, v, ds['mu'], ds['sd'])[0]
            raw = pred[0] * ds['sd'][v, 0] + ds['mu'][v, 0]
            true = ds['X'][t, v, 0]
            available = np.isfinite(x).sum() / ((n - 1) * p)
            U = np.flatnonzero(~np.isfinite(x.ravel()))
            O = np.flatnonzero(np.isfinite(x.ravel()))
            Q = m['sheaf'].S
            target = v * p
            position = int(np.flatnonzero(U == target)[0])
            gain = np.linalg.norm(np.linalg.solve(Q[np.ix_(U, U)], Q[np.ix_(U, O)])[position]) * ds['sd'][v, 0]
            rows.append({'t': t, 'station_index': v, 'station': str(station), 'timestamp': str(ds['times'][t]), 'day': str(ds['times'][t].date()), 'truth': true, 'prediction': raw, 'city_prediction': city[0] * ds['sd'][v, 0] + ds['mu'][v, 0], 'availability': available, 'pre_outcome_scale': max(gain, 0.001)})
    return pd.DataFrame(rows)

def main():
    summary=[]
    for region in regions():
        ds=load(region); fitted=models(region)
        cal=snapshot_stream(ds,fitted,np.flatnonzero(ds['split']==3),0.,7501)
        test=snapshot_stream(ds,fitted,np.flatnonzero(ds['split']==2),0.,7601)
        training=ds['X'][ds['split']==0,:,0]
        risk=np.nanquantile(training,.75,axis=0)
        extreme=np.nanquantile(training,.95,axis=0)
        thresholds=dict(zip(map(str,ds['stations']),risk))
        for frame in [cal,test]:
            frame['risk']=np.maximum(frame.prediction,frame.city_prediction)>frame.station.map(thresholds)
        cal['error']=abs(cal.truth-cal.prediction)
        test['extreme']=test.truth.to_numpy()>extreme[test.station_index.to_numpy(dtype=int)]
        rows=[]
        for method in ['pooled_fixed','station_fixed','risk_station']:
            frame=test.copy();cache={};radii=[]
            for row in frame.itertuples():
                key=(row.station if method!='pooled_fixed' else '',bool(row.risk) if method=='risk_station' else False)
                if key not in cache:
                    group=cal if method=='pooled_fixed' else cal[cal.station==row.station]
                    if method=='risk_station':group=group[group.risk==row.risk]
                    cache[key]=quantile(group.error)
                radii.append(cache[key])
            frame['method']=method;frame['radius']=radii
            frame['covered']=abs(frame.truth-frame.prediction)<=frame.radius
            rows.append(frame)
            for label,group in [('all',frame),('extreme',frame[frame.extreme])]:
                summary.append({'cohort':region,'method':method,'group':label,'readings':len(group),'covered':int(group.covered.sum()),'coverage':float(group.covered.mean()) if len(group) else None})
        frame_save(region+'/interval_cases.csv',pd.concat(rows,ignore_index=True))
    frame_save('interval_summary.csv',pd.DataFrame(summary))
