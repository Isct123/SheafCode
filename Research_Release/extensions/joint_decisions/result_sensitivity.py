                                                                              
import numpy as np
import pandas as pd


def pair_cases(frame, method, reference):
    keys = ['day', 'station', 'method']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate station/day/method; aggregate seeds explicitly first')
    if not np.isfinite(frame.mse).all() or (frame.mse < 0).any():
        raise ValueError('Invalid saved MSE')
    wide = frame.pivot(index=['day','station'], columns='method', values='mse')
    pair = wide[[method,reference]].dropna().reset_index()
    pair.columns = ['day','station','method_mse','reference_mse']
    if pair.empty or pair.reference_mse.mean() <= 0:
        raise ValueError('No usable positive-reference common cases')
    return pair


def cluster_sensitivity(pair, cluster_days, family_size=10, draws=20000, seed=913):
                                                                           

                                                                         
                                                                        
                                                                       
       
    if cluster_days not in [1,7,14] or family_size < 1 or draws < 100:
        raise ValueError('Invalid sensitivity configuration')
    day = pd.to_datetime(pair.day)
    labels = ((day-pd.Timestamp('1970-01-05')).dt.days // cluster_days).to_numpy()
    x = pair[['method_mse','reference_mse']].to_numpy(float)
    if not np.isfinite(x).all() or (x < 0).any() or x[:,1].mean() <= 0:
        raise ValueError('Invalid paired outcomes')
    ids = np.unique(labels)
    sums = np.array([x[labels==k].sum(0) for k in ids])
    counts = np.array([(labels==k).sum() for k in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0,len(ids),size=(draws,len(ids)))
    sampled = sums[idx].sum(1)
    if (sampled[:,1] <= 0).any():
        raise ValueError('Some bootstrap reference sums are zero')
    gain = 100*(1-np.sqrt(sampled[:,0]/sampled[:,1]))
    point = float(100*(1-np.sqrt(x[:,0].sum()/x[:,1].sum())))
    alpha = .05/family_size
    adjusted = np.quantile(gain,[alpha/2,1-alpha/2])
    return {'station_days':len(x), 'calendar_days':int(day.nunique()),
            'cluster_days':cluster_days, 'clusters':len(ids),
            'small_cluster_warning':len(ids)<8,
            'gain_percent':point,
            'nominal_95_low':float(np.quantile(gain,.025)),
            'nominal_95_high':float(np.quantile(gain,.975)),
            'family_sensitivity_low':float(adjusted[0]),
            'family_sensitivity_high':float(adjusted[1]),
            'nominal_family_size':family_size, 'bootstrap_draws':draws,
            'largest_cluster_case_fraction':float(counts.max()/counts.sum())}


def leave_cluster_out(pair, cluster_days=7):
    day = pd.to_datetime(pair.day)
    labels = ((day-pd.Timestamp('1970-01-05')).dt.days // cluster_days).to_numpy()
    rows=[]
    for k in np.unique(labels):
        keep = pair.loc[labels!=k]
        if keep.empty or keep.reference_mse.sum()<=0:
            continue
        rows.append({'excluded_cluster':int(k), 'remaining_cases':len(keep),
                     'gain_percent':float(100*(1-np.sqrt(keep.method_mse.sum()/keep.reference_mse.sum())))})
    return rows
