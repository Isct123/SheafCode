                                                                             
from pathlib import Path
import hashlib
import json
import pickle
import sys
import zipfile
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Research_Release/replication_20261012_results'
sys.path.insert(0, str(ROOT / 'Research_Release/extensions/recent_data'))
import common
common.OUT = OUT

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def near(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)

def save(p, value):
    p.write_text(json.dumps(common.clean(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')

def independent_gain(f, a, b, metric):
    left = f[f.method==a].set_index(['day','station'])[metric]
    right = f[f.method==b].set_index(['day','station'])[metric]
    pair = pd.concat([left.rename('a'),right.rename('b')],axis=1).dropna()
    return {'cases':len(pair), 'gain_percent':float(100*(1-np.sqrt(pair.a.sum()/pair.b.sum())))} if len(pair) else {'cases':0}

def audit(name, reconstruction=True):
    folder = OUT / name
    ds = common.load(name)
    manifest = json.loads((folder / 'dataset_manifest.json').read_text())
    X, Z, split = ds['X'], ds['Z'], ds['split']
    dates = ds['times']
    assert dates[0] >= pd.Timestamp('2025-10-14') and dates[-1] <= pd.Timestamp('2026-10-12 23:59:59')
    assert dates.is_unique and dates.is_monotonic_increasing
    assert np.all(np.diff(dates.to_numpy(dtype='datetime64[ns]'))==np.timedelta64(1,'h'))
    assert all((split==part).any() for part in [0,1,3,2])
    assert max(np.flatnonzero(split==0)) < min(np.flatnonzero(split==1))
    assert max(np.flatnonzero(split==1)) < min(np.flatnonzero(split==3)) < min(np.flatnonzero(split==2))
    assert np.isfinite(X[split==0]).mean(0).min() >= .5
    assert np.isfinite(X[split==0]).sum(0).min() >= 720
    near(ds['mu'], np.nanmean(X[split==0],axis=0))
    near(ds['sd'], np.nanstd(X[split==0],axis=0,ddof=1))
    near(Z, (X-ds['mu'])/ds['sd'])
    assert not np.any(X[np.isfinite(X)]<0)
    for source in manifest['sources']:
        p = ROOT / source['file']
        assert sha(p)==source['sha256'], p
    observations = pd.read_csv(folder/'source_tensor_observations.csv',dtype={'station':str},parse_dates=['timestamp'])
    assert len(observations)==int(np.isfinite(X).sum())
    assert not observations.duplicated(['timestamp','station','pollutant']).any()
    ti = dates.get_indexer(observations.timestamp)
    vi = pd.Index(ds['stations'].astype(str)).get_indexer(observations.station)
    ci = pd.Index(ds['channels']).get_indexer(observations.pollutant)
    assert min(ti.min(),vi.min(),ci.min())>=0
    near(observations.value.to_numpy(),X[ti,vi,ci])
                                                                               
                                       
    if manifest['provider']=='EPA':
        raw=pd.read_csv(OUT/(name+'_raw_observations.csv'),parse_dates=['timestamp'],low_memory=False)
        for v,station in enumerate(ds['stations']):
            for j,monitor in enumerate(manifest['chosen_monitors'][v]):
                unit='Micrograms/cubic meter (LC)' if j==0 else 'Parts per billion' if ds['channels'][j]=='NO2_ppb' else 'Parts per million'
                good=raw[(raw['Site Num']==int(station))&(raw.parameter==monitor['parameter'])&(raw.POC==monitor['POC'])&(raw['Units of Measure']==unit)&raw['Sample Measurement'].ge(0)]
                values=good.groupby('timestamp')['Sample Measurement'].mean().reindex(dates).to_numpy()
                near(values,X[:,v,j])
    elif manifest['provider']=='NSW':
        packet=json.loads((ROOT/manifest['sources'][0]['file']).read_text(encoding='utf-8-sig'))
        values=np.full_like(X,np.nan)
        station_index={str(s):v for v,s in enumerate(ds['stations'])}
        channel_index={str(s):j for j,s in enumerate(ds['channels'])}
        seen=set()
        for entry in packet:
            station=str(entry['Site_Id']);parameter=entry['Parameter']
            if station not in station_index or parameter['ParameterCode'] not in channel_index:
                continue
            assert parameter['Frequency']=='Hourly average' and parameter['SubCategory']=='Hourly'
            unit=parameter['Units'].replace('µ','u').replace('μ','u').replace('³','3').replace(' ','')
            assert unit=='ug/m3', unit
            t=dates.get_indexer([pd.Timestamp(entry['Date'])+pd.Timedelta(hours=int(entry['Hour'])-1)])[0]
            if t<0: continue
            key=(t,station_index[station],channel_index[parameter['ParameterCode']])
            assert key not in seen
            seen.add(key)
            value=entry['Value']
            if value is not None and value>=0:values[key]=value
        near(values,X)
    elif manifest['provider']=='AURN':
        sys.path.insert(0,str(ROOT/'audit_tools/uk_packages'))
        import rdata,warnings
        warnings.filterwarnings('ignore',message='Missing constructor for R class')
        for v,station in enumerate(ds['stations']):
            source=next(s for s in manifest['sources'] if Path(s['file']).name==str(station)+'_2026.RData')
            raw=rdata.read_rda(ROOT/source['file'])[str(station)+'_2026']
            raw.columns=[str(c).lower() for c in raw.columns]
            raw.index=pd.to_datetime(raw.date,unit='s',utc=True).dt.tz_localize(None) if pd.api.types.is_numeric_dtype(raw.date) else pd.to_datetime(raw.date,utc=True).dt.tz_localize(None)
            values=raw.reindex(index=dates,columns=['pm2.5','pm10']).to_numpy(dtype=float,copy=True)
            values[~np.isfinite(values)|(values<0)]=np.nan
            near(values,X[:,v])
    if not reconstruction:
        eligible=sum(int((np.isfinite(Z[ix[24:]]).sum(axis=(0,2))>=12*Z.shape[2]).sum())
                     for _,ix in common.blocks(ds,2))
        assert eligible==0
        return {'cohort':name,'status':'passed','outcome':'unavailable_data',
                'reason':'No test station-day meets the fixed 12p target-cell rule',
                'tensor_cells':int(np.isfinite(X).sum()),'source_rows_verified':len(observations),
                'preserved_source_independently_reparsed':True,'eligible_test_station_days':0,
                'dataset_sha256':sha(folder/'dataset.npz')}
    f = pd.read_csv(folder/'outage_cases.csv',dtype={'station':str})
    assert not f.duplicated(['day','station','method']).any()
    methods = ['sheaf_causal','var_causal','seasonal','city_mean','sheaf_retrospective','spatial_sheaf','gaussian']
    expected = set(zip(f[f.method=='sheaf_causal'].day,f[f.method=='sheaf_causal'].station))
    for method in methods:
        g=f[f.method==method]
        assert set(zip(g.day,g.station))==expected
        assert np.isfinite(g.mse).all() and g.mse.ge(0).all()
    summary = json.loads((folder/'reconstruction_summary.json').read_text())
    for scope, frame in [('primary_every_third_day',f[f.primary]),('all_eligible_test_days',f)]:
        for metric, entry in [('mse',summary[scope]['var_causal']),('pm25_raw_mse',summary[scope]['pm25_physical']['var_causal'])]:
            check=independent_gain(frame,'sheaf_causal','var_causal',metric)
            assert check['cases']==entry['cases']
            if check['cases']:
                near(check['gain_percent'],entry['gain_percent'])
                                                                                 
    models = common.models(name)
    eligible = {day:ix for day,ix in common.blocks(ds,2)}
    cases = f[f.method=='sheaf_causal'].iloc[[0,-1]].drop_duplicates(['day','station'])
    replayed=0
    invariance=False
    for _, row in cases.iterrows():
        ix=eligible[row.day]
        v=list(ds['stations'].astype(str)).index(row.station)
        truth=Z[ix[24:],v]
        good=np.isfinite(truth)
        assert good.sum()>=12*Z.shape[2]
        hidden=Z[ix].copy()
        hidden[24:,v]=np.nan
        for method in ['sheaf_causal','var_causal']:
            pred = models['sheaf'].causal(hidden,v) if method=='sheaf_causal' else models['var'].filter(hidden)[24:,v]
            ref=f[(f.day==row.day)&(f.station==row.station)&(f.method==method)].iloc[0]
            near(np.mean((pred-truth)[good]**2),ref.mse)
            pm=np.isfinite(truth[:,0])
            if pm.any(): near(np.mean(((pred[:,0]-truth[:,0])*ds['sd'][v,0])[pm]**2),ref.pm25_raw_mse)
                                                                                        
        altered=hidden.copy()
        altered[24:,v]=123456.
        altered[25:,np.arange(Z.shape[1])!=v]=789012.
        near(models['sheaf'].causal(altered,v)[0],models['sheaf'].causal(hidden,v)[0])
        masked=altered.copy();masked[24:,v]=np.nan
        near(models['var'].filter(masked)[24,v],models['var'].filter(hidden)[24,v])
        invariance=True
        replayed+=1
    rf=OUT/'random_forest'/name
    rf_status=json.loads((rf/'status.json').read_text()) if (rf/'status.json').exists() else None
    if rf_status and rf_status['status']=='completed':
        r=pd.read_csv(rf/'cases.csv',dtype={'station':str})
        assert set(zip(r.day,r.station))==expected
        assert rf_status['target_future_feature_invariance']
        joined=pd.concat([f,r],ignore_index=True)
        for metric in ['mse','pm25_raw_mse']:
            independent=independent_gain(joined,'sheaf_causal','random_forest_causal',metric)
            near(independent['gain_percent'],rf_status['summary']['7'][metric]['gain_percent'])
    return {'cohort':name,'status':'passed','tensor_cells':int(np.isfinite(X).sum()),
        'source_rows_verified':len(observations),'preserved_source_independently_reparsed':True,'outage_station_days':len(expected),
        'sampled_station_days_replayed':replayed,'target_future_invariance':invariance,
        'rf_matched_and_audited':bool(rf_status and rf_status['status']=='completed'),
        'dataset_sha256':sha(folder/'dataset.npz'),'cases_sha256':sha(folder/'outage_cases.csv')}

def fmt(v):
    return f'{v:.2f}' if v is not None and np.isfinite(v) else '—'

def main():
    names=json.loads((OUT/'cohorts.json').read_text())['eligible']
    checks, rows, detailed = [], [], {}
    for name in names:
        folder=OUT/name
        run=json.loads((folder/'run_status.json').read_text()) if (folder/'run_status.json').exists() else {'status':'pending'}
        if run['status']!='completed':
            rows.append({'cohort':name,'status':run['status'],'reason':run.get('reason')})
            if run['status']=='unavailable_data':checks.append(audit(name,reconstruction=False))
            continue
        checks.append(audit(name))
        ds=common.load(name)
        f=pd.read_csv(folder/'outage_cases.csv',dtype={'station':str})
        rf=OUT/'random_forest'/name
        if (rf/'status.json').exists() and json.loads((rf/'status.json').read_text())['status']=='completed':
            r=pd.read_csv(rf/'cases.csv',dtype={'station':str})
            flags=f[f.method=='sheaf_causal'][['day','station','primary']]
            r=r.merge(flags,on=['day','station'],validate='one_to_one')
            f=pd.concat([f,r],ignore_index=True)
        for scope,frame in [('primary_every_third_day',f[f.primary]),('all_eligible_test_days',f)]:
            comparisons={}
            for baseline in ['var_causal','random_forest_causal','city_mean','seasonal','gaussian','spatial_sheaf']:
                comparisons[baseline]={str(days):{metric:common.paired(frame,'sheaf_causal',baseline,value=metric,days=days) for metric in ['mse','pm25_raw_mse']} for days in [7,14]}
            detailed.setdefault(name,{})[scope]=comparisons
            var=comparisons['var_causal']['7']['mse']
            pm=comparisons['var_causal']['7']['pm25_raw_mse']
            forest=comparisons['random_forest_causal']['7']['mse']
            manifest=json.loads((folder/'dataset_manifest.json').read_text())
            rows.append({'cohort':name,'status':'completed','scope':scope,'provider':manifest['provider'],
                'stations':Xn if (Xn:=len(ds['stations'])) else None,'channels':len(ds['channels']),
                'dates_start':str(ds['times'][0]),'dates_end':str(ds['times'][-1]),
                'cases':var['cases'],'test_dates':var.get('dates'),'calendar_bins':var.get('calendar_bins'),
                'normalized_gain_vs_var':var.get('gain_percent'),'ci_low':var['ci95'][0] if var.get('ci95') else None,
                'ci_high':var['ci95'][1] if var.get('ci95') else None,'pm25_gain_vs_var':pm.get('gain_percent'),
                'normalized_gain_vs_rf':forest.get('gain_percent'), 'rf_cases':forest.get('cases',0)})
    save(OUT/'audit.json',{'status':'passed' if len(checks)==len(names) else 'incomplete','checks':checks,
        'scope':'Source/tensor checks and sampled inference replay; not independent refitting or field validation'})
    save(OUT/'comparisons.json',detailed)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'ALL_COHORT_RESULTS.csv',index=False)
    primary=frame[(frame.status=='completed')&(frame.scope=='primary_every_third_day')]
    all_days=frame[(frame.status=='completed')&(frame.scope=='all_eligible_test_days')]
    summary={'attempted_acquired_cohorts':len(names),'completed_cohorts':len(primary),
        'unavailable_cohorts':int((frame.status=='unavailable_data').sum()),
        'primary_paired_station_days':int(primary.cases.sum()),'secondary_paired_station_days':int(all_days.cases.sum()),
        'primary_numerical_wins_vs_var':int((primary.normalized_gain_vs_var>0).sum()),
        'primary_losses_vs_var':int((primary.normalized_gain_vs_var<0).sum()),
        'rf_completed_cohorts':int(primary.normalized_gain_vs_rf.notna().sum()),
        'primary_numerical_wins_vs_rf':int((primary.normalized_gain_vs_rf>0).sum()),
        'all_day_numerical_wins_vs_var':int((all_days.normalized_gain_vs_var>0).sum()),
        'all_outcomes_retained':True,'manuscript_changed':False,'claim_scope':'Exploratory regional reconstruction extension; common providers/dates; no test-driven cohort choice'}
    save(OUT/'summary.json',summary)
    lines=['# Additional recent-data reconstruction results','',
        'This batch retains every acquired cohort and all gains and losses. Sites and monitor identities were selected using training eligibility and fixed ordering before their test results. It is exploratory work following known earlier results, not an independent preregistered replication.','',
        'EPA observations span 14 October 2025–31 March 2026. UK and NSW observations span 1 January–30 June 2026. Every observation used in these experiments falls within the preceding twelve months as of 12 October 2026. These are recent observation windows of less than a year, not twelve full months of data.','',
        f"Completed {len(primary)} of {len(names)} acquired cohorts. Primary every-third-day comparisons contain {int(primary.cases.sum())} matched station-days. Sheaf has numerically lower normalized RMSE than VAR in {summary['primary_numerical_wins_vs_var']} and higher RMSE in {summary['primary_losses_vs_var']}. RF comparisons are complete for {summary['rf_completed_cohorts']} cohorts, with {summary['primary_numerical_wins_vs_rf']} numerical sheaf wins.",'',
        'Positive gain means lower sheaf RMSE. Brackets give paired seven-day bootstrap intervals, omitted below five calendar bins. Cohort wins are not multiplicity-adjusted significance claims. Physical PM2.5 gains are separate from normalized multichannel gains.','',
        '| Cohort | Cases / dates | vs VAR (%) [interval] | PM2.5 vs VAR (%) | vs RF (%) |',
        '|---|---:|---:|---:|---:|']
    for _,r in primary.sort_values('cohort').iterrows():
        ci=f'[{fmt(r.ci_low)}, {fmt(r.ci_high)}]' if pd.notna(r.ci_low) else '[insufficient bins]'
        lines.append(f'| {r.cohort} | {int(r.cases)} / {int(r.test_dates)} | {fmt(r.normalized_gain_vs_var)} {ci} | {fmt(r.pm25_gain_vs_var)} | {fmt(r.normalized_gain_vs_rf)} |')
    for _,r in frame[frame.status!='completed'].iterrows():
        lines.append(f"| {r.cohort} | {r.status}: {r.get('reason','')} | unavailable | unavailable | unavailable |")
    lines+=['','## Strongest positive comparisons','',
        'These are selected highlights from the complete table above. Selecting the best test results after running many cohorts can overstate how well the method will transfer. They do not replace the fixed Indian benchmark.','']
    for _,r in primary.sort_values('normalized_gain_vs_var',ascending=False).head(5).iterrows():
        lines.append(f'- {r.cohort}: normalized gain {fmt(r.normalized_gain_vs_var)}%; physical PM2.5 gain {fmt(r.pm25_gain_vs_var)}%; {int(r.cases)} cases across {int(r.test_dates)} dates.')
    lines+=['','## Audit and reproduction','',
        'audit.json records source hashes, tensor identity, training-only normalization, matching method case sets, independent gain arithmetic and sampled causal inference replay. Both first and last eligible station-day are replayed per cohort. Target/future perturbations check the first outage prediction. RF also checks causal feature invariance. This is internal verification, not independent replication.','',
        'ALL_COHORT_RESULTS.csv contains primary and secondary results. comparisons.json retains all baselines, physical errors and seven-/fourteen-day interval sensitivities. EPA cohorts use PM2.5/O3 or PM2.5/NO2; UK and NSW cohorts use PM2.5/PM10. Source exclusions and unavailable requests remain in the candidate ledgers. Original sources, models, cases and logs remain in per-cohort folders. PROTOCOL.md, UK_PROTOCOL.md and PMNO2_PROTOCOL.md record the selection rules.','',
        'The manuscript and original experiments were not changed. No faults, interval coverage, query savings, regulatory decision recovery, emissions reductions or health benefits are established by this reconstruction batch.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'README.md').write_text('Start with RESULTS.md. ALL_COHORT_RESULTS.csv retains all primary and secondary comparisons; comparisons.json includes all controls and interval sensitivities. audit.json records validation. Reproduce with Research_Release/extensions/replication_20261012/run.py, random_forest.py and audit_report.py. Acquisition scripts and outcome-blind protocols are in that directory. The original manuscript and experiment outputs remain unchanged.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
