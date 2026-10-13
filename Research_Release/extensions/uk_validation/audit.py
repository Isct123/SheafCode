                                                                         
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECENT = ROOT / 'Research_Release/extensions/recent_data'
OUT = ROOT / 'Research_Release/uk_validation_results'
sys.path.insert(0, str(RECENT))
import common
common.OUT = OUT
from common import load, models, blocks, paired, frame_save, save
import final_confirmation as replay
sys.path.insert(0, str(ROOT / 'audit_tools/uk_packages'))
import rdata

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checks = []
    def check(name, result):
        checks.append({'check': name, 'passed': bool(result)})
        if not result: raise AssertionError(name)
    folder = OUT / 'uk_aurn'; ds = load('uk_aurn'); m = models('uk_aurn')
    manifest = json.loads((folder / 'dataset_manifest.json').read_text())
    check('protocol_hash', manifest['protocol_sha256'] == sha(HERE / 'PROTOCOL.md'))
    check('full_hourly_calendar', ds['times'].equals(pd.date_range('2026-01-01','2026-06-30 23:00',freq='h')))
    check('split_exact', np.array_equal(ds['split'], np.select([np.arange(len(ds['times'])) < int(.6*len(ds['times'])), np.arange(len(ds['times'])) < int(.7*len(ds['times'])), np.arange(len(ds['times'])) < int(.8*len(ds['times']))], [0,1,3], default=2)))
    tr = ds['X'][ds['split']==0]
    check('train_only_means', np.allclose(ds['mu'], np.nanmean(tr,0)))
    check('train_only_scales', np.allclose(ds['sd'], np.nanstd(tr,0,ddof=1)))
    check('normalized_tensor', np.allclose(ds['Z'], (ds['X']-ds['mu'])/ds['sd'], equal_nan=True))
    qualified = [r['code'] for r in manifest['eligibility'] if r['qualified']][:5]
    check('sorted_training_only_cohort', ds['stations'].tolist() == qualified)
    for source in manifest['sources']:
        check('raw_hash/'+source['url'].rsplit('/',1)[1], sha(OUT/'raw'/source['url'].rsplit('/',1)[1]) == source['sha256'])
    for v, station in enumerate(ds['stations']):
        f = rdata.read_rda(OUT/'raw'/(station+'_2026.RData'))[station+'_2026'].copy()
        f.columns = [str(c).lower() for c in f.columns]
        dates = pd.to_datetime(f.date, unit='s', utc=True).dt.tz_localize(None) if pd.api.types.is_numeric_dtype(f.date) else pd.to_datetime(f.date, utc=True).dt.tz_localize(None)
        check('hour_beginning_utc/'+station, dates.dt.year.eq(2026).any() and dates.dt.minute.eq(0).all())
        f.index = dates
        raw = f[['pm2.5','pm10','o3']].reindex(ds['times']).to_numpy(dtype=float, copy=True); raw[raw<0] = np.nan
        for j, channel in enumerate(ds['channels']):
            check('source_to_tensor/'+station+'/'+channel, np.allclose(raw[:,j], ds['X'][:,v,j], equal_nan=True))
    quality = pd.read_csv(OUT/'ratification.csv')
    check('selected_period_ratified', quality[quality.station.isin(ds['stations'])].period_fully_ratified.all())
    for name in ['fit_lock.json','evaluation_lock.json']:
        lock = json.loads((folder/name).read_text())
        check(name+'/dataset', lock['dataset_sha256'] == sha(folder/'dataset.npz'))
    choice = json.loads((folder/'model_choices.json').read_text())
    check('model_hash', choice['model_sha256'] == sha(folder/'models.pkl'))
    ref = pd.read_csv(folder/'outage_cases.csv', dtype={'station':str})
    expected_methods = {'sheaf_causal','sheaf_retrospective','var_causal','seasonal','city_mean','gaussian','spatial_sheaf'}
    ixmap = dict(blocks(ds,2)); primary_days = {d for d,_ in blocks(ds,2)[::3]}
    Q = {W: replay.objective(m['sheaf'], W) for W in range(25,49)}
    records = []
    for (day,station), group in ref.groupby(['day','station']):
        check('method_universe/'+day+'/'+station, set(group.method)==expected_methods)
        check('common_target_cells/'+day+'/'+station, group.target_cells.nunique()==1)
        check('primary_anchor/'+day+'/'+station, group.primary.eq(day in primary_days).all())
        ix = ixmap[day]; v = int(np.flatnonzero(ds['stations']==station)[0]); x = ds['Z'][ix].copy(); x[24:,v] = np.nan
        truth = ds['Z'][ix[24:],v]; good = np.isfinite(truth)
        predictions = {'sheaf_causal':np.array([replay.solve(Q[j+1],x[:j+1])[-1,v] for j in range(24,48)]),
                       'sheaf_retrospective':replay.solve(Q[48],x)[24:,v],
                       'var_causal':replay.filter_var(m['var'],x)[24:,v]} | replay.controls(x,v,ds,m)
        for method, prediction in predictions.items():
            mse = float(np.mean((prediction-truth)[good]**2)); stored = float(group.loc[group.method==method,'mse'].iloc[0])
            check('independent_replay/'+day+'/'+station+'/'+method, abs(mse-stored)<=1e-9*(1+abs(stored)))
            records.append({'day':day,'station':station,'method':method,'independent_mse':mse,'stored_mse':stored,'absolute_difference':abs(mse-stored)})
    frame_save('INDEPENDENT_REPLAY.csv',pd.DataFrame(records))
    summaries = json.loads((folder/'reconstruction_summary.json').read_text())
    for scope, f in [('primary_every_third_day',ref[ref.primary]),('all_eligible_test_days',ref)]:
        result = paired(f,'sheaf_causal','var_causal')
        check(scope+'/gain', abs(result['gain_percent']-summaries[scope]['var_causal']['gain_percent'])<1e-10)
        check(scope+'/denominator', result['cases']==summaries[scope]['var_causal']['cases'])
    ranks = pd.read_csv(folder/'structured_rank_cases.csv')
    check('all_actual_structured_ranks', ranks.agree.all())
    margins = pd.read_csv(folder/'exhaustive_block_margins.csv')
    import math
    check('exhaustive_four_block_enumeration', len(margins)==sum(math.comb(len(ds['stations']),k) for k in range(1,min(4,len(ds['stations']))+1)))
    joint = pd.read_csv(folder/'functional_joint_bounded_cases.csv')
    for i,row in joint.iterrows():
        if row.bounded_assumptions_met:
            t=ds['times'].get_loc(pd.Timestamp(row.timestamp));v=int(np.flatnonzero(ds['stations']==row.station)[0])
            truth=max(ds['X'][t,v]/np.array([60.,100.,100.]))
            check('conditional_joint_truth/'+str(i), row.joint_status=='bounded' and row.joint_lower-1e-7<=truth<=row.joint_upper+1e-7)
        if row.bounded_status=='bounded':
            check('dual_outer/'+str(i), row.dual_lower<=row.bounded_lower+1e-6 and row.dual_upper>=row.bounded_upper-1e-6)
    f=pd.read_csv(folder/'synthetic_certificate_cases.csv')
    check('synthetic_exact_containment', ((f.exact_lower<=f.truth+1e-7)&(f.exact_upper>=f.truth-1e-7)).all())
    check('synthetic_dual_outer_containment', ((f.dual_lower<=f.exact_lower+1e-7)&(f.dual_upper>=f.exact_upper-1e-7)).all())
    f=pd.read_csv(folder/'operator_uncertainty_cases.csv')
    check('operator_uncertainty_conditional_containment', f.robust_covered.all())
    intervals=pd.read_csv(folder/'uncertainty_cases.csv')
    check('interval_coverage_flags', intervals.covered.eq((intervals.truth>=intervals.lower)&(intervals.truth<=intervals.upper)).all())
    summary=pd.read_csv(folder/'uncertainty_summary.csv')
    for i,row in summary[summary.group=='all'].iterrows():
        f=intervals[(intervals.context_loss==row.context_loss)&(intervals.method==row.method)]
        check('interval_denominator/'+str(i), len(f)==row.cases)
        check('interval_coverage/'+str(i), abs(f.covered.mean()-row.coverage)<1e-12)
    preserved={}
    save('AUDIT.json',{'status':'passed','check_count':len(checks),'checks':checks,'station_days_replayed':len(records)//7,
         'source_to_tensor_checked':True,'preserved_documents':preserved,'scope':'Internal source and independent inference audit, not independent refitting or formal proof verification'})
    print('UK audit passed:',len(checks),'checks;',len(records)//7,'all-method station-days replayed',flush=True)

if __name__ == '__main__': main()
