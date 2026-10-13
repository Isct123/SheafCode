                                                                                    
from pathlib import Path
import sys,json,math,warnings
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'Research_Release/expanded_validation_results'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
common.OUT=OUT
import final_confirmation as replay
from common import load,models,blocks,paired,sha,save,frame_save
sys.path.insert(0,str(ROOT/'audit_tools/uk_packages'))
def audit(region):
    folder=OUT/region;checks=[]
    def check(label,truth):
        checks.append({'check':label,'passed':bool(truth)})
        if not truth:raise AssertionError(region+': '+label)
    ds=load(region);manifest=json.loads((folder/'dataset_manifest.json').read_text())
    check('protocol_hash',manifest['protocol_sha256']==sha(HERE/'PROTOCOL.md'))
    check('complete_hourly_calendar',ds['times'].equals(pd.date_range(ds['times'][0],ds['times'][-1],freq='h')))
    check('chronological_split_order',np.all(np.diff(np.array([{0:0,1:1,3:2,2:3}[int(x)] for x in ds['split']]))>=0))
    tr=ds['X'][ds['split']==0]
    check('training_only_eligibility',(np.isfinite(tr).mean(0)>=.5).all() and (np.isfinite(tr).sum(0)>=720).all())
    check('training_means',np.allclose(ds['mu'],np.nanmean(tr,0)))
    check('training_scales',np.allclose(ds['sd'],np.nanstd(tr,0,ddof=1)))
    check('normalization',np.allclose(ds['Z'],(ds['X']-ds['mu'])/ds['sd'],equal_nan=True))
    for source in manifest['sources']:
        check('raw_hash/'+source['file'],sha(ROOT/source['file'])==source['sha256'])
        if 'request_sha256' in source:check('request_hash',sha(ROOT/source['request_file'])==source['request_sha256'])
    provider=manifest['provider']
    if provider=='EPA':
        f=pd.read_csv(OUT/(region+'_raw_observations.csv'));f['timestamp']=pd.to_datetime(f.timestamp)
        for v,station in enumerate(ds['stations']):
            for j,monitor in enumerate(manifest['chosen_monitors'][v]):
                unit=['Micrograms/cubic meter (LC)','Parts per million'][j]
                g=f[(f['Site Num']==int(station))&(f.parameter==monitor['parameter'])&(f.POC==monitor['POC'])&(f['Units of Measure']==unit)&f['Sample Measurement'].ge(0)]
                values=g.groupby('timestamp')['Sample Measurement'].mean().reindex(ds['times']).to_numpy()
                check('source_tensor/'+station+'/'+ds['channels'][j],np.allclose(values,ds['X'][:,v,j],equal_nan=True))
    elif provider=='AURN':
        import rdata
        warnings.filterwarnings('ignore',message='Missing constructor for R class')
        for v,station in enumerate(ds['stations']):
            f=rdata.read_rda(OUT/'raw'/(station+'_2026.RData'))[station+'_2026'].copy();f.columns=[str(x).lower() for x in f.columns]
            dates=pd.to_datetime(f.date,unit='s',utc=True).dt.tz_localize(None) if pd.api.types.is_numeric_dtype(f.date) else pd.to_datetime(f.date,utc=True).dt.tz_localize(None)
            check('unique_hourly_source/'+station,not dates.duplicated().any());f.index=dates
            values=f.reindex(index=ds['times'],columns=[c.lower() for c in ds['channels']]).to_numpy(dtype=float,copy=True);values[values<0]=np.nan
            check('source_tensor/'+station,np.allclose(values,ds['X'][:,v],equal_nan=True))
    elif provider=='NSW':
        source=manifest['sources'][0];rows=json.loads((ROOT/source['file']).read_text(encoding='utf-8-sig'));observations={}
        for row in rows:
            key=(str(row['Site_Id']),row['Parameter']['ParameterCode'],pd.Timestamp(row['Date'])+pd.Timedelta(hours=int(row['Hour'])-1))
            if key in observations:raise AssertionError('Duplicate NSW raw record')
            observations[key]=row['Value']
        for v,station in enumerate(ds['stations']):
            for j,c in enumerate(ds['channels']):
                values=np.array([observations.get((station,c,t),np.nan) for t in ds['times']],dtype=float);values[values<0]=np.nan
                check('source_tensor/'+station+'/'+c,np.allclose(values,ds['X'][:,v,j],equal_nan=True))
    elif provider=='OpenAQ':
        old=np.load(ROOT/manifest['sources'][0]['file'],allow_pickle=False)
        vi=[old['stations'].tolist().index(s) for s in ds['stations']];ji=[old['channels'].tolist().index(c) for c in ds['channels']]
        check('preserved_original_tensor',np.allclose(old['X'][:,vi][:,:,ji],ds['X'],equal_nan=True))
    else:raise AssertionError('Unrecognized source')
    state=json.loads((folder/'run_status.json').read_text())
    if state['status']=='unavailable_data':
        check('no_test_observations',not np.isfinite(ds['X'][ds['split']==2]).any())
        save(region+'/AUDIT.json',{'status':'passed','check_count':len(checks),'checks':checks,'all_method_station_days_replayed':0,'scope':'Raw-source and normalization audit only; no empirical test observations available','empirical_results_unavailable':True})
        print(region,'source audit passed; empirical study unavailable',flush=True);return
    m=models(region)
    for name in ['fit_lock.json','evaluation_lock.json']:
        value=json.loads((folder/name).read_text());check(name+'/dataset_hash',value['dataset_sha256']==sha(folder/'dataset.npz'))
    check('model_hash',json.loads((folder/'model_choices.json').read_text())['model_sha256']==sha(folder/'models.pkl'))
    f=pd.read_csv(folder/'outage_cases.csv',dtype={'station':str});ixmap=dict(blocks(ds,2));anchors={d for d,_ in blocks(ds,2)[::3]};Q={w:replay.objective(m['sheaf'],w) for w in range(25,49)};records=[]
    expected={'sheaf_causal','sheaf_retrospective','var_causal','seasonal','city_mean','gaussian','spatial_sheaf'}
    for (day,station),g in f.groupby(['day','station']):
        check('method_universe/'+day+'/'+station,set(g.method)==expected)
        check('common_truth/'+day+'/'+station,g.target_cells.nunique()==1)
        check('anchor/'+day+'/'+station,g.primary.eq(day in anchors).all())
        v=ds['stations'].tolist().index(station);ix=ixmap[day];x=ds['Z'][ix].copy();x[24:,v]=np.nan;truth=ds['Z'][ix[24:],v];good=np.isfinite(truth)
        predictions={'sheaf_causal':np.array([replay.solve(Q[j+1],x[:j+1])[-1,v] for j in range(24,48)]),'sheaf_retrospective':replay.solve(Q[48],x)[24:,v],'var_causal':replay.filter_var(m['var'],x)[24:,v]}|replay.controls(x,v,ds,m)
        for name,pred in predictions.items():
            mse=float(np.mean((pred-truth)[good]**2));stored=float(g.loc[g.method==name,'mse'].iloc[0]);check('replay/'+day+'/'+station+'/'+name,abs(mse-stored)<=1e-9*(1+abs(stored)))
            records.append({'day':day,'station':station,'method':name,'independent_mse':mse,'stored_mse':stored})
    frame_save(region+'/independent_replay.csv',pd.DataFrame(records))
    stats={scope:{str(days):paired(subset,'sheaf_causal','var_causal',days=days) for days in [7,14]} for scope,subset in [('all_eligible_test_days',f),('primary_every_third_day',f[f.primary])]}
    save(region+'/dependence_sensitivity.json',stats)
    stored=json.loads((folder/'reconstruction_summary.json').read_text())
    for scope in stats:
        check(scope+'/gain',abs(stats[scope]['7']['gain_percent']-stored[scope]['var_causal']['gain_percent'])<1e-9)
    margins=pd.read_csv(folder/'exhaustive_block_margins.csv');check('exhaustive_four_block_sets',len(margins)==sum(math.comb(len(ds['stations']),k) for k in range(1,min(4,len(ds['stations']))+1)))
    ranks=pd.read_csv(folder/'structured_rank_cases.csv');check('all_nonempty_supports_enumerated',len(ranks)==2**len(ds['stations'])-1)
    check('actual_structured_rank_identity',ranks.agree.all())
    f=pd.read_csv(folder/'uncertainty_cases.csv');check('coverage_flags',f.covered.eq((f.truth>=f.lower)&(f.truth<=f.upper)).all())
    summary=pd.read_csv(folder/'uncertainty_summary.csv')
    for i,row in summary[summary.group=='all'].iterrows():
        g=f[(f.context_loss==row.context_loss)&(f.method==row.method)];check('interval_denominator/'+str(i),len(g)==row.cases);check('interval_coverage/'+str(i),abs(g.covered.mean()-row.coverage)<1e-12)
    syn=pd.read_csv(folder/'synthetic_certificate_cases.csv');check('planted_sharp_containment',((syn.exact_lower<=syn.truth+1e-6)&(syn.exact_upper>=syn.truth-1e-6)).all());check('planted_dual_outer',((syn.dual_lower<=syn.exact_lower+1e-6)&(syn.dual_upper>=syn.exact_upper-1e-6)).all())
    joint=pd.read_csv(folder/'functional_joint_bounded_cases.csv');refs={'PM2.5':60.,'PM10':100.,'O3':100.,'O3_ppm':.05,'NO2_ppb':40.}
    for i,row in joint.iterrows():
        if row.bounded_assumptions_met:
            t=ds['times'].get_loc(pd.Timestamp(row.timestamp));v=ds['stations'].tolist().index(str(row.station));truth=max(ds['X'][t,v]/[refs[c] for c in ds['channels']]);check('conditional_joint/'+str(i),row.joint_status=='bounded' and row.joint_lower-1e-6<=truth<=row.joint_upper+1e-6)
    operators=pd.read_csv(folder/'operator_uncertainty_cases.csv');check('conditional_operator_bounds',operators.robust_covered.all())
    save(region+'/AUDIT.json',{'status':'passed','check_count':len(checks),'checks':checks,'all_method_station_days_replayed':len(records)//7,'scope':'Raw-source reconstruction and independent inference implementation; not independent refitting or external proof review'})
    print(region,'audit passed',len(checks),'checks',len(records)//7,'station-days',flush=True)
def main():
    names=sys.argv[1:] or sorted(p.parent.name for p in OUT.glob('*/dataset.npz'))
    for name in names:audit(name)
if __name__=='__main__':main()
