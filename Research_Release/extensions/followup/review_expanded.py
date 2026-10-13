                                                                             

                                                          
   
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]

def week_cluster(dates):
    return ((pd.to_datetime(dates)-pd.Timestamp('1970-01-01'))//pd.Timedelta(days=7)).astype(int)

def main():
    base=ROOT/'results_expanded';out=ROOT/'expanded_docs';rows=[];checks=[];details={}
    for folder in sorted(base.iterdir()):
        if not (folder/'manifest.json').exists():continue
        m=json.loads((folder/'manifest.json').read_text())
        if m['status']=='excluded':continue
        assert m['status']=='complete'
        for name,h in m['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
        frame=pd.read_csv(folder/'prolonged_with_gaussian.csv');assert np.isfinite(frame.mse).all()
        for mode,g in frame.groupby('mode'):
            paired=g.pivot_table(index=['day','station'],columns='method',values='mse')
            for ref in ['stable_var','gaussian','spatial_sheaf','seasonal_persistence']:
                p=paired[['temporal_sheaf',ref]].dropna().reset_index();p['delta']=p[ref]-p.temporal_sheaf;p['week']=week_cluster(p.day)
                w=p.groupby('week').delta.agg(['sum','count']);assert len(w)>1
                rng=np.random.default_rng(907);ids=rng.integers(0,len(w),(4000,len(w)))
                draws=w['sum'].to_numpy()[ids].sum(1)/w['count'].to_numpy()[ids].sum(1)
                lo,hi=np.quantile(draws,[.025,.975]);gain=1-np.sqrt(p.temporal_sheaf.mean()/p[ref].mean())
                rows.append({'region':folder.name,'mode':mode,'reference':ref,'cases':len(p),'days':p.day.nunique(),
                    'week_clusters':len(w),'rmse_gain':float(gain),'week_mse_difference_lower':float(lo),'week_mse_difference_upper':float(hi)})
        summary=json.loads((folder/'summary.json').read_text());fault=json.loads((folder/'fault_structure_summary.json').read_text())
        assert fault['input_model_sha256']==hashlib.sha256((folder/'model.npz').read_bytes()).hexdigest()
        details[folder.name]={'snapshot':summary['snapshot'],'intervals':summary['intervals'],'environment':summary['environment'],
            'two_fault_two_magnitude':[x for x in fault['results'] if x.get('fault_count')==2 and x.get('magnitude')==2.],
            'rank_agreements':fault['rank_agreements'],'rank_sets':fault['rank_sets']}
        checks.append({'region':folder.name,'source_hashes_verified':len(m['code_sha256']),'finite_errors':True,'fault_model_hash_verified':True})
    t=pd.DataFrame(rows);t.to_csv(out/'reviewed_comparisons.csv',index=False)
    (out/'reviewed_details.json').write_text(json.dumps({'checks':checks,'results':details,'correction':'Original seven-day bins assumed nanosecond timestamps; actual pandas resolution was microseconds. Corrected with Timedelta arithmetic. Point estimates unchanged.'},indent=2))
    lines=['# Review of completed expanded results','','All seven eligible network models completed. Mumbai and Kolkata were excluded by the fixed training-coverage rule. Source hashes, finite prolonged-case errors and fault-model hashes were checked.','',
        '**Audit correction:** the original seven-day confidence intervals were degenerate because timestamp integers were interpreted in the wrong units. Use the corrected intervals below; the reconstruction point estimates are unchanged. Do not use the original seven-day columns.','',
        '| Network | Causal comparator | Cases / days | RMSE reduction | Seven-day clusters | Corrected MSE-difference interval |','|---|---|---:|---:|---:|---|']
    for _,r in t[t['mode']=='causal'].iterrows():lines.append(f'| {r.region} | {r.reference} | {r.cases} / {r.days} | {100*r.rmse_gain:.1f}% | {r.week_clusters} | [{r.week_mse_difference_lower:.4f}, {r.week_mse_difference_upper:.4f}] |')
    lines+=['','Positive difference favors the sheaf. Intervals are exploratory dependence-sensitivity estimates, not multiplicity-adjusted significance tests. Networks overlap and are not seven independent replications.','',
        'The raw Delhi causal result is the clearest reconstruction finding. Raw Bengaluru does not outperform seasonal persistence.','',
        'Delhi raw threshold study: 697 common station-days, 142 positives. Sheaf missed 36.6% versus city-mean 56.3%, but false alarms increased from 4.9% to 9.2%. Kriging had higher overall accuracy and fewer false alarms; superiority across operating points is not established. These approximate daily concentration thresholds are not GRAP actions.','',
        'Delhi raw two-fault magnitude-two positive-shift study: known-count localization 95.6%, but exact end-to-end success 7.6% and alarm rate 29.5%, across 342 sampled hours. Fault direction and unknown count matter substantially. The former is not operational fault-detection accuracy.','',
        'Delhi raw fixed-no-label intervals show 93.8% pooled coverage; adaptive revealed-label intervals show 90.7%. Only the fixed setting avoids target-label feedback during the artificial outage; interval widths and low-coverage periods remain relevant. The summary quarter fields partition the test timeline and must not be called actual calendar quarters without the separate replay.','',
        'This review recalculates saved-case comparisons and inspects result summaries; it does not independently repeat all training or every interval/fault trial. The archives were previously examined. No actual pollution, exposure or emissions reduction is demonstrated.']
    (out/'REVIEWED_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
                                                                            
                                                             
    (out/'EXPANDED_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(t[t['mode']=='causal'].to_string(index=False))

if __name__=='__main__':main()
