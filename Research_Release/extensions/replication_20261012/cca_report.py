                                                                                     
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'Research_Release/replication_20261012_results'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common

def main():
    rows=[]
    for name in json.loads((OUT/'cohorts.json').read_text())['eligible']:
        folder=OUT/name
        status=folder/'cca_identity/status.json'
        if not status.exists():continue
        result=json.loads(status.read_text())
        assert result['status']=='completed'
        assert common.sha(folder/'cca_identity/cases.csv')==result['cases_sha256']
        assert common.sha(folder/'cca_identity/models.pkl')==result['model_sha256']
        identity=result['identity']
        assert common.sha(folder/'dataset.npz')==identity['dataset_sha256']
        assert common.sha(folder/'models.pkl')==identity['cca_model_sha256']
        reference=pd.read_csv(folder/'outage_cases.csv',dtype={'station':str})
        controls=pd.read_csv(folder/'cca_identity/cases.csv',dtype={'station':str})
        expected=set(zip(reference[reference.method=='sheaf_causal'].day,reference[reference.method=='sheaf_causal'].station))
        for method in ['identity_spatial_fixed','identity_spatial_tuned']:
            subset=controls[controls.method==method]
            assert set(zip(subset.day,subset.station))==expected
            assert np.isfinite(subset.mse).all()
        grid=pd.read_csv(folder/'cca_identity/selection_grid.csv')
        choice=result['tuned_identity_choice']
        np.testing.assert_allclose(choice['selection_mse'],grid.selection_mse.min(),rtol=1e-12,atol=1e-12)
        joined=pd.concat([reference,controls],ignore_index=True)
        for scope,frame in [('primary_every_third_day',joined[joined.primary]),('all_eligible_test_days',joined)]:
            row={'cohort':name,'scope':scope}
            for method,label in [('identity_spatial_fixed','fixed_identity'),('identity_spatial_tuned','tuned_identity')]:
                summary=result['summary'][scope][method]['7']
                for metric,key in [('mse','gain_vs_'),('pm25_raw_mse','pm25_gain_vs_')]:
                    left=frame[frame.method=='sheaf_causal'].set_index(['day','station'])[metric]
                    right=frame[frame.method==method].set_index(['day','station'])[metric]
                    matched=pd.concat([left.rename('a'),right.rename('b')],axis=1).dropna()
                    gain=100*(1-np.sqrt(matched.a.sum()/matched.b.sum()))
                    np.testing.assert_allclose(gain,summary[metric]['gain_percent'],rtol=1e-9,atol=1e-9)
                    row[key+label]=float(gain)
                row['cases']=summary['mse']['cases']
            rows.append(row)
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'CCA_IDENTITY_RESULTS.csv',index=False)
    primary=table[table.scope=='primary_every_third_day']
    lines=['# Ancillary spatial CCA controls','',
        'This follow-up is separate from the requested primary sheaf-versus-VAR comparison. '
        'It retains all completed cohorts and losses. Both identity-spatial controls retain temporal CCA; '
        'the comparison does not establish a benefit over a completely CCA-free model or establish novelty.','',
        f"Across {len(primary)} evaluable cohorts, spatial CCA has lower primary normalized RMSE in "
        f"{int((primary.gain_vs_fixed_identity>0).sum())} against the fixed component ablation and "
        f"{int((primary.gain_vs_tuned_identity>0).sum())} against the equally selection-tuned identity-spatial control.",'',
        'CCA_IDENTITY_RESULTS.csv includes primary and secondary outcomes and physical PM2.5 gains. '
        'Each cohort folder retains both models, the full selection grid and case results. '
        'status.json retains seven- and fourteen-day interval sensitivities. '
        'Checks here verify source/model hashes, matched case identities, selection minimum and gain arithmetic; '
        'they do not independently refit these models.','',
        '| Cohort | Cases | CCA gain vs fixed identity (%) | CCA gain vs tuned identity (%) |',
        '|---|---:|---:|---:|']
    for _,r in primary.sort_values('cohort').iterrows():
        lines.append(f'| {r.cohort} | {int(r.cases)} | {r.gain_vs_fixed_identity:.2f} | {r.gain_vs_tuned_identity:.2f} |')
    unavailable=[name for name in json.loads((OUT/'cohorts.json').read_text())['eligible'] if name not in set(primary.cohort)]
    if unavailable:lines+=['','No ancillary result is available for: '+', '.join(unavailable)+'.']
    (OUT/'CCA_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Ancillary spatial CCA tables checked: {len(primary)} cohorts; main VAR results unchanged.')

if __name__=='__main__':main()
