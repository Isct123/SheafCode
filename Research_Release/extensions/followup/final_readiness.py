                                                                                
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pymupdf
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def main():
    out=ROOT/'final_verification';out.mkdir(exist_ok=True);cache={};checks=[]
    def digest(p):
        p=Path(p)
        if p not in cache:cache[p]=hashlib.sha256(p.read_bytes()).hexdigest()
        return cache[p]
    comparisons=pd.read_csv(ROOT/'expanded_docs/reviewed_comparisons.csv')
    for region in ['delhi2025','bengaluru_raw','hyderabad','bengaluru','delhi','ncr','india','mumbai','kolkata']:
        folder=ROOT/'results_expanded'/region;m=json.loads((folder/'manifest.json').read_text())
        assert m['status'] in ['complete','excluded'],region
        for path,sha in m['code_sha256'].items():assert digest(ROOT/path)==sha,('Frozen model source changed',region,path)
        for path,sha in m['data_sha256'].items():assert digest(ROOT/path)==sha,('Data changed',region,path)
        if m['status']=='excluded':checks.append({'region':region,'status':'excluded','reason':m.get('reason')});continue
        for name,keys in [('prolonged.csv',['day','station','mode','method']),('snapshots.csv',['day','station','pattern','method'])]:
            frame=pd.read_csv(folder/name);assert np.isfinite(frame.mse).all();assert not frame.duplicated(keys).any(),(region,name)
        frame=pd.read_csv(folder/'prolonged_with_gaussian.csv')
        for mode,g in frame.groupby('mode'):
            pairs=g.pivot(index=['day','station'],columns='method',values='mse')
            for reference in ['stable_var','gaussian','spatial_sheaf','seasonal_persistence']:
                p=pairs[['temporal_sheaf',reference]].dropna();gain=1-np.sqrt(p.temporal_sheaf.mean()/p[reference].mean())
                c=comparisons[(comparisons.region==region)&(comparisons['mode']==mode)&(comparisons.reference==reference)].iloc[0]
                assert c.cases==len(p) and abs(gain-c.rmse_gain)<1e-12
                assert c.week_clusters>1 and c.week_mse_difference_lower<=c.week_mse_difference_upper
        summary=json.loads((folder/'summary.json').read_text());interval=pd.read_csv(folder/'intervals.csv')
        for method,g in interval.groupby('method'):
            stored=summary['intervals'][method];pooled=float((g.coverage*g.n).sum()/g.n.sum())
            assert abs(pooled-stored['pooled_coverage'])<1e-12 and int(g.n.sum())==stored['n_cells']
        faults=pd.read_csv(folder/'fault_trials.csv');fmeta=json.loads((folder/'fault_structure_summary.json').read_text())
        assert fmeta['input_model_sha256']==digest(folder/'model.npz')
        for row in fmeta['results']:
            g=faults[(faults.fault_count==row['fault_count'])&(faults.magnitude==row['magnitude'])&(faults.direction==row['direction'])]
            assert len(g)==row['n_hours']
            for field in ['ls_alarm','heuristic_alarm','ls_exact_end_to_end','heuristic_exact_end_to_end','ls_known_count_exact','heuristic_known_count_exact']:
                assert abs(g[field].mean()-row[field])<1e-12,(region,field)
        checks.append({'region':region,'status':'verified','frozen_source_and_data':True,'comparison_arithmetic':True,'interval_counts':True,'fault_arithmetic':True})
    for directory in ['followup_results','constrained_results']:
        manifest=json.loads((ROOT/directory/'manifest.json').read_text())
        if directory=='constrained_results':
            for name,sha in manifest['source_sha256'].items():assert digest(HERE/name)==sha
            cases=pd.read_csv(ROOT/directory/'cases.csv');assert cases.loc[cases.assumptions_met,'covered'].all()
            assert int(cases.assumptions_met.sum())==manifest['conditional_containment_checks']
        else:
            for region,c in manifest['config'].items():assert digest(ROOT/'results_expanded'/region/'model.npz')==c['model_sha256']
    for name in ['run.py','neural.py']:
        meta=json.loads((ROOT/'followup_results'/('manifest.json' if name=='run.py' else 'neural_control/summary.json')).read_text())
        expected=meta['source_sha256'][name] if isinstance(meta['source_sha256'],dict) else meta['source_sha256']
        assert digest(HERE/name)==expected
    figures=ROOT/'Results_Figure_Pack';fm=json.loads((figures/'manifest.json').read_text())
    for path,sha in fm['source_sha256'].items():assert digest(path)==sha
    assert digest(HERE/'build_figure_pack.py')==fm['generator_sha256']
    with pymupdf.open(figures/'Research_Figures.pdf') as pdf:assert len(pdf)==10
    for path,sha in json.loads((ROOT/'followup_results/release_hashes.json').read_text()).items():assert digest(ROOT/path)==sha,('Packaged file changed',path)
    limitations=['First-ever mathematical priority is unestablished; written proofs and tests are not independent proof review.',
        'No untouched external observations or field intervention outcomes have been supplied.',
        'A published temporal graph model has not been faithfully benchmarked; the custom neural control is clearly labeled.',
        'Complete manuscript and historical-archive reconciliation is not certified by this numeric readiness audit.',
        'Exploratory matched-FPR and bootstrap comparisons are not multiplicity-adjusted confirmatory evidence.',
        'Fault-robust bounds often abstain; direct pollution, health and emissions improvement is not demonstrated.']
    report={'status':'completed_local_results_verified','networks':checks,'distinct_files_hashed':len(cache),'figures':10,'remaining_evidence_limits':limitations}
    (out/'readiness.json').write_text(json.dumps(report,indent=2))
    lines=['# Final local-results readiness audit','','Completed local experiment outputs passed frozen-source/data checks, saved-case reconstruction arithmetic, interval counts, fault-summary arithmetic, conditional containment and figure/provenance checks.','',
        'No further long run is required to document these completed studies. The recommended reconstruction headline remains the raw Delhi causal 23.2% improvement over full-covariance VAR. The figure pack preserves negative results and distinct protocols.','',
        'This is readiness of a defined local experiment package, not scientific certification that all possible improvements are exhausted.','']+['- '+x for x in limitations]
    (out/'READINESS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Final local-results audit passed:',len(checks),'network statuses;',len(cache),'distinct files checked.',flush=True)

if __name__=='__main__':main()
