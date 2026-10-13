                                                                                 
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def main():
    out=ROOT/'constrained_results';f=pd.read_csv(out/'cases.csv');summary=pd.read_csv(out/'summary.csv');m=json.loads((out/'manifest.json').read_text())
    for name,h in m['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h
    checks=0
    for keys,g in f.groupby(['region','horizon','scenario','method']):
        row=summary[(summary.region==keys[0])&(summary.horizon==keys[1])&(summary.scenario==keys[2])&(summary.method==keys[3])].iloc[0]
        assert len(g)==row.cases;assert abs(g.covered.mean()-row.coverage)<1e-12
        assert abs((g.decision!='abstain').mean()-row.decided_fraction)<1e-12;checks+=3
    assumption=f[f.assumptions_met];assert assumption.covered.all();assert len(assumption)==m['conditional_containment_checks']
                                                                             
                                                                            
    p=f.pivot(index=['region','timestamp','station','horizon','scenario'],columns='method',values=['lower','upper'])
    mask=p['lower']['joint_temporal'].notna()&p['lower']['joint_budget_no_history'].notna()
    assert (p['lower']['joint_temporal'][mask]>=p['lower']['joint_budget_no_history'][mask]-1e-5).all()
    assert (p['upper']['joint_temporal'][mask]<=p['upper']['joint_budget_no_history'][mask]+1e-5).all()
    matched=pd.read_csv(out/'matched_false_alarm_comparisons.csv');bench=pd.read_csv(out/'budget_dual_benchmark.csv')
    verification=json.loads((out/'exact_certificate_manifest.json').read_text())
    assert verification['status']=='completed'
    for name,h in verification['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h
    clean=summary[(summary.horizon==24)&(summary.scenario=='clean')]
    lines=['# Completed constrained-quantity research study','','The study completed, including negative results. It scored '+str(len(f))+' method/case rows (not independent observations), with '+str(len(assumption))+' conditional containment checks. '+str(checks)+' summary arithmetic checks and matched-budget temporal nestedness checks passed.','',
        '## Theoretical contributions','',
        'Sharp bounded-fault support-union envelopes; jointly calibrated residual/causal-history constraints; polynomial-size support-free dual certificates. The dual is generally conservative, with a tested strict-gap example. Proofs and prior-work attribution are in CONSTRAINED_QUANTITY_THEORY.md. First-ever mathematical priority is not established.','',
        '## 24-hour clean-scenario evaluation','',clean.to_string(index=False),'',
        'Different residual norms/budgets must not be treated as nested. The joint-temporal method is nested only against joint-budget-no-history. Calibration uses an earlier archive-test period after model selection, evaluation uses the later period. Both were previously examined archives.','',
        'Temporal information can shrink bounds substantially, especially in Bengaluru, but threshold decisions remain rare. The new bounds are not a demonstrated operational alert success. Full scenarios and 72-hour results are in summary.csv.','',
        '## Matched false-alarm evaluation','',matched[matched.target_fpr==.05].to_string(index=False),'',
        'On raw Bengaluru with no additional cell missingness, empirical sensitivity at a matched 5% false-positive rate was 61.25% versus 40.75% city fallback: a 20.5 percentage-point difference. Its exploratory paired-day bootstrap interval was approximately [2.9,38.5] percentage points. There were 80 positive cases. These nominal intervals are not adjusted across all 18 comparisons. Under 50% additional missingness this advantage disappeared. Delhi differences were uncertain.','',
        'These are descriptive empirical ROC operating points with interpolated randomized cutoffs. Do not claim they are validation-frozen deployed thresholds or evidence of reduced emissions/exposure. Original validation-selected alert results remain in followup_results.','',
        '## Computation versus tightness','',bench.groupby(['fault_stations','budget'])[['exact_seconds','dual_seconds','speedup','exact_width','dual_width']].median().to_string(),' ',
        'For 18 synthetic candidate fault stations and budget 3, the median support-free computation was approximately 408 times faster over three repeats, but its envelope was wider. This is a synthetic runtime/tightness tradeoff, not demonstrated real-network acceleration.','',
        '## Exact nominal certificate verification','',
        str(verification['cases'])+' perturbed floating-point solver vectors were checked using exact rational residual corrections. See VERIFIED_CERTIFICATE_THEOREM.md. The exact statement concerns the stored nominal matrix model and trusted finite state bounds; it does not certify physical model validity or first-ever priority.','',
        '## Evidence boundaries','',
        'Bias bound two standardized units is a declared stress scenario, not an independently established bound on real instruments. Causal history assumes the last pre-outage target vector is trustworthy. Joint score calibration has an exchangeability-conditional interpretation, not a guarantee under arbitrary pollution shifts. Scores are sampled outage endpoints, not every outage hour. LP feasibility/equality uses numerical tolerances. Hourly PM thresholds are illustrative, not official AQI/GRAP. Independent external and field intervention evidence are still absent.']
    (out/'CONSTRAINED_RESEARCH_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4));colors=['#245c87','#b55a2a','#597a35','#765795']
    methods=['unrestricted_fault','bounded_fault','joint_budget_no_history','joint_temporal']
    for i,region in enumerate(['delhi2025','bengaluru_raw']):
        g=clean[clean.region==region].set_index('method').loc[methods]
        axes[i].bar(np.arange(4),g.coverage*100,color=colors);axes[i].set_ylim(0,105);axes[i].set_xticks(np.arange(4),['Unrestricted','Bounded','Joint budget','+ History'],rotation=25)
        axes[i].set_title(region);axes[i].set_ylabel('Empirical coverage (%)')
        for j,(_,row) in enumerate(g.iterrows()):axes[i].text(j,4,f'Decided: {100*row.decided_fraction:.1f}%',rotation=90,ha='center',va='bottom',color='white')
    fig.suptitle('24-hour clean-scenario endpoints: coverage and decision yield');fig.tight_layout();fig.savefig(out/'coverage_and_decision_yield.png',dpi=180);plt.close(fig)
    paths=[p for p in out.rglob('*') if p.is_file() and p.name!='review_hashes.json']+[p for p in HERE.glob('*') if p.suffix in ['.py','.md','.ps1']]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};(out/'review_hashes.json').write_text(json.dumps(hashes,indent=2))
    with zipfile.ZipFile(ROOT/'Constrained_Research_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(ROOT))
        z.write(out/'review_hashes.json','constrained_results/review_hashes.json')
    print('Completed report, plot and result archive; arithmetic checks:',checks,flush=True)

if __name__=='__main__':main()
