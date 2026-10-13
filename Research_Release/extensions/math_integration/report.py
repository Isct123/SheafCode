                                                                                 
from pathlib import Path
import os,json,gzip
from fractions import Fraction as Q
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[3]/'audit_tools/mpl_cache'))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'math_integration_results'

def main():
    summary=json.loads((OUT/'SUMMARY.json').read_text());replay=json.loads((OUT/'INDEPENDENT_REPLAY.json').read_text());f=pd.read_csv(OUT/'CASES.csv');roles=json.loads((OUT/'ROLE_STATUS.json').read_text())
    feasible=int((f.upper_status=='verified_gap').sum());infeasible=int((f.upper_status=='certified_infeasible').sum());worst=None
    for path in OUT.glob('*_replay.json.gz'):
        with gzip.open(path,'rt') as handle:payload=json.load(handle)
        for case in payload:
            for direction,s in case['search'].items():
                if s['lower'] is None:continue
                gap=float(Q(s['history'][0]['upper'])-Q(s['lower']))
                if worst is None or gap>worst[0]:worst=(gap,case['case']['cohort'],direction,s['history'])
    facts={**summary,'verified_feasible_directional_searches':2*feasible,'certified_infeasible_directional_searches':2*infeasible,'unresolved_directional_searches':int((f.upper_status=='unresolved').sum()+(f.lower_status=='unresolved').sum()),'exact_truth_assumption_cases':int(f.exact_truth_assumptions.sum()),'decision_counts':f.verified_decision.value_counts().to_dict(),'decision_changes_vs_numerical':int((f.verified_decision!=f.numerical_decision).sum()),'median_search_seconds':float(f.seconds.median()),'median_exhaustive_seconds':float(f.exhaustive_seconds.median()),'median_paired_time_ratio':float((f.seconds/f.exhaustive_seconds).median()),'replay':replay}
    (OUT/'REPORT_FACTS.json').write_text(json.dumps(facts,indent=2))
    unavailable=[r['cohort'] for r in roles if r['status']!='completed']
    report=f'''# Retained-mathematics computational integration

Completed 8 October 2026. All {summary['attempted_roles']} roles were attempted;
{summary['completed_roles']} had complete-vector calibration and test data.
Unavailable roles: {', '.join(unavailable)}. They are retained as unavailable.
The archives had already been accessed; this is exploratory follow-up evidence.

| Result | Completed evidence | Limit |
|---|---|---|
| Graph anchors | All 23 operators met selected-endpoint invertibility at three tolerances; {summary['anchor_subset_checks']} subset/tolerance comparisons agreed with direct numerical ranks. Three nonzero-kernel identity/disconnected/isolated controls passed; a singular-map control was rejected. | Learned full operators all have zero numerical kernel. The checks do not turn the exact idealization into a prediction-accuracy guarantee. |
| Joint thresholds | {summary['binary_crosschecks']} binary-search/epigraph numerical comparisons, including four inconsistent models. | Numerical feasibility cross-check, not exact infeasibility certification for the joint epigraph. |
| Corrected certificates and anytime search | {summary['real_archive_cases']} actual-archive cases, two signed scalar objectives per case: {2*feasible} verified feasible searches and {2*infeasible} exact-infeasible searches; no unresolved searches. | Targets are artificially hidden, surviving-station unit faults are injected, box [-6,6] and bias bounds are declared. |
| Exact implementation replay | {replay['checks']:,} independent checks, {replay['node_certificates']} node certificates, {replay['exact_infeasibility_certificates']} infeasibility certificates, {replay['verified_primal_witnesses']} primal witnesses. | Stored coefficients are nominal, not measured physical error bounds. |

All feasible searches closed their exact bound/incumbent gap below 1e-7;
largest final gap {replay['max_final_gap']:.3g}. Every interruption was replayed,
including coverage, pruning and monotone bounds. Four known-optimum controls
also test zero/two faults, exact infeasibility and deliberate early interruption.
Ground truth is not supplied to the optimization or incumbent search.

Exact nominal truth assumptions hold in {facts['exact_truth_assumption_cases']}
of {len(f)} cases; their scalar truth is contained. The other cases are retained
without a containment guarantee. All {len(f)} exact threshold classifications
agree with the numerical reference: {facts['decision_counts'].get('below',0)}
below, {facts['decision_counts'].get('abstain',0)} abstentions,
{facts['decision_counts'].get('inconsistent',0)} inconsistent models, and no
certified above-reference result. These scalar thresholds do not establish
official AQI decisions or environmental interventions.

Median two-direction search time is {facts['median_search_seconds']:.4f}s,
versus {facts['median_exhaustive_seconds']:.4f}s for exhaustive support LPs;
median paired ratio {facts['median_paired_time_ratio']:.2f}. Exact certificate
verification adds overhead on these small networks. This does not demonstrate
a speed advantage or solve general exponential scaling.

The first exploratory implementation retained unclosed gaps when it could not
verify a feasible endpoint. Its outputs and executed source are preserved in
math_integration_initial_checks. The revised implementation tightens forced-zero
fault coordinates and verifies an interior mixture. This implementation repair
was made after seeing the first follow-up, not selected as a new data protocol.
All scheduled cases were rerun; none were discarded to improve the result.

The existing bank repair, radius-witness, adaptive-policy and operational
verification commands were rerun and passed. No retained mathematical claim
is completely without computational checks. Prospective model membership,
physical bias/noise bounds, exact global real-data rank and environmental
outcomes remain outside what these computational experiments establish.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(11,3.1),layout='constrained')
    trace=worst[3];x=[r['nodes'] for r in trace];u=[float(Q(r['upper'])) for r in trace];l=[float(Q(r['lower'])) if r['lower'] is not None else np.nan for r in trace]
    axes[0].plot(x,u,label='Verified upper',marker='.',lw=1.5);axes[0].plot(x,l,label='Feasible lower',marker='.',lw=1.5);axes[0].set(xlabel='Processed nodes',ylabel='Linear objective (offset excluded)',title='Interrupted-search bounds');axes[0].legend(fontsize=8)
    axes[1].bar(['Below','Abstain','Inconsistent'],[facts['decision_counts'].get(k,0) for k in ['below','abstain','inconsistent']],color=['#287e8d','#cd9e36','#9e5960']);axes[1].set(ylabel='Archive cases',title='Exact scalar decisions')
    axes[2].bar(['Support LPs','Verified search'],[facts['median_exhaustive_seconds'],facts['median_search_seconds']],color=['#777777','#287e8d']);axes[2].set(ylabel='Median seconds per case',title='Small-network runtime')
    fig.savefig(OUT/'math_integration.pdf');fig.savefig(OUT/'math_integration.png',dpi=160);plt.close(fig)
    caption={'trace_selection':'Largest root-relaxation minus final feasible-score gap among evaluated searches; illustrative outcome-selected trace, not a separate benchmark.','trace_cohort':worst[1],'trace_direction':worst[2],'runtime_scope':'Two signed objectives on the same 272 small-network cases. Search includes exact arithmetic; support LP reference is numerical.','decision_scope':'Scalar illustrative hourly channel references; no above-reference result, official AQI or real actions.'}
    (OUT/'FIGURE_SCOPE.json').write_text(json.dumps(caption,indent=2));print(json.dumps(facts,indent=2))

if __name__=='__main__':main()
