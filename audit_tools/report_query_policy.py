                                                                                   
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Research_Release/query_policy_results'
WORK=ROOT/'SUBMISSION_WORKSPACE'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    result=json.loads((OUT/'SUMMARY.json').read_text());replay=json.loads((OUT/'INDEPENDENT_REPLAY.json').read_text())
    if replay['status']!='passed':raise RuntimeError('Independent replay required')
    aggregate={}
    for f in [0,1]:
        r=[x for x in result['cohorts'] if x['faults']==f]
        v=sum(x['validated'] for x in r);t=sum(x['training'] for x in r);full=sum(x['full'] for x in r)
        aggregate[str(f)]={'selected_queries':v,'training_queries':t,'full_queries':full,'gain_vs_training':100*(1-v/t),'saving_vs_full':100*(1-v/full),
                           'role_wins':sum(x['validated']<x['training'] for x in r),'role_losses':sum(x['validated']>x['training'] for x in r),'roles':len(r)}
    small=[r for r in result['scaling'] if r['exact_queries'] is not None]
    opt=sum(r['greedy_queries']==r['exact_queries'] for r in small);maxgap=max(r['greedy_queries']-r['exact_queries'] for r in small)
    max_time=max(r['greedy_seconds'] for r in result['scaling']);tight=sum(r['lower_bound']==r['greedy_queries'] for r in result['scaling'])
    report=f'''# Technical hardening: query policies, safe interface and scaling

Completed 8 October 2026. This follow-up adds no new theorem. It uses the same retrospective archives and retains their dependence and artificial-fault limitations.

## Safer operational interface

operational_guard.py accepts a declared station universe and actual queried measurements only. It never issues a label based on scenario membership or a reconstructed value. It returns exceedance, safe or abstain with explicit witness stations and a reason. Missing measurements, threshold ties, malformed inputs and arbitrary-corruption ambiguity cannot be silently treated as safe. Its certificates remain conditional on justified clean-error/fault bounds; these bounds are not automatically verified on a real instrument.

Eight focused tests pass, covering boundary abstention, missing readings, false-high corruption, heterogeneous bounds and invalid inputs. The JSON CLI requires reading_kind="measured" explicitly, although that flag is a caller declaration rather than proof of measurement provenance. Independent integration replay matches {replay['api_integration_checks']:,} sampled stress trials. The example JSON contains invented demonstration readings and must not be presented as a real observation.

From the workspace root:

    audit_tools/python/python.exe Research_Release/extensions/decision_repair/assess_readings.py Research_Release/extensions/decision_repair/example_readings.json

## Validation-only query order

For each of the 18 complete roles and each fault budget, all station permutations are scored on split-1 validation observations only. The selected order and validation score are frozen before opening the existing held-out stress-case table. This is exploratory reuse after prior test access, not blinded external validation. Reordering preserves the direct guard's labels and abstentions; it changes only how early a certificate is reached.

| Fault budget | Selected-order queries | Training-frequency queries | Full-network queries | Saving vs full | Gain vs training frequency | Roles improved / worsened |
|---|---:|---:|---:|---:|---:|---:|
| 0 | {aggregate['0']['selected_queries']:,} | {aggregate['0']['training_queries']:,} | {aggregate['0']['full_queries']:,} | {aggregate['0']['saving_vs_full']:.2f}% | {aggregate['0']['gain_vs_training']:.3f}% | {aggregate['0']['role_wins']} / {aggregate['0']['role_losses']} |
| 1 | {aggregate['1']['selected_queries']:,} | {aggregate['1']['training_queries']:,} | {aggregate['1']['full_queries']:,} | {aggregate['1']['saving_vs_full']:.2f}% | {aggregate['1']['gain_vs_training']:.3f}% | {aggregate['1']['role_wins']} / {aggregate['1']['role_losses']} |

The gains against the simple order are small; validation selection loses in several roles under shift. The earlier 17.24%/11.02% box-prioritized result stays in the manuscript under its own definition. These results are not interchangeable percentages or real monetary savings. Complete paired rows are in PAIRED_COSTS.csv.

## Synthetic scaling and feasible bounds

Sixty fixed-seed sparse/dense multicover instances span 8, 16, 32, 64 and 128 stations, with corruption budgets 0/1. Every greedy cover is checked against every constraint. On 24 cases with exact comparison (at most 16 stations), greedy matches {opt} optima and uses at most {maxgap} extra query in the other cases. Maximum cover-selection time is {max_time:.4f}s on this machine, excluding edge construction and model/data fitting. Timing is not a complexity proof or a deployment guarantee.

Larger cases report feasible upper costs and elementary lower costs. Each constraint needs 2f+1 selected stations; divide total required hits by the largest per-station constraint degree, rounding up. If a constraint has exactly 2f+1 members, all are forced: their union supplies another lower bound. The stronger forced-item check was added after the first exploratory scaling check, without changing generated cases, selected policies or their cost objective. Bounds equal the greedy cost on {tight}/60 cases. Other gaps are retained; large-instance global optimality is not asserted.

Independent replay reconstructs all 36 validation choices using a separate mask-frequency objective, all {replay['paired_trials']:,} held-out query costs and every scaling cover. It independently enumerates the 24 small synthetic optima and passes {replay['checks']:,} checks. This is internal reproducibility, not an independent field study.

## Remaining boundaries

The safer interface removes operational dependence on unvalidated scenario membership, but scenario-based theoretical results remain conditional. It does not fix real-world sensor-error bounds, extreme-event interval undercoverage, dependence between archived roles, independent student authorship or measured environmental impact. Those facts are not supplied by code. The practical order optimizer is exhaustive only for the small observed 3–5-station networks; scalable greedy repair supplies valid approximate plans, not exact large-network optimization.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8');(WORK/'Guides/Technical_Hardening.md').write_text(report,encoding='utf-8')
    source=WORK/'Source/manuscript.tex';audit=json.loads((WORK/'Guides/WORKSPACE_DOCUMENT_CHECK.json').read_text())
    if sha(source)!=audit['source_sha256']:raise RuntimeError('Personal source edits detected; report saved, automatic insertion refused')
    text=source.read_text(encoding='utf-8');marker=r'\section{Reliability, limitations and conclusion}'
    addition=rf'''
\textbf{{Validation-only ordering and scalable approximate repair.}} A further exploratory control selects each order from all station permutations using only the separate validation block, before scoring the later-date trials. Direct certificates and abstentions remain unchanged. The selected orders use {aggregate['0']['selected_queries']:,}/{aggregate['1']['selected_queries']:,} queries without/with one corruption, versus {aggregate['0']['training_queries']:,}/{aggregate['1']['training_queries']:,} for training-frequency order: gains of only {aggregate['0']['gain_vs_training']:.3f}\%/{aggregate['1']['gain_vs_training']:.3f}\%, with losses in {aggregate['0']['role_losses']}/{aggregate['1']['role_losses']} of eighteen roles. These are small dependent aggregate gains, not universal superiority. A separate queried-reading interface returns witness stations and explicit abstentions; it does not accept scenario membership as evidence of a measured exceedance.

Sixty fixed-seed synthetic sparse/dense multicover cases test greedy repair up to 128 stations. All returned covers satisfy every constraint. On 24 cases with exact comparison (at most 16 stations), greedy matches {opt} optima and uses at most {maxgap} extra query otherwise. Maximum greedy selection time is {max_time:.4f}s here, excluding edge generation and fitting. Larger cases retain feasible-cost upper bounds and elementary hit-count/forced-item lower bounds, without asserting global optimality. An independent mask-frequency replay verifies all 36 validation choices, all {replay['paired_trials']:,} paired query costs and the small synthetic optima. Synthetic scaling and a safe software interface strengthen implementation evidence; they do not establish real instrument-error bounds or field impact.

'''
    if text.count(marker)!=1:raise RuntimeError('Ambiguous manuscript marker')
    source.write_text(text.replace(marker,addition+marker,1),encoding='utf-8')
    (OUT/'MANUSCRIPT_ADDITION.tex').write_text(addition,encoding='utf-8')
    (OUT/'AGGREGATES.json').write_text(json.dumps({'query_orders':aggregate,'scaling_cases':60,'small_optima_matched':opt,'max_small_gap':maxgap,'max_greedy_seconds':max_time,'tight_bound_cases':tight},indent=2))
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
