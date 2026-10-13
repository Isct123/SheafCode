                                                                                    
from pathlib import Path
import ast, csv, hashlib, json, os, re, shutil, subprocess, zipfile
import pymupdf as fitz
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'IRIS_Submission_Recent_Data'
SRC = PKG / 'Editable_Manuscript'
OUT = PKG / 'Upload_Files/SheafCode_Concise_Manuscript.pdf'
OLD = PKG / 'Upload_Files/SheafCode_Full_Empirical_Manuscript.pdf'
AUDIT = PKG / 'Review_Notes/CONCISE_MANUSCRIPT_AUDIT.json'

ABSTRACT = '''Air-quality stations can stop reporting while neighbouring sensors remain active. SheafCode learns spatial and temporal comparisons to reconstruct missing readings and develops decision-first reliability reasoning under ambiguous or faulty observations. The fixed Indian causal benchmark reduces normalized RMSE by 16.48% against a tuned full-covariance vector autoregressive (VAR) filter across 77 paired station-days (seven-day calendar-bin interval: 12.62–20.16%); physical PM2.5 error falls 10.45%. An exploratory expansion adds 20 evaluable US, UK and Australian cohorts, with 73 station memberships and 2,114 paired station-days. Causal sheaf has lower normalized error in 13/20 cohorts against VAR and 18/20 against a tuned causal random forest. Retrospective neural comparisons retain every seed, including Australian losses. The mathematical contribution connects graph anchors, joint pollutant decisions, corruption-aware measurement repair, continuous-box stability radii and adaptive query policies. Complete proofs use graph traversal, weighted inequalities, separating sets, order statistics and minimax recursion; underlying theory is credited. Stress tests retain unequal-fault localization failures, extreme-event interval undercoverage and alert sensitivity/false-alarm tradeoffs. Environmental benefit concerns computational monitoring continuity; no emissions reduction or health benefit was measured.'''

def read_csv(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def escape(s):
    return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

def table(caption, columns, header, rows, note=''):
    return (r'\begin{table}[htbp]\centering\small' + '\n' +
            r'\caption{' + caption + '}\n' + r'\begin{tabular}{' + columns + r'}\toprule' + '\n' +
            header + r'\\\midrule' + '\n' + '\n'.join(' & '.join(row) + r'\\' for row in rows) +
            '\n' + r'\bottomrule\end{tabular}' + '\n' +
            (r'\par\smallskip\small ' + note + '\n' if note else '') + r'\end{table}')

def main():
    original_hash = sha(OLD)
    cohorts_all = read_csv('Research_Release/expanded_validation_results/COHORT_RESULTS.csv')
    cohorts = [r for r in cohorts_all if r['provider'] != 'OpenAQ']
    forest = {r['cohort']: r for r in read_csv('Research_Release/weakness_resolution_results/NONLINEAR_SUMMARY.csv')}
    neural = read_csv('Research_Release/weakness_resolution_results/NEURAL_SUMMARY.csv')
    risk = read_csv('Research_Release/weakness_resolution_results/RISK_INTERVAL_AGGREGATES.csv')
    assert len(cohorts) == 20 and len(neural) == 9
    assert sum(int(r['cases']) for r in cohorts) == 2114
    assert sum(int(r['stations']) for r in cohorts) == 73
    assert sum(int(r['observed_pollutant_hour_cells']) for r in cohorts) == 549302
    var_wins = sum(float(r['gain']) > 0 for r in cohorts)
    rf_wins = sum(float(forest[r['cohort']]['gain']) > 0 for r in cohorts)
    assert (var_wins, rf_wins) == (13, 18)
    rows = []
    for r in cohorts:
        name = r['cohort'].replace('epa_', 'EPA ').replace('uk_', 'UK ').replace('nsw_', 'NSW ').replace('_', ' ')
        rows.append([escape(r['code']), escape(name), r['cases'], f"{float(r['gain']):.2f}",
                     f"{float(forest[r['cohort']]['gain']):.2f}", f"{float(r['pm25_gain']):.2f}"])
    cohort_table = table(r'Every evaluable external cohort: gain (\%) on matched station-days. Negative values are losses.',
                         'llrrrr', r'Code & Region & Cases & vs VAR & vs RF & PM2.5 vs VAR', rows,
                         'VAR/RF columns use normalized multichannel error; the final column uses physical PM2.5 error. Codes map to retained station IDs. EPA 06-073 and 06-111 lack usable test observations and are excluded, not counted as successful runs. Weekly intervals are plotted below; complete split counts and fourteen-day sensitivities remain in the technical reference.')
    cohort_table = cohort_table.replace(r'\begin{tabular}', r'\label{tab:external}\begin{tabular}', 1)
    primary = table('Fixed original primary comparisons against causal VAR. Gains are normalized; each row is a separate network.',
                    'lrrrr', r'Network & Station-days & Dates & Gain (\%) & Weekly interval', [
                        ['India (A)', '77', '16', '16.48', '12.62--20.16'],
                        ['USA (B)', '24', '6', '8.68', 'Suppressed'],
                        ['UK (C)', '60', '12', '3.79', '1.78--7.83']],
                    'Intervals use paired calendar bins. US PM2.5-only comparisons have 23 eligible cases; normalized comparisons have 24.')
    neural_rows = [[r['code'], r['seed'], r['cases'], f"{float(r['gain']):.2f}",
                    '--'.join(f'{x:.2f}' for x in ast.literal_eval(r['ci7']))] for r in neural]
    original_neural = json.loads((ROOT / 'Research_Release/recent_data_results/delhi/grin_summary.json').read_text())['comparisons']
    original_cases = read_csv('Research_Release/recent_data_results/delhi/outage_cases.csv')
    grin_cases = read_csv('Research_Release/recent_data_results/delhi/grin_cases.csv')
    original_neural_checks = {}
                                                                                     
    convert = lambda gain: 100 * (1 - 1 / (1 - gain / 100))
    for seed, summary in sorted(original_neural.items()):
        truth = {(r['day'], r['station']): float(r['mse']) for r in original_cases if r['method'] == 'sheaf_retrospective'}
        cases = [r for r in grin_cases if r['seed'] == seed]
        gain = 100 * (1 - (sum(truth[(r['day'], r['station'])] for r in cases) / sum(float(r['mse']) for r in cases)) ** .5)
        assert len(cases) == summary['cases'] == 242
        assert abs(gain - convert(summary['gain_percent'])) < 1e-9
        ci = [convert(summary['ci95'][1]), convert(summary['ci95'][0])]
        neural_rows.insert(int(seed), ['A', seed, str(len(cases)), f'{gain:.2f}', '--'.join(f'{x:.2f}' for x in ci)])
        original_neural_checks[seed] = {'gain_vs_grin': gain, 'ci95': ci, 'cases': len(cases)}
    primary_summary = json.loads((ROOT / 'Research_Release/recent_data_results/delhi/reconstruction_summary.json').read_text())['primary_every_third_day']
    primary_case_checks = {}
    for method in ('var_causal', 'seasonal', 'city_mean', 'gaussian', 'spatial_sheaf'):
        sheaf = {(r['day'], r['station']): float(r['mse']) for r in original_cases if r['method'] == 'sheaf_causal' and r['primary'] == 'True'}
        baseline = [r for r in original_cases if r['method'] == method and r['primary'] == 'True']
        gain = 100 * (1 - (sum(sheaf[(r['day'], r['station'])] for r in baseline) / sum(float(r['mse']) for r in baseline)) ** .5)
        assert len(baseline) == primary_summary[method]['cases'] == 77
        assert abs(gain - primary_summary[method]['gain_percent']) < 1e-9
        primary_case_checks[method] = gain
    primary_pm = [r for r in original_cases if r['primary'] == 'True' and r['method'] in ('sheaf_causal', 'var_causal')]
    pm_loss = {method: sum(float(r['pm25_raw_mse']) for r in primary_pm if r['method'] == method) for method in ('sheaf_causal', 'var_causal')}
    pm_gain = 100 * (1 - (pm_loss['sheaf_causal'] / pm_loss['var_causal']) ** .5)
    assert abs(pm_gain - primary_summary['pm25_physical']['var_causal']['gain_percent']) < 1e-9
    span_checks = {}
    for region, expected in [('delhi', 11), ('epa_la', 7)]:
        records = read_csv(f'Research_Release/recent_data_results/{region}/exact_nuisance_span_counts.csv')
        assert all(int(r['support_count']) == int(r['distinct_exact_decimal_spans']) == expected for r in records)
        span_checks[region] = {'targets': len(records), 'supports_each': expected, 'distinct_spans_each': expected}
    neural_table = table(r'Every tested GRIN seed: normalized retrospective sheaf gain (\%).', 'lrrrr',
                         'Cohort & Seed & Cases & Gain vs GRIN & Weekly interval', neural_rows,
                         'A = original Indian all-day comparison; D1 = EPA 04-013; E1 = AURN central; F1 = NSW Illawarra. All gains use sheaf as numerator. Seeds share truth observations and are not independent datasets.')
    selected = [r for r in risk if float(r['loss']) == 0 and r['method'] in ('station_fixed', 'risk_station')]
    risk_rows = []
    for group in ('all', 'extreme'):
        for method in ('station_fixed', 'risk_station'):
            r = next(r for r in selected if r['group'] == group and r['method'] == method)
            risk_rows.append(['All' if group == 'all' else 'Extreme', 'Fixed' if method == 'station_fixed' else 'Risk grouped',
                              r['cells'], f"{100*float(r['macro_coverage']):.1f}", f"{100*float(r['pooled_coverage']):.1f}",
                              f"{float(r['macro_finite_width']):.2f}", f"{100*float(r['full_fraction']):.3f}"])
    interval_table = table(r'No-feedback PM2.5 snapshot intervals, nominal 90\%, without additional context loss.',
                           'llrrrrr', r'Group & Method & Cells & Macro (\%) & Pooled (\%) & Width & Full (\%)', risk_rows,
                           r'Macro averages cohort coverage (20 overall; 19 nonempty extreme cohorts). Pooled coverage weights cells. Width is the cohort-macro mean finite full interval width in native PM2.5 units; full intervals are infinite and excluded from finite widths. Full frequency is cell pooled. Worst individual station-quarter coverage remains 57.8\% for fixed and 74.6\% for risk-grouped overall intervals.')
    repair = json.loads((ROOT / 'Research_Release/decision_repair_results/SUMMARY.json').read_text())
    repair_verification = json.loads((ROOT / 'Research_Release/decision_repair_results/VERIFICATION.json').read_text())
    assert repair['status']=='completed_and_checked' and repair_verification['status']=='passed'
    adaptive = json.loads((ROOT / 'Research_Release/decision_repair_results/ADAPTIVE_SUMMARY.json').read_text())
    adaptive_verification = json.loads((ROOT / 'Research_Release/decision_repair_results/ADAPTIVE_VERIFICATION.json').read_text())
    adaptive_replay = json.loads((ROOT / 'Research_Release/decision_repair_results/ADAPTIVE_POLICY_REPLAY.json').read_text())
    assert adaptive['status']=='completed' and adaptive_verification['status']==adaptive_replay['status']=='passed'
    assert all(r['search_limit']==0 for r in adaptive['summaries'].values())
    radius_report=json.loads((ROOT/'Research_Release/decision_repair_results/RADIUS_SUMMARY.json').read_text())
    radius_verification=json.loads((ROOT/'Research_Release/decision_repair_results/RADIUS_VERIFICATION.json').read_text())
    radius_replay=json.loads((ROOT/'Research_Release/decision_repair_results/RADIUS_WITNESS_REPLAY.json').read_text())
    assert radius_report['status']=='completed' and radius_verification['status']=='passed'
    assert radius_replay['status']=='passed'
    assert radius_verification['source_sha256']['radius_repair.py']==sha(ROOT/'Research_Release/extensions/decision_repair/radius_repair.py')
    radius_rows=read_csv('Research_Release/decision_repair_results/RADIUS_CONFIGURATIONS.csv')
    assert len(radius_rows)==414
    repair_rows = []
    for f in ['0','1']:
        r = repair['summaries_by_query_fault_budget'][f]
        repair_rows.append([f,str(r['nontrivial_banks']),str(r['decision_feasible']),str(r['decision_possible_pattern_impossible']),
                            str(r['both_feasible']),str(r['decision_query_sum_common'])+'/'+str(r['pattern_query_sum_common'])])
    repair_table = table('Exact finite archive-scenario repair. Counts are dependent configurations, not independent trials.',
                         'rrrrrl','Query faults & Two labels & Decision feasible & Pattern impossible & Both feasible & Query totals',repair_rows,
                         'Pattern impossible counts only settings with a feasible decision plan. Query totals are decision/full-pattern queries on the common feasible set. Empty and one-label banks are excluded. With one corrupted query, 46/61 two-label settings have no robust decision plan.')
    repair_text = (r'\subsection{A new capability: decision-specific repair planning}'+'\n'+
        r'The repair theorem is implemented with exact rational scenario comparisons and an exhaustive small-network multicover master. At most 24 evenly spaced fully observed PM2.5 test hours per cohort define the declared offline possibilities. The decision is whether any selected station exceeds the illustrative PM2.5 reference. All 23 existing cohort roles are attempted; 18 provide complete station vectors. Fixed references 15/35/60 $\mu$g/m$^3$, error radii 1/3/5 and query-corruption budgets zero/one produce 414 settings. States within the declared reference margin are excluded and counted; this is a benchmark restriction, not knowledge available about hidden field truth.'+'\n\n'+
        r'This experiment asks how many measurements distinguish decisions within a known archived scenario bank. The comparison task identifies the entire archived scenario, not arbitrary continuous pollution vectors. It uses archived test values as possible states, so it is neither a prospective forecast nor an independent repair deployment. Membership in this finite bank and its decision labels is an assumption. Configurations reuse the same observations; they are not independent samples.'+'\n\n'+repair_table+'\n'+
        r'Without corrupted queries, decision plans are feasible in all 61 two-label settings. Across the ten where both tasks are feasible, they require 18 queries versus 24 for full-pattern identification (25\% fewer); six are strictly cheaper. In 51 other settings, the decision is resolvable although full-pattern identification is impossible at the declared reading tolerance. With one corrupted query, only 15/61 decision plans are feasible: robust repair is not generally available. Every completed plan matches exhaustive optimization; boundary-noise/corrupted-response replays and 160 randomized banks test correctness, not physical validity.'+'\n\n'+
        r'The contribution is witness-driven, minimum-cost decision repair with explicit impossibility outcomes. Credited multicover and error-correction theory supplies its foundations; the evaluated monitoring formulation connects these to pollution decisions.'+'\n')
    a0,a1 = adaptive['summaries']['0'],adaptive['summaries']['1']
    repair_text += (r'\textbf{Adaptive continuation of the same repair problem.} The exact minimax dynamic program reuses every one of the 414 settings, with the same bank, margin, noise and corruption assumptions. It optimizes worst-case queries over every real response class, including interval boundaries and exterior readings; each station is queried once. All settings finish below the declared 50,000-state limit. With no corrupted query, adaptive repair reduces the summed worst-case cost from '+str(a0['fixed_query_sum_common'])+' to '+str(a0['adaptive_query_sum_common'])+' across '+str(a0['optimal'])+r' feasible settings, improving only '+str(a0['strictly_cheaper'])+r'. With one corrupted query it reduces '+str(a1['fixed_query_sum_common'])+' to '+str(a1['adaptive_query_sum_common'])+' across '+str(a1['optimal'])+r' feasible settings ('+f"{a1['saving_percent_common']:.2f}"+r'\% fewer), improving '+str(a1['strictly_cheaper'])+r'; the same 46 settings remain impossible. These are sums of per-setting worst-case costs across dependent configurations, not expected field expenditure. Eighty randomized small banks agree with a separate unpruned exhaustive implementation, and every exported policy is replayed by a solver-independent response-class audit. The adaptive method extends the repair capability; it neither improves prediction RMSE nor establishes real-world pollution reduction.'+'\n')
    r0,r1=radius_report['summaries']['0'],radius_report['summaries']['1']
    repair_text += (r'\textbf{Continuous-neighbourhood transfer and the budget frontier.} The stability theorem evaluates every subset and attainable budget on the same 414 settings. At the minimum fixed-plan cost, maximizing the label-preserving interior radius improves '+str(r0['strict_same_cost_improvements'])+'/'+str(r0['feasible'])+' feasible no-corruption settings and '+str(r1['strict_same_cost_improvements'])+'/'+str(r1['feasible'])+' feasible one-corruption settings (Figure~'+r'\ref{fig:repair-radius}'+r'). Increasing the budget improves a further '+str(r0['strict_larger_budget_improvements'])+' and '+str(r1['strict_larger_budget_improvements'])+r' relative to each same-cost optimum. Every setting and budget is retained. The theorem reports conditional tolerances, not calibrated coverage: concentrations must lie in a declared scenario box, and radii must stay strictly below both the separation limit and centre-to-threshold semantic margin. '+str(radius_report['boundary_witnesses'])+r' exact boundary counterexamples and 100 independently checked small banks test the sharp separation condition. A larger coupled feasible model still needs its own coverage and compatibility validation.'+'\n'+
        r'\begin{figure}[htbp]\centering\includegraphics[width=.94\linewidth]{figures/repair_radius_frontier.pdf}'+'\n'+
        r'\caption{At the same minimum number of fixed queries, choose the plan with the largest decision-preserving continuous-box radius. Every feasible configuration is plotted (61 without corruption, 15 with one corrupt query); settings reuse archive observations. Radii are capped by the semantic threshold margin and are valid strictly inside the shown limit. Points above the diagonal improve a conditional tolerance without additional queries; overlapping points remain plotted. These are not field uncertainty estimates or independent replications.}\label{fig:repair-radius}\end{figure}'+'\n')
    abstract = ABSTRACT.replace('normalized root mean station-day squared error', 'normalized RMSE (root mean station-day MSE)')
    abstract = abstract.replace('graph anchors, breadth-first search over equivalent unrestricted fault effects, and joint pollutant thresholds under bounded faults', 'graph anchors, joint pollutant thresholds under bounded faults, and minimum-cost decision repair with conflict witnesses')
    abstract += ' A finite archive-scenario benchmark saves 25% of repair queries across ten settings where decision and full-pattern identification are both feasible; the guarantee is conditional on the scenario bank.'
    abstract += ' A stability-radius theorem extends repair guarantees to declared continuous perturbations of archived scenarios.'
    abstract_tex = escape(abstract).replace('–', '--')
    replacements = {'ABSTRACT': abstract_tex, 'PRIMARY_TABLE': primary, 'COHORT_TABLE': cohort_table,
                    'NEURAL_TABLE': neural_table, 'INTERVAL_TABLE': interval_table, 'REPAIR_RESULTS':repair_text,
                    'RADIUS_THEOREM': (SRC/'repair_radius_theorem.tex').read_text(encoding='utf-8'),
                    'EXTERNAL_SUMMARY': f'Causal sheaf improves numerically over VAR in {var_wins}/20 external cohorts and over the causal forest in {rf_wins}/20. These counts are not adjusted significance claims. UK central illustrates comparator dependence: sheaf loses to VAR but beats the tested GRIN seeds. Australian Lower Hunter loses to VAR, while the selected Illawarra neural comparison also favors GRIN. All external reconstruction outcomes, including losses, are retained in Table below and Figure~\\ref{{fig:external}}.'}
    source = (SRC / 'manuscript_template.tex').read_text(encoding='utf-8')
    for key, value in replacements.items():
        source = source.replace('@@' + key + '@@', value)
    source = source.replace('Table below', r'Table~\ref{tab:external}')
    source = source.replace(r'$\square$', r'\textit{QED.}')
    assert '@@' not in source
    (SRC / 'manuscript.tex').write_text(source, encoding='utf-8')
    bib_source = (PKG / 'Expanded_Evidence_Source/references.bib').read_text(encoding='utf-8')
    extra = SRC / 'additional_references.bib'
    if extra.exists():
        bib_source += '\n' + extra.read_text(encoding='utf-8')
    (SRC / 'references.bib').write_text(bib_source,encoding='utf-8')
    figs = SRC / 'figures'
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.add_patch(Rectangle((0, 0), .75, .75, facecolor='#ddebd7', edgecolor='#50734a', alpha=.8))
    ax.plot([0, 2], [2, 0], lw=2.5, color='#176b91', label='Feasible explanations: a + b = 2')
    ax.scatter([0, 2, 1], [2, 0, 1], color='#176b91', s=28)
    ax.text(.06, .25, 'Both scores ≤ 1', fontsize=9)
    ax.annotate('Joint minimum at (1, 1)\nscore = 4/3', (1, 1), (1.1, 1.45), arrowprops={'arrowstyle': '->'}, fontsize=9)
    ax.set(xlim=(0, 2.15), ylim=(0, 2.15), xlabel='Illustrative concentration a', ylabel='Illustrative concentration b')
    ax.set_aspect('equal'); ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.03, 1.2))
    fig.savefig(figs / 'joint_example.pdf'); fig.savefig(figs / 'joint_example.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for i, r in enumerate(cohorts):
        var_ci = [float(r['ci_low']), float(r['ci_high'])] if r['ci_low'] and r['ci_high'] else None
        forest_ci = ast.literal_eval(forest[r['cohort']]['ci7']) if forest[r['cohort']]['ci7'] else None
        for delta, col, value, ci in [(-.12, '#176b91', float(r['gain']), var_ci),
                                     (.12, '#a95818', float(forest[r['cohort']]['gain']), forest_ci)]:
                                                                                                   
            if ci is not None:
                ax.plot(ci, [i+delta, i+delta], color=col, lw=1.1)
            ax.plot([value], [i+delta], 'o', color=col, ms=4)
    ax.plot([], [], 'o-', color='#176b91', label='vs causal VAR')
    ax.plot([], [], 'o-', color='#a95818', label='vs causal random forest')
    ax.axvline(0, color='black', lw=.8)
    ax.set_yticks(range(len(cohorts)), [r['code'] for r in cohorts]); ax.invert_yaxis()
    ax.set_xlabel('Normalized RMSE gain (%) — positive favors causal sheaf')
    ax.legend(loc='lower right', fontsize=9); ax.grid(axis='x', alpha=.2)
    fig.savefig(figs / 'external_comparisons.pdf'); fig.savefig(figs / 'external_comparisons.png', dpi=180); plt.close(fig)
    shutil.copy2(PKG / 'Expanded_Evidence_Source/figures/followup_risk.pdf', figs / 'followup_risk.pdf')
    us_primary = json.loads((ROOT / 'Research_Release/recent_data_results/epa_la/reconstruction_summary.json').read_text())['primary_every_third_day']
    us_cases = read_csv('Research_Release/recent_data_results/epa_la/outage_cases.csv')
    us_sheaf = {(r['day'],r['station']):float(r['mse']) for r in us_cases if r['method']=='sheaf_causal' and r['primary']=='True'}
    us_primary_checks = {}
    for method in ('var_causal', 'seasonal', 'city_mean', 'gaussian', 'spatial_sheaf'):
        baseline = [r for r in us_cases if r['method']==method and r['primary']=='True']
        gain = 100 * (1 - (sum(us_sheaf[(r['day'],r['station'])] for r in baseline)/sum(float(r['mse']) for r in baseline))**.5)
        assert len(baseline)==us_primary[method]['cases']==24
        assert abs(gain-us_primary[method]['gain_percent'])<1e-9
        us_primary_checks[method]=gain
    controls = [('var_causal','Causal VAR'), ('seasonal','Seasonal'), ('city_mean','City mean'), ('gaussian','Gaussian'), ('spatial_sheaf','Spatial sheaf')]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    for ax, title, summary in [(axes[0], 'India: 77 cases / 16 dates', primary_summary),
                               (axes[1], 'USA: 24 cases / 6 dates', us_primary)]:
        for i, (method, label) in enumerate(controls):
            item = summary[method]; gain = item['gain_percent']
            color = '#a95818' if method == 'var_causal' else '#176b91'
            if item.get('ci95'):
                ax.plot(item['ci95'], [i,i], color=color, lw=1.7)
            ax.plot(gain, i, 'o', color=color, ms=5)
        ax.axvline(0, color='black', lw=.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-3,60); ax.set_xticks([0,20,40,60]); ax.tick_params(labelsize=10)
        ax.grid(axis='x', alpha=.15)
    axes[0].set_yticks(range(5), [label for _,label in controls], fontsize=10)
    axes[0].invert_yaxis()
    fig.supxlabel('Causal sheaf normalized RMSE gain (%)', fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / 'primary_reconstruction.pdf'); fig.savefig(figs / 'primary_reconstruction.png', dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.1,3.35))
    for fault,ax in enumerate(axes):
        rs=[r for r in radius_rows if r['query_faults']==str(fault) and r['status']=='certified_inside_radius']
        xs=[float(r['baseline_radius']) for r in rs];ys=[float(r['same_cost_radius']) for r in rs]
        limit=max(xs+ys)*1.07
        ax.plot([0,limit],[0,limit],color='gray',linestyle='--',linewidth=1)
        ax.scatter(xs,ys,color='#236c91',s=27,alpha=.6)
        ax.set(xlim=(0,limit),ylim=(0,limit),title=f'{fault} corrupt query: {len(rs)} settings',
               xlabel='Original plan radius',ylabel='Same-cost optimized radius')
        ax.grid(alpha=.13)
    fig.supxlabel('Conditional PM2.5 radius limits (micrograms per cubic metre)',fontsize=10)
    fig.tight_layout();fig.savefig(figs/'repair_radius_frontier.pdf');fig.savefig(figs/'repair_radius_frontier.png',dpi=180);plt.close(fig)
    env = os.environ.copy(); env['TECTONIC_CACHE_DIR'] = str(ROOT / 'audit_tools/tex_cache')
    proc = subprocess.run([str(ROOT / 'audit_tools/tectonic/tectonic.exe'), '--only-cached', '--keep-logs',
                           '--outdir', str(SRC), str(SRC / 'manuscript.tex')], cwd=SRC, env=env,
                          capture_output=True, text=True, encoding='utf-8', errors='replace')
    (PKG / 'Review_Notes/concise_compile.txt').write_text(proc.stdout + '\n' + proc.stderr, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(proc.stderr[-5000:])
    shutil.copy2(SRC / 'manuscript.pdf', OUT)
    doc = fitz.open(OUT); text = '\n'.join(page.get_text() for page in doc)
    assert '??' not in text and '@@' not in text
    log = (SRC / 'manuscript.log').read_text(encoding='utf-8', errors='replace')
    assert 'undefined' not in log.lower(), 'Unresolved references or citations'
    assert 'Overfull' not in log, 'Text extends beyond its intended box'
    assert sha(OLD) == original_hash, 'Frozen complete manuscript changed'
    bib = (SRC / 'references.bib').read_text(encoding='utf-8')
    keys = set(re.findall(r'@\w+\{([^,]+),', bib))
    cites = {k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', source) for k in group.split(',')}
    assert cites <= keys, cites - keys
    assert source.count(r'\begin{proposition}') == 3
    assert source.count(r'\begin{theorem}') == 1
    assert len(doc)<=18, 'The final focused manuscript exceeds its chosen length budget'
    assert len(abstract.split()) <= 250
    assert OUT.stat().st_size < 20_000_000
    assert not [(i+1, b[:4]) for i, p in enumerate(doc) for b in p.get_text('blocks')
                if b[0] < 0 or b[1] < 0 or b[2] > p.rect.width+1 or b[3] > p.rect.height+1]
    report = {'status': 'compiled_and_checked', 'pages': len(doc), 'previous_pages': len(fitz.open(OLD)),
              'pdf_bytes': OUT.stat().st_size, 'abstract_words': len(abstract.split()),
              'external_cohorts': len(cohorts), 'external_station_memberships': 73, 'paired_station_days': 2114,
              'var_numerical_gains': var_wins, 'forest_numerical_gains': rf_wins, 'neural_seed_rows': len(neural),
              'original_neural_case_checks': original_neural_checks, 'total_neural_seed_rows': len(neural_rows),
              'primary_case_checks': primary_case_checks,
              'us_primary_case_checks': us_primary_checks,
              'decision_repair_summary': repair['summaries_by_query_fault_budget'],
              'decision_repair_archive_checks': repair['checks'], 'decision_repair_random_checks':repair_verification['checks'],
              'adaptive_repair_summary': adaptive['summaries'], 'adaptive_verification':adaptive_verification,
              'adaptive_exported_policy_replay': adaptive_replay,
              'radius_theorem_summary':radius_report,'radius_theorem_verification':radius_verification,
              'radius_boundary_replay':radius_replay,
              'primary_physical_pm25_case_gain': pm_gain, 'original_nuisance_span_checks': span_checks,
              'highlighted_propositions': 3, 'highlighted_theorems':1,'resolved_citation_keys': sorted(cites),
              'out_of_page_text_blocks': 0,
              'unresolved_references': 0, 'overfull_text_boxes': 0,
              'frozen_complete_manuscript_unchanged': original_hash, 'pdf_sha256': sha(OUT),
              'source_sha256': sha(SRC / 'manuscript.tex'),
              'data_sources': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'Research_Release/expanded_validation_results/COHORT_RESULTS.csv',
                  ROOT / 'Research_Release/weakness_resolution_results/NONLINEAR_SUMMARY.csv', ROOT / 'Research_Release/weakness_resolution_results/NEURAL_SUMMARY.csv',
                  ROOT / 'Research_Release/weakness_resolution_results/RISK_INTERVAL_AGGREGATES.csv',
                  ROOT / 'Research_Release/recent_data_results/delhi/outage_cases.csv',
                  ROOT / 'Research_Release/recent_data_results/delhi/grin_cases.csv',
                  ROOT / 'Research_Release/recent_data_results/delhi/grin_summary.json',
                  ROOT / 'Research_Release/recent_data_results/delhi/reconstruction_summary.json',
                  ROOT / 'Research_Release/recent_data_results/epa_la/reconstruction_summary.json',
                  ROOT / 'Research_Release/recent_data_results/epa_la/outage_cases.csv',
                  ROOT / 'Research_Release/recent_data_results/delhi/exact_nuisance_span_counts.csv',
                  ROOT / 'Research_Release/recent_data_results/epa_la/exact_nuisance_span_counts.csv',
                  ROOT / 'Research_Release/decision_repair_results/SUMMARY.json',
                  ROOT / 'Research_Release/decision_repair_results/VERIFICATION.json',
                  ROOT / 'Research_Release/decision_repair_results/SCENARIO_BANKS.json',
                  ROOT / 'Research_Release/decision_repair_results/REPAIR_CONFIGURATIONS.csv',
                  ROOT / 'Research_Release/decision_repair_results/ADAPTIVE_SUMMARY.json',
                  ROOT / 'Research_Release/decision_repair_results/ADAPTIVE_POLICIES.json',
                  ROOT / 'Research_Release/decision_repair_results/ADAPTIVE_CONFIGURATIONS.csv',
                  ROOT / 'Research_Release/decision_repair_results/RADIUS_SUMMARY.json',
                  ROOT / 'Research_Release/decision_repair_results/RADIUS_CONFIGURATIONS.csv',
                  ROOT / 'Research_Release/decision_repair_results/RADIUS_FRONTIERS.json',
                  SRC/'repair_radius_theorem.tex']},
              'scope': 'Saved-evidence consistency and document compilation; no new training runs or independent proof certification.'}
    AUDIT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    (PKG / 'Upload_Files/Project_Abstract_Concise.txt').write_text(abstract + '\n', encoding='utf-8')
    assets = [SRC / 'manuscript.tex', SRC / 'references.bib', SRC / 'README_EDITING.md'] + sorted(figs.glob('*.pdf'))
    archive = PKG / 'Supporting_Reference/SheafCode_Editable_Manuscript.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in assets:
            z.write(path, path.relative_to(SRC).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path in assets:
            assert z.read(path.relative_to(SRC).as_posix()) == path.read_bytes()
    report['editable_archive_sha256'] = sha(archive)
    report['editable_archive_verified_assets'] = len(assets)
    new_evidence = PKG / 'Supporting_Reference/SheafCode_Decision_Repair_Evidence.zip'
    experiment = ROOT / 'Research_Release/extensions/decision_repair'
    results = ROOT / 'Research_Release/decision_repair_results'
    package_files = {('decision_repair/'+p.name):p for p in experiment.glob('*.py')}
    package_files['decision_repair/PROTOCOL.md'] = experiment / 'PROTOCOL.md'
    package_files['decision_repair/ADAPTIVE_PROTOCOL.md'] = experiment / 'ADAPTIVE_PROTOCOL.md'
    package_files['decision_repair/RADIUS_PROTOCOL.md'] = experiment / 'RADIUS_PROTOCOL.md'
    package_files.update({('results/'+p.name):p for p in results.iterdir() if p.is_file()})
    package_hashes = {name:sha(path) for name,path in package_files.items()}
    with zipfile.ZipFile(new_evidence,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in package_files.items():
            z.write(path,name)
        z.writestr('SHA256.json',json.dumps(package_hashes,indent=2))
    with zipfile.ZipFile(new_evidence) as z:
        assert z.testzip() is None
        for name,digest in package_hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest
    report['decision_repair_evidence_zip_sha256'] = sha(new_evidence)
    report['decision_repair_evidence_assets'] = len(package_files)
    final_pdf=PKG/'Upload_Files/SheafCode_Final_Manuscript.pdf'
    shutil.copy2(OUT,final_pdf)
    assert sha(final_pdf)==sha(OUT)
    report['final_manuscript_file']=str(final_pdf.relative_to(ROOT))
    report['final_manuscript_sha256']=sha(final_pdf)
    AUDIT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
