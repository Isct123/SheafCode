                                                                                       
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'SUBMISSION_WORKSPACE'
tracked = {}
checks = []

def read(name):
    path = ROOT / name
    data = path.read_bytes()
    tracked[name] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8-sig')

def rows(name):
    return list(csv.DictReader(read(name).splitlines()))

def load(name):
    return json.loads(read(name))

def require(value, label):
    if not value:
        raise AssertionError(label)
    checks.append(label)

def main():
    text = read('SUBMISSION_WORKSPACE/Source/manuscript.tex')
    compact = ' '.join(text.split())
    def present(phrase):
        require(' '.join(phrase.split()) in compact, 'manuscript: ' + phrase)
    cohorts = [r for r in rows('Research_Release/expanded_validation_results/COHORT_RESULTS.csv') if r['provider'] != 'OpenAQ']
    forests = {r['cohort']: r for r in rows('Research_Release/weakness_resolution_results/NONLINEAR_SUMMARY.csv')}
    require(len(cohorts) == 20, '20 evaluable external cohorts')
    for column, expected in [('cases', 2114), ('stations', 73), ('observed_pollutant_hour_cells', 549302)]:
        require(sum(int(r[column]) for r in cohorts) == expected, f'external total {column}={expected}')
    require(sum(float(r['gain']) > 0 for r in cohorts) == 13, '13/20 VAR numerical wins')
    require(sum(float(forests[r['cohort']]['gain']) > 0 for r in cohorts) == 18, '18/20 forest numerical wins')
    for r in cohorts:
        vals = [r['cases'], f"{float(r['gain']):.2f}", f"{float(forests[r['cohort']]['gain']):.2f}", f"{float(r['pm25_gain']):.2f}"]
        present(' & '.join(vals) + r'\\')
    networks = [('delhi', 'Research_Release/recent_data_results/delhi', 'India (A)'),
                ('epa_la', 'Research_Release/recent_data_results/epa_la', 'USA (B)'),
                ('uk_aurn', 'Research_Release/uk_validation_results/uk_aurn', 'UK (C)')]
    for _, folder, label in networks:
        summary = load(folder + '/reconstruction_summary.json')['primary_every_third_day']['var_causal']
        cases = rows(folder + '/outage_cases.csv')
        pair = {}
        for method in ['sheaf_causal', 'var_causal']:
            pair[method] = {(r['day'], r['station']): float(r['mse']) for r in cases if r['method'] == method and r['primary'] == 'True'}
        require(pair['sheaf_causal'].keys() == pair['var_causal'].keys(), label + ' paired primary cells')
        gain = 100 * (1 - math.sqrt(sum(pair['sheaf_causal'].values()) / sum(pair['var_causal'].values())))
        require(abs(gain - summary['gain_percent']) < 1e-9, label + ' primary gain recomputed from case MSE')
        interval = '--'.join(f'{x:.2f}' for x in summary['ci95']) if summary['ci95'] else 'Suppressed'
        present(f"{label} & {summary['cases']} & {summary['dates']} & {gain:.2f} & {interval}")
    neural = rows('Research_Release/weakness_resolution_results/NEURAL_SUMMARY.csv')
    for r in neural:
        ci = '--'.join(f'{x:.2f}' for x in ast.literal_eval(r['ci7']))
        present(f"{r['code']} & {r['seed']} & {r['cases']} & {float(r['gain']):.2f} & {ci}")
    originals = load('Research_Release/recent_data_results/delhi/grin_summary.json')['comparisons']
    original_cases = rows('Research_Release/recent_data_results/delhi/outage_cases.csv')
    grin = rows('Research_Release/recent_data_results/delhi/grin_cases.csv')
    truth = {(r['day'], r['station']): float(r['mse']) for r in original_cases if r['method'] == 'sheaf_retrospective'}
    def convert(g):
        return 100 * (1 - 1 / (1 - g / 100))
    for seed, r in originals.items():
        selected = [c for c in grin if c['seed'] == seed]
        gain = 100 * (1 - math.sqrt(sum(truth[(c['day'], c['station'])] for c in selected) / sum(float(c['mse']) for c in selected)))
        require(abs(gain - convert(r['gain_percent'])) < 1e-9, 'GRIN original seed ' + seed + ' numerator conversion')
        ci = '--'.join(f'{convert(x):.2f}' for x in reversed(r['ci95']))
        present(f"A & {seed} & {len(selected)} & {gain:.2f} & {ci}")
    risk = rows('Research_Release/weakness_resolution_results/RISK_INTERVAL_AGGREGATES.csv')
    for r in risk:
        if float(r['loss']) == 0 and r['method'] in ['station_fixed', 'risk_station']:
            group = 'All' if r['group'] == 'all' else 'Extreme'
            method = 'Fixed' if r['method'] == 'station_fixed' else 'Risk grouped'
            vals = [group, method, r['cells'], f"{100*float(r['macro_coverage']):.1f}", f"{100*float(r['pooled_coverage']):.1f}", f"{float(r['macro_finite_width']):.2f}", f"{100*float(r['full_fraction']):.3f}"]
            present(' & '.join(vals))
    for folder, opposite, unequal, n in [('delhi', 99.6, 0.7, 456), ('epa_la', 95.8, 15.0, 432)]:
        faults = rows(f'Research_Release/recent_data_results/{folder}/available_fault_summary.csv')
        for kind, expected in [('opposite', opposite), ('unequal', unequal)]:
            r = next(r for r in faults if r['kind'] == kind and r['true_count'] == '2')
            require(round(100 * float(r['exact_support']), 1) == expected and int(r['cases']) == n, folder + ' ' + kind + ' localization and denominator')
    alerts = rows('Research_Release/recent_data_results/delhi/alert_summary.csv')
    for method, sens, fp in [('sheaf_causal', 69.1, 9.6), ('var_causal', 52.7, 6.1)]:
        r = next(r for r in alerts if r['method'] == method)
        require(round(100*float(r['sensitivity']),1) == sens and round(100*float(r['false_alarm_rate']),1) == fp, method + ' matched alert tradeoff')
        require((int(r['positive_cases']), int(r['negative_cases'])) == (1177, 4012), 'alert positive/negative counts')
    mask = load('Research_Release/recent_data_results/delhi/matched_missing_mask_summary.json')
    require(mask['cases'] == 719 and mask['source_patterns'] == 155 and round(mask['gain_percent'], 2) == 3.94, 'natural-mask validation counts and gain')
    operational = load('Research_Release/operational_validation_results/SUMMARY.json')
    completed = [c for c in operational['cohorts'] if c['status'] == 'complete']
    require(len(completed) == 18, '18 operational PM-only roles')
    require(sum(c['complete_hours']['2'] for c in completed) == 13187, '13187 operational hour-role memberships')
    require(sum(c['test_box_members'] for c in completed) == 10292, '10292 operational box members')
    for f, trials, issued, abstain in [('0',39561,39309,252), ('1',207936,42699,165237)]:
        totals = {k: sum(c['metrics'][f]['direct'][k] for c in completed) for k in ['trials','issued','wrong','abstained']}
        require(totals == dict(trials=trials, issued=issued, wrong=0, abstained=abstain), 'direct certificate totals f=' + f)
        present(f'{int(f)} & {trials:,} & {issued:,} / 0')
    query = load('Research_Release/query_policy_results/SUMMARY.json')['cohorts']
    for f, planned, frequency in [(0,128738,129021), (1,751733,752332)]:
        rs = [r for r in query if r['faults'] == f]
        require(sum(r['validated'] for r in rs) == planned and sum(r['training'] for r in rs) == frequency, 'validation-selected query totals f=' + str(f))
    joint = load('Research_Release/joint_certification_results/SUMMARY.json')
    require((joint['cases'], joint['issued'], joint['wrong'], joint['positive_truth_cases']) == (272,155,0,40), 'exact joint case/issued/error/positive totals')
    require(joint['status_counts'] == dict(verified_bounds=260, certified_infeasible=12), 'joint feasible/inconsistent counts')
    present('All 40 positive truth cases abstained')
    math_report = load('Research_Release/math_integration_results/REPORT_FACTS.json')
    require((math_report['attempted_roles'], math_report['completed_roles'], math_report['verified_feasible_directional_searches'], math_report['certified_infeasible_directional_searches']) == (23,17,520,24), 'supporting math cases and exclusions')
    require(math_report['replay']['max_final_gap'] < 1e-7, 'scalar exact nominal gap below 1e-7')
    present(r'0\leq\rho<\min\{G,r_B\}')
    present(r'\max_{y\,\text{compatible}}')
    require(text.count(r'\begin{proposition}') == 5 and text.count(r'\begin{theorem}') == 1 and text.count(r'\textbf{Proof.}') == 7, 'all five propositions, theorem and adaptive extension have proofs')
    abstract_tex = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',text,re.S).group(1).replace(r'\%', '%').replace('--','\u2013')
    abstract = read('SUBMISSION_WORKSPACE/Abstract.txt').strip()
    require(' '.join(abstract_tex.split()) == ' '.join(abstract.split()), 'manuscript and abstract match')
    require(len(abstract.split()) == 237 and len(abstract.split()) <= 300, 'abstract words and current portal cap')
    require(' '.join(abstract.split()) in ' '.join(read('SUBMISSION_WORKSPACE/Portal_Text.txt').split()), 'portal repeats matching abstract')
    for name in ['primary_reconstruction','external_comparisons','joint_example','followup_risk','repair_radius_frontier','math_integration']:
        original = ROOT / ('Research_Release/math_integration_results/math_integration.pdf' if name == 'math_integration' else f'IRIS_Submission_Recent_Data/Editable_Manuscript/figures/{name}.pdf')
        current = WORK / 'Source/figures' / (name + '.pdf')
        require(current.read_bytes() == original.read_bytes(), 'figure matches checked evidence asset: ' + name)
        tracked[str(current.relative_to(ROOT))] = hashlib.sha256(current.read_bytes()).hexdigest()
        tracked[str(original.relative_to(ROOT))] = hashlib.sha256(original.read_bytes()).hexdigest()
    diagrams = load('SUBMISSION_WORKSPACE/Guides/DIAGRAM_CHECK.json')
    require(diagrams['network']['stations'] == ['17','50','8235','8239','10485'] and len(diagrams['network']['edges']) == 9 and diagrams['network']['operator_shape'] == [36,20], 'data-derived diagram network dimensions and station IDs')
    require(diagrams['worked_example'] == dict(reading_error=1, separation_radius=4, semantic_margin=5, boundary_shared_reading=5), 'diagram matches the proved stability example')
    for name, expected in diagrams['figures'].items():
        path = WORK/'Source/figures'/(name+'.pdf')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, 'diagram matches generator audit: ' + name)
        tracked[str(path.relative_to(ROOT))] = actual
    report = dict(status='passed', reviewed_date='2026-10-08', checks=len(checks), items=checks, evidence_sha256=tracked,
                  scope='Current manuscript displayed numerical rows, primary/GRIN case-loss recomputation, selected supplementary counts, mathematical text structure and companion alignment. Written proof review and independent full evidence replays are recorded separately; no refitting or physical validation.')
    (WORK/'Guides/FINAL_MANUSCRIPT_FACT_CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['items','evidence_sha256']},indent=2))

if __name__ == '__main__':
    main()
