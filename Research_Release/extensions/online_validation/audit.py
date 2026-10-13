import sys, ast, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

def main():
    checks = []

    def check(name, ok, detail=''):
        checks.append({'check': name, 'passed': bool(ok), 'detail': detail})
        if not ok:
            raise AssertionError(name)
    for region in ['delhi2025', 'bengaluru_raw']:
        m = json.loads((ROOT / 'results_expanded' / region / 'manifest.json').read_text())
        for path, h in m['code_sha256'].items():
            check('frozen_core/' + region + '/' + path, sha(ROOT / path) == h)
    for filename in ['replay_lock.json', 'beijing_lock.json', 'persistent_fault_lock.json', 'uncertainty_lock.json', 'raw_gain_lock.json']:
        lockfile = json.loads((OUT / filename).read_text())
        source = {'replay_lock.json': 'replay.py', 'beijing_lock.json': 'beijing.py', 'persistent_fault_lock.json': 'fault_episodes.py', 'uncertainty_lock.json': 'uncertainty_shift.py', 'raw_gain_lock.json': 'raw_gain_faults.py'}[filename]
        check('locked_source/' + source, sha(HERE / source) == lockfile['source_sha256'])
    replay = pd.read_csv(OUT / 'replay_cases.csv')
    summary = json.loads((OUT / 'replay_summary.json').read_text())
    check('all_headline_cases', len(replay) == 798 and (not replay.duplicated(['day', 'station', 'method']).any()))
    check('headline_individual_loss_replay', np.max(abs(replay.mse - replay.saved_mse)) < 1e-09)
    p = replay.pivot(index=['day', 'station'], columns='method', values='mse')
    gain = 100 * (1 - np.sqrt(p.temporal_sheaf.mean() / p.full_covariance_var.mean()))
    check('replayed_headline_gain', abs(gain - summary['comparison']['gain_percent']) < 1e-10)
    with zipfile.ZipFile(OUT / 'beijing_original.zip') as archive:
        check('original_public_dataset_CRC', archive.testzip() is None)
    beijing = pd.read_csv(OUT / 'beijing_corrected_cases.csv')
    old = pd.read_csv(OUT / 'beijing_cases.csv')
    bj = json.loads((OUT / 'beijing_corrected_summary.json').read_text())
    check('beijing_unique_shared_cases', len(beijing) == 2008 and (not beijing.duplicated(['day', 'station', 'method']).any()))
    check('baseline_correction_preserves_three_methods', np.allclose(beijing.loc[beijing.method != 'city_mean', 'mse'], old.loc[old.method != 'city_mean', 'mse'], rtol=0, atol=0))
    model = np.load(OUT / 'beijing_model.npz')
    check('beijing_fitted_model_record', model['delta'].shape[1] == 48 and model['mean'].shape == (12, 4))
    check('beijing_positive_prior', np.linalg.eigvalsh(model['P']).min() > 0)
    check('beijing_VAR_stability', max(abs(np.linalg.eigvals(model['var_A']))) <= 0.99500001)
    for name, comparison in bj['comparisons'].items():
        p = beijing.pivot(index=['day', 'station'], columns='method', values='mse')
        actual = 100 * (1 - np.sqrt(p.temporal_sheaf.mean() / p[name].mean()))
        check('beijing_gain/' + name, abs(actual - comparison['gain_percent']) < 1e-09)
    for prefix in ['persistent_fault', 'raw_gain_fault']:
        f = pd.read_csv(OUT / (prefix + '_cases.csv'))
        g = pd.read_csv(OUT / (prefix + '_summary.csv'))
        fields = ['region', 'fault_type', 'fault_count', 'method'] if prefix == 'persistent_fault' else ['region', 'gain_factor', 'fault_count', 'method']
        for _, row in g.iterrows():
            mask = np.ones(len(f), bool)
            for col in fields:
                mask &= f[col].to_numpy() == row[col]
            actual = f[mask]
            key = '/'.join((str(row[c]) for c in fields))
            check(prefix + '/cases/' + key, len(actual) == row.cases)
            check(prefix + '/exact/' + key, abs(actual.exact.mean() - row.exact_end_to_end) < 1e-12)
            check(prefix + '/alarm/' + key, abs(actual.alarm.mean() - row.alarm_rate) < 1e-12)
        for row in f.itertuples():
            truth = ast.literal_eval(row.true_support)
            prediction = ast.literal_eval(row.estimated_support)
            check(prefix + '/support_truth', bool(row.exact) == (truth == prediction))
            check(prefix + '/unknown_count_capacity', len(prediction) <= 2)
    intervals = pd.read_csv(OUT / 'uncertainty_shift_corrected_summary.csv')
    count = 0
    for path in sorted(OUT.glob('uncertainty_*_cases.csv')):
        f = pd.read_csv(path)
        count += len(f)
        check('interval_case_keys/' + path.name, not f.duplicated(['timestamp', 'station', 'method']).any())
        actual = (f.lower <= f.truth) & (f.truth <= f.upper)
        check('interval_coverage_flags/' + path.name, np.array_equal(actual, f.covered))
        check('interval_width/' + path.name, np.allclose(f.upper - f.lower, f.width, atol=1e-08, rtol=1e-09))
        for key, g in f.groupby(['region', 'extra_missing_rate', 'method']):
            q = intervals[(intervals.region == key[0]) & (intervals.extra_missing_rate == key[1]) & (intervals.method == key[2])]
            for _, row in q.iterrows():
                selected = g
                if row.group_type == 'quarter':
                    selected = g[g.calendar_quarter == row['group']]
                elif row.group_type == 'train_extreme':
                    selected = g[g.train_defined_extreme == (row['group'] == 'True')]
                label = '/'.join((str(v) for v in key)) + '/' + row.group_type + '/' + row['group']
                check('interval_summary_count/' + label, len(selected) == row.cases)
                check('interval_summary_coverage/' + label, abs(selected.covered.mean() - row.coverage) < 1e-12)
                if row.weeks < 5:
                    check('sparse_cluster_interval_suppressed/' + label, pd.isna(row.coverage_ci_low) and pd.isna(row.coverage_ci_high))
    check('all_shift_case_rows', count == 459216)
    exact = json.loads((OUT / 'exact_math_results.json').read_text())
    check('exact_rational_checks', exact['status'] == 'passed' and all((c['passed'] for c in exact['checks'])))
    official = json.loads((OUT / 'official_acquisition.json').read_text())
    for record in official:
        if record['status'] == 'downloaded':
            check('official_pdf/' + record['date'] + '/' + record['kind'], sha(OUT / record['path']) == record['sha256'])
    station = pd.read_csv(OUT / 'official_station_records.csv')
    proxy = pd.read_csv(OUT / 'official_proxy_comparison.csv')
    lookup = station.set_index(['date', 'station']).official_aqi.to_dict()
    for row in proxy.itertuples():
        check('official_alignment/' + row.date + '/' + row.station, lookup[row.date, row.station] == row.official_aqi)
    check('common_official_proxy_cases', proxy.groupby(['date', 'station']).method.nunique().eq(3).all())
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob('*')) if p.is_file()}
    derived = {'audit.json', 'report_metadata.json', 'DOCUMENT_AUDIT.json', 'STAGE_STATUS.json', 'RUN_LOCK.json', 'PACKAGE_AUDIT.json', 'ONLINE_VALIDATION_REPORT.pdf'}
    outputs = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix in ['.csv', '.json', '.pdf', '.zip', '.txt', '.npz'] and (p.name not in derived) and (not {'figures', 'preview'}.intersection(p.relative_to(OUT).parts))}
    result = {'status': 'passed', 'checks': checks, 'total_checks': len(checks), 'exact_rational_checks': exact['total_checks'], 'new_unit_tests': 7, 'sources': sources, 'outputs': outputs, 'scope': 'Saved-case independent aggregation, every fault support flag, interval coverage/width/groups, source locks, raw replay and downloaded source hashes. Finite exact tests are not formal unrestricted theorem proofs.'}
    save('audit.json', result)
    print('Online extension audit passed', len(checks), 'checks;', exact['total_checks'], 'exact checks', flush=True)
if __name__ == '__main__':
    main()
