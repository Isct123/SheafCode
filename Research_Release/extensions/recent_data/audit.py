import sys, csv, ast
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

def main():
    checks = []

    def check(name, ok):
        checks.append({'check': name, 'passed': bool(ok)})
        if not ok:
            raise AssertionError(name)
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
        check('syntax/' + path.name, True)
    acquired = json.loads((DATA / 'acquisition_records.json').read_text())
    valid = [r for r in acquired if r['status'] in ['cached', 'downloaded']]
    for r in valid:
        check('raw_hash/' + str(r['location_id']) + '/' + r['date'], sha(DATA / 'raw' / f"{r['location_id']}_{pd.Timestamp(r['date']):%Y%m%d}.csv.gz") == r['sha256'])
    obs = pd.read_csv(DATA / 'observations.csv', parse_dates=['timestamp'])
    check('all_recent_observation_dates', obs.timestamp.min() >= pd.Timestamp('2026-01-01') and obs.timestamp.max() <= pd.Timestamp('2026-10-05 23:59:59'))
    check('no_negative_observations', obs.value.ge(0).all())
    check('unique_hour_channel', not obs.duplicated(['region', 'station', 'timestamp', 'pollutant']).any())
    for region in regions() + ['epa_la']:
        ds = load(region)
        m = models(region)
        ch = json.loads((OUT / region / 'model_choices.json').read_text())
        check(region + '/model_hash', sha(OUT / region / 'models.pkl') == ch['model_sha256'])
        check(region + '/normalization_training_only', np.allclose(ds['mu'], np.nanmean(ds['X'][ds['split'] == 0], 0)))
        manifest = json.loads((OUT / region / 'dataset_manifest.json').read_text())
        check(region + '/all_four_splits_disjoint', set(np.unique(ds['split'])) == {0, 1, 2, 3})
        check(region + '/all_calendar_dates_recent', ds['times'].min() >= pd.Timestamp('2026-01-01') and ds['times'].max() <= pd.Timestamp('2026-10-05 23:00'))
        check(region + '/chronological_four_blocks', all((ds['times'][ds['split'] == a].max() < ds['times'][ds['split'] == b].min() for a, b in [(0, 1), (1, 3), (3, 2)])))
        check(region + '/training_standard_deviations', np.allclose(ds['sd'], np.nanstd(ds['X'][ds['split'] == 0], 0, ddof=1)))
        check(region + '/normalized_tensor', np.allclose(ds['Z'], (ds['X'] - ds['mu']) / ds['sd'], equal_nan=True))
        fitlock = json.loads((OUT / region / 'fit_lock.json').read_text())
        for k, p in [('source_sha256', HERE / 'fit.py'), ('common_sha256', HERE / 'common.py'), ('reference_sha256', ROOT / 'extensions/online_validation/reference.py'), ('dataset_sha256', OUT / region / 'dataset.npz')]:
            check(region + '/frozen_fit/' + k, sha(p) == fitlock[k])
        if region == 'delhi':
            check(region + '/data_manifest_hash', manifest['observations_sha256'] == sha(DATA / 'observations.csv'))
            for v, station in enumerate(ds['stations']):
                for j, pollutant in enumerate(ds['channels']):
                    q = obs[(obs.station.astype(str) == str(station)) & (obs.pollutant == pollutant)].set_index('timestamp').value.reindex(ds['times']).to_numpy()
                    check(region + '/source_tensor/' + str(station) + '/' + pollutant, np.allclose(q, ds['X'][:, v, j], equal_nan=True))
        else:
            for source in manifest['source_hashes']:
                check(region + '/epa_raw_zip/' + source['url'].split('/')[-1], sha(OUT / region / 'raw' / source['url'].split('/')[-1]) == source['sha256'])
            source = pd.read_csv(OUT / region / 'source_observations.csv', parse_dates=['timestamp'])
            codes = [88101, 44201, 42602]
            units = ['Micrograms/cubic meter (LC)', 'Parts per million', 'Parts per billion']
            for v, station in enumerate(ds['stations']):
                pocs = manifest['selected_monitor_pocs'][v]['pocs']
                for j, code in enumerate(codes):
                    q = source[(source['Site Num'] == int(station)) & (source.parameter == code) & (source.POC == pocs[j]) & (source['Units of Measure'] == units[j]) & (source['Sample Measurement'] >= 0)].groupby('timestamp')['Sample Measurement'].mean().reindex(ds['times']).to_numpy()
                    check(region + '/source_tensor/' + str(station) + '/' + str(code), np.allclose(q, ds['X'][:, v, j], equal_nan=True))
        f = pd.read_csv(OUT / region / 'outage_cases.csv', dtype={'station': str})
        check(region + '/unique_outage_rows', not f.duplicated(['day', 'station', 'method']).any())
        check(region + '/all_outage_losses_finite', np.isfinite(f.mse).all() and f.mse.ge(0).all())
        summ = json.loads((OUT / region / 'reconstruction_summary.json').read_text())
        calc = paired(f[f.primary], 'sheaf_causal', 'var_causal')
        check(region + '/headline_reaggregates', abs(calc['gain_percent'] - summ['primary_every_third_day']['var_causal']['gain_percent']) < 1e-12)
        by = f.groupby(['day', 'station']).method.nunique()
        check(region + '/outage_common_method_sets', by.eq(7).all())
        check(region + '/matching_target_cells', f.groupby(['day', 'station']).target_cells.nunique().eq(1).all())
        for key, ff in [('primary_every_third_day', f[f.primary]), ('all_eligible_test_days', f)]:
            for method in ['var_causal', 'seasonal', 'city_mean', 'gaussian', 'spatial_sheaf']:
                z = paired(ff, 'sheaf_causal', method)
                saved = summ[key][method]
                check(region + '/' + key + '/' + method, abs(z['gain_percent'] - saved['gain_percent']) < 1e-10 and z['cases'] == saved['cases'] and (z['dates'] == saved['dates']))
                check(region + '/' + key + '/' + method + '/short_bin_intervals', z['ci95'] is None and saved['ci95'] is None or (z['ci95'] is not None and saved['ci95'] is not None and np.allclose(z['ci95'], saved['ci95'], rtol=0, atol=1e-10)))
        for status, source in [('uncertainty_lock.json', 'uncertainty.py'), ('evaluation_lock.json', 'evaluate.py')]:
            z = json.loads((OUT / region / status).read_text())
            check(region + '/stage_source/' + source, z['source_sha256'] == sha(HERE / source))
        u = pd.read_csv(OUT / region / 'uncertainty_cases.csv')
        check(region + '/coverage_recomputed', np.array_equal(u.covered.to_numpy(), ((u.truth >= u.lower) & (u.truth <= u.upper)).to_numpy()))
        check(region + '/interval_widths_recomputed', np.allclose(u.width, u.upper - u.lower))
        check(region + '/truth_dates_test_only', pd.to_datetime(u.timestamp).min() >= ds['times'][ds['split'] == 2].min())
        for fn in ['available_fault_episode_cases.csv', 'available_fault_hour_cases.csv']:
            ff = pd.read_csv(OUT / region / fn).fillna({'true_support': '', 'predicted_support': ''})
            for i, r in ff.iterrows():
                a = set(str(r.true_support).split(',')) - {''}
                b = set(str(r.predicted_support).split(',')) - {''}
                check(region + '/' + fn + f'/support_{i}', bool(r.exact) == (a == b))
                check(region + '/' + fn + f'/count_limit_{i}', len(b) <= 2)
        ranks = pd.read_csv(OUT / region / 'structured_rank_cases.csv')
        check(region + '/structured_rank_all_subsets', ranks.agree.all())
        margins = pd.read_csv(OUT / region / 'exhaustive_block_margins.csv')
        from math import comb
        check(region + '/four_union_margins_exhaustive', len(margins) == sum((comb(len(ds['stations']), k) for k in range(1, min(4, len(ds['stations'])) + 1))))
        b = pd.read_csv(OUT / region / 'functional_joint_bounded_cases.csv')
        check(region + '/conditional_bounds_contain_truth', b.loc[b.bounded_assumptions_met, 'bounded_covered'].all())
        thinned = pd.read_csv(OUT / region / 'thinning_cases.csv')
        check(region + '/thinning_agreement_flags', np.array_equal(thinned.agree, thinned.generic_pattern_rank == thinned.learned_rank))
        synthetic = pd.read_csv(OUT / region / 'synthetic_certificate_cases.csv')
        check(region + '/synthetic_exact_envelope', (synthetic.exact_lower <= synthetic.truth + 1e-08).all() and (synthetic.exact_upper >= synthetic.truth - 1e-08).all())
        check(region + '/synthetic_dual_outer', (synthetic.dual_lower <= synthetic.exact_lower + 1e-08).all() and (synthetic.dual_upper >= synthetic.exact_upper - 1e-08).all())
        perturb = pd.read_csv(OUT / region / 'operator_uncertainty_cases.csv')
        check(region + '/declared_operator_perturbation_containment', perturb.robust_covered.all() and len(perturb) == 48)
        refine = pd.read_csv(OUT / region / 'verified_refinement_cases.csv')
        check(region + '/refinement_outer_bounds', (refine.lower <= refine.exact_lower + 1e-08).all() and (refine.upper >= refine.exact_upper - 1e-08).all())
        check(region + '/rational_correction_outer', (refine.verified_upper >= refine.exact_upper - 1e-08).all())
        blocks = pd.read_csv(OUT / region / 'block_interval_cases.csv', dtype={'station': str})
        days = pd.read_csv(OUT / region / 'block_interval_day_cases.csv', dtype={'station': str})
        calc = blocks.groupby(['context_loss', 'method', 'day', 'station']).covered.all().reset_index()
        joined = calc.merge(days, on=['context_loss', 'method', 'day', 'station'], validate='one_to_one')
        check(region + '/simultaneous_days_recomputed', len(joined) == len(days) == len(calc) and np.array_equal(joined.covered.to_numpy(), joined.all_covered.to_numpy()))
        check(region + '/week_full_sets_are_infinite', np.isposinf(blocks.loc[blocks.method == 'week_max', 'width']).all())
        missing = pd.read_csv(OUT / region / 'matched_missing_mask_cases.csv')
        check(region + '/matched_masks_have_truth', np.isfinite(missing.truth).all())
        check(region + '/matched_mask_days_differ', (missing.source_mask_day != missing.evaluation_day).all())
        hybrid = pd.read_csv(OUT / region / 'hybrid_cases.csv', dtype={'station': str})
        choice = json.loads((OUT / region / 'hybrid_choice.json').read_text())
        grid = pd.read_csv(OUT / region / 'hybrid_validation_grid.csv')
        check(region + '/hybrid_validation_selection', np.isclose(choice['validation_mse'], grid.validation_mse.min()))
        meta = json.loads((OUT / region / 'additional_status.json').read_text())
        check(region + '/hybrid_row_count', meta['hybrid_rows'] == len(hybrid))
        alerts = pd.read_csv(OUT / region / 'hourly_alert_predictions.csv')
        asum = pd.read_csv(OUT / region / 'alert_summary.csv')
        for row in asum.itertuples():
            q = alerts[alerts.method == row.method]
            check(region + '/alert_counts/' + row.method, len(q) == row.cases and q.positive.sum() == row.positive_cases)
            check(region + '/alert_false_alarm/' + row.method, np.isclose(q.loc[~q.positive, 'alarm'].mean(), row.false_alarm_rate))
            check(region + '/alert_sensitivity/' + row.method, np.isclose(q.loc[q.positive, 'alarm'].mean(), row.sensitivity, equal_nan=True))
    neural = json.loads((OUT / 'delhi/neural_lock.json').read_text())
    check('neural/frozen_source', neural['source_sha256'] == sha(HERE / 'neural.py'))
    check('neural/frozen_data', neural['dataset_sha256'] == sha(OUT / 'delhi/dataset.npz'))
    grin = pd.read_csv(OUT / 'delhi/grin_cases.csv', dtype={'station': str})
    choices = json.loads((OUT / 'delhi/grin_choices.json').read_text())
    check('neural/all_seeds', set(grin.seed) == {0, 1, 2} and {z['seed'] for z in choices} == {0, 1, 2})
    ref = pd.read_csv(OUT / 'delhi/outage_cases.csv', dtype={'station': str})
    saved = json.loads((OUT / 'delhi/grin_summary.json').read_text())
    for choice in choices:
        options = []
        for width in [32, 64]:
            training = pd.read_csv(OUT / 'delhi' / f"grin_seed{choice['seed']}_width{width}_training.csv")
            best = training.loc[training.validation_mse.idxmin()]
            options.append((best.validation_mse, width, int(best.epoch)))
        best = min(options)
        check('neural/validation_selection/' + str(choice['seed']), np.isclose(choice['validation_mse'], best[0]) and choice['width'] == best[1] and (choice['epoch'] == best[2]))
        z = paired(pd.concat([grin[grin.seed == choice['seed']], ref[ref.method == 'sheaf_retrospective']]), 'grin_retrospective', 'sheaf_retrospective')
        check('neural/paired_summary/' + str(choice['seed']), np.isclose(z['gain_percent'], saved['comparisons'][str(choice['seed'])]['gain_percent']) and z['cases'] == 242)
    vendor = ROOT / 'extensions/validation/vendor/grin'
    vsource = json.loads((vendor / 'SOURCE.json').read_text())
    check('neural/vendor_commit', neural['vendor_commit'] == vsource['commit'])
    for name, digest in vsource['files'].items():
        check('neural/vendor_hash/' + name, sha(vendor / name) == digest)
    official = json.loads((OUT / 'delhi/official_acquisition.json').read_text())
    for row in official:
        check('official/date_verified/' + row['date'], row['date_verified'])
        check('official/pdf_hash/' + row['date'], sha(OUT / 'delhi/official' / f"{pd.Timestamp(row['date']):%Y%m%d}_detail.pdf") == row['sha256'])
    exact = json.loads((OUT / 'exact_math_results.json').read_text())
    check('exact_rational/all_checks', len(exact['checks']) == 37678 and all((z['passed'] for z in exact['checks'])))
    final = json.loads((OUT / 'FINAL_CONFIRMATION_REPLAY.json').read_text())
    check('independent_replay/all_cases_and_methods', final['status'] == 'passed' and final['all_eligible_station_days'] == 310 and (final['methods_replayed'] == 7) and all((z['passed'] for z in final['checks'])))
    check('independent_replay/source_hash', final['source_sha256'] == sha(HERE / 'final_confirmation.py'))
    neuralreplay = json.loads((OUT / 'NEURAL_CONFIRMATION_REPLAY.json').read_text())
    check('neural_checkpoint_replay/all_seeds', neuralreplay['status'] == 'passed' and {z['seed'] for z in neuralreplay['checks']} == {0, 1, 2} and all((z['passed'] and z['cases'] == 242 for z in neuralreplay['checks'])))
    check('neural_checkpoint_replay/source_hash', neuralreplay['source_sha256'] == sha(HERE / 'neural_confirmation.py'))
    overlap = json.loads((OUT / 'PRIOR_RECENT_OVERLAP.json').read_text())
    ov = pd.read_csv(OUT / 'PRIOR_RECENT_OBSERVATION_OVERLAP.csv')
    check('prior_overlap/matched_records', len(ov) == overlap['shared_hour_channel_rows'] and ov.value_agrees.sum() == overlap['equal_aggregated_values'])
    status = {'status': 'passed', 'checks': checks, 'check_count': len(checks), 'networks': regions() + ['epa_la'], 'exact_finite_rational_checks': 37678, 'scope': 'Recent-data source hashes and tensors, chronology, frozen implementations, common-case aggregates, every neural seed/validation choice, fault support flags, interval/alert aggregates, structured margins and conditional containment. Internal audit, not organiser eligibility or unrestricted proof certification.'}
    save('FINAL_EVIDENCE_AUDIT.json', status)
    print('Recent evidence audit passed', len(checks), 'checks', flush=True)
if __name__ == '__main__':
    main()
