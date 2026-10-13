import pickle, json, hashlib
from pathlib import Path
import common
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
HERE=Path(__file__).resolve().parent
SOURCE=common.OUT
OUT=common.OUT/"random_forest"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def features(ds, ix, v):
    x = ds['Z'][ix].copy()
    x[24:, v] = np.nan
    n, p = x.shape[1:]
    matrices = [x[24:].reshape(24, -1), x[23:47].reshape(24, -1), x[:24].reshape(24, -1)]
    values = [np.nan_to_num(a, nan=0.0) for a in matrices]
    masks = [np.isfinite(a).astype(float) for a in matrices]
    last = np.zeros(p)
    known = np.zeros(p)
    for c in range(p):
        q = np.flatnonzero(np.isfinite(x[:24, v, c]))
        if len(q):
            last[c] = x[q[-1], v, c]
            known[c] = 1
    station = np.zeros(n)
    station[v] = 1
    times = ds['times'][ix[24:]]
    clocks = np.column_stack([np.sin(times.hour * 2 * np.pi / 24), np.cos(times.hour * 2 * np.pi / 24), np.sin(times.dayofweek * 2 * np.pi / 7), np.cos(times.dayofweek * 2 * np.pi / 7), np.arange(1, 25) / 24])
    return np.column_stack(values + masks + [np.tile(last, (24, 1)), np.tile(known, (24, 1)), np.tile(station, (24, 1)), clocks])

def cases(ds, part, cap=None):
    bs = common.blocks(ds, part)
    if cap:
        bs = common.subset_days(bs, cap)
    ans = []
    for day, ix in bs:
        for v, station in enumerate(ds['stations']):
            truth = ds['Z'][ix[24:], v]
            if np.isfinite(truth).sum() >= 12 * truth.shape[1]:
                ans.append((day, str(station), v, ix, features(ds, ix, v), truth))
    return ans

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folders = sorted((p.parent for p in SOURCE.glob('*/outage_cases.csv')))
    records = []
    for folder in folders:
        name = folder.name
        dest = OUT / name
        dest.mkdir(exist_ok=True)
        identity = {'source_sha256': sha(Path(__file__)), 'dataset_sha256': sha(folder / 'dataset.npz'), 'reference_cases_sha256': sha(folder / 'outage_cases.csv')}
        if (dest / 'status.json').exists():
            old = json.loads((dest / 'status.json').read_text())
            if old['identity'] != identity:
                raise RuntimeError('Nonlinear input/code changed')
            records.append(old)
            continue
        dd = np.load(folder / 'dataset.npz', allow_pickle=False)
        ds = {k: dd[k] for k in dd.files} | {'times': pd.DatetimeIndex(dd['times'])}
        train = cases(ds, 0, 80)
        valid = cases(ds, 1, 12)
        test = cases(ds, 2)
        if not train or not valid or (not test):
            result = {'status': 'unavailable', 'cohort': name, 'identity': identity, 'reason': 'Insufficient eligible training/selection/test cases'}
            (dest / 'status.json').write_text(json.dumps(result, indent=2))
            records.append(result)
            continue
        _, _, v, ix, original, _ = test[0]
        modified = dict(ds)
        modified['Z'] = ds['Z'].copy()
        modified['Z'][ix[24:], v] = 123456.0
        modified['Z'][ix[-1] + 1:] = 789012.0
        if not np.array_equal(features(modified, ix, v), original):
            raise AssertionError('Target/future feature leakage')
        X = np.concatenate([c[4] for c in train])
        Y = np.concatenate([c[5] for c in train])
        choices = []
        for leaf in [5, 20]:
            models = []
            for j in range(Y.shape[1]):
                good = np.isfinite(Y[:, j])
                model = RandomForestRegressor(n_estimators=64, max_depth=12, min_samples_leaf=leaf, random_state=2039 + j, n_jobs=1)
                model.fit(X[good], Y[good, j])
                models.append(model)
            scores = []
            for _, _, _, _, f, truth in valid:
                pred = np.column_stack([model.predict(f) for model in models])
                good = np.isfinite(truth)
                scores.append(np.mean((pred - truth)[good] ** 2))
            choices.append((float(np.mean(scores)), leaf, models))
        loss, leaf, models = min(choices, key=lambda c: c[0])
        rows = []
        for day, station, v, ix, f, truth in test:
            pred = np.column_stack([model.predict(f) for model in models])
            good = np.isfinite(truth)
            pm = good[:, 0]
            rows.append({'day': day, 'station': station, 'method': 'random_forest_causal', 'mse': float(np.mean((pred - truth)[good] ** 2)), 'pm25_raw_mse': float(np.mean(((pred[:, 0] - truth[:, 0]) * ds['sd'][v, 0])[pm] ** 2)) if pm.any() else np.nan, 'target_cells': int(good.sum()), 'pm25_cells': int(pm.sum())})
        frame = pd.DataFrame(rows)
        frame.to_csv(dest / 'cases.csv', index=False)
        ref = pd.read_csv(folder / 'outage_cases.csv', dtype={'station': str})
        keys = lambda f: set(map(tuple, f[['day', 'station']].to_numpy()))
        if keys(frame) != keys(ref[ref.method == 'sheaf_causal']):
            raise AssertionError('Nonlinear case eligibility differs')
        pd.DataFrame([{'leaf': x[1], 'selection_mse': x[0]} for x in choices]).to_csv(dest / 'selection.csv', index=False)
        with (dest / 'models.pkl').open('wb') as stream:
            pickle.dump(models, stream)
        joined = pd.concat([frame, ref], ignore_index=True)
        summary = {str(days): {metric: common.paired(joined, 'sheaf_causal', 'random_forest_causal', value=metric, days=days) for metric in ['mse', 'pm25_raw_mse']} for days in [7, 14]}
        result = {'status': 'completed', 'cohort': name, 'identity': identity, 'training_station_days': len(train), 'selection_station_days': len(valid), 'test_station_days': len(test), 'leaf': leaf, 'selection_mse': loss, 'target_future_feature_invariance': True, 'summary': summary, 'model_sha256': sha(dest / 'models.pkl'), 'cases_sha256': sha(dest / 'cases.csv'), 'scope': 'Exploratory training/selection-fitted nonlinear causal control; same test cases and cells; no refitting from test outcomes'}
        (dest / 'status.json').write_text(json.dumps(result, indent=2))
        records.append(result)
        print('Nonlinear control', name, 'cases', len(test), 'sheaf gain', summary['7']['mse']['gain_percent'], flush=True)
    (OUT / 'status.json').write_text(json.dumps({'status': 'completed', 'cohorts': records}, indent=2))
