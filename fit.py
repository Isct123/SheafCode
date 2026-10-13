import time
from common import *

def main():
    for region in regions():
        ds = load(region)
        Z = ds['Z']
        tr = Z[ds['split'] == 0]
        n, p = tr.shape[1:]
        edges = graph_correlation(tr)
        lock(region + '/fit_lock.json', {'source_sha256': sha(__file__), 'dataset_sha256': sha(OUT / region / 'dataset.npz'), 'choices': 'fixed grids, first validation half only'})
        if (OUT / region / 'models.pkl').exists():
            print(region, 'models already complete', flush=True)
            continue
        val = []
        for day, ix in subset_days(blocks(ds, 1), 12):
            for v in range(n):
                good = np.isfinite(Z[ix[24:], v])
                if good.sum() >= 12 * p:
                    val.append((day, ix, v, good))
        if not val:
            raise RuntimeError(region + ' no model selection cases')
        trials = []
        best = None
        for lam, ts in itertools.product([0.1, 1.0], [0.5, 2.0]):
            model = QuadraticDecoder(tr, edges, lam, ts)
            scores = []
            for day, ix, v, good in val:
                x = Z[ix].copy()
                x[24:, v] = np.nan
                pred = model.causal(x, v)
                scores.append(float(np.mean((pred - Z[ix[24:], v])[good] ** 2)))
            item = {'method': 'sheaf', 'lam': lam, 'temporal_scale': ts, 'validation_mse': float(np.mean(scores))}
            trials.append(item)
            if best is None or item['validation_mse'] < best[0]['validation_mse']:
                best = (item, model)
            print(region, 'sheaf choice', item, flush=True)
            frame_save(region + '/validation_grid.csv', pd.DataFrame(trials))
        training = complete_training(tr)
        vbest = None
        for ridge, noise, shrink in itertools.product([1.0, 100.0], [0.05, 0.5], [0.0, 0.2, 1.0]):
            model = FullVAR(training, n, p, ridge, noise, shrink)
            scores = []
            for day, ix, v, good in val:
                x = Z[ix].copy()
                x[24:, v] = np.nan
                pred = model.filter(x)[24:, v]
                scores.append(float(np.mean((pred - Z[ix[24:], v])[good] ** 2)))
            item = {'method': 'var', 'ridge': ridge, 'noise': noise, 'shrink': shrink, 'validation_mse': float(np.mean(scores))}
            trials.append(item)
            if vbest is None or item['validation_mse'] < vbest[0]['validation_mse']:
                vbest = (item, model)
            print(region, 'VAR choice', item, flush=True)
            frame_save(region + '/validation_grid.csv', pd.DataFrame(trials))
        covariance = np.cov(training, rowvar=False) + 0.001 * np.eye(n * p)
        gaussian = np.linalg.inv(covariance)
        sheaf = best[1]
        sheaf.precisions = {}
        fitted = {'sheaf': sheaf, 'var': vbest[1], 'gaussian': gaussian, 'edges': edges}
        temp = OUT / region / 'models.tmp'
        with temp.open('wb') as h:
            pickle.dump(fitted, h)
        temp.replace(OUT / region / 'models.pkl')
        save(region + '/model_choices.json', {'sheaf': best[0], 'var': vbest[0], 'validation_cases': len(val), 'edges': edges, 'model_sha256': sha(OUT / region / 'models.pkl'), 'scope': 'Both point models selected using first validation half; second half untouched for calibration'})
        print(region, 'model fitting complete', flush=True)
