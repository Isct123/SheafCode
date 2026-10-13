from common import *
from reconstruction import predict

def category(x):
    return int(np.searchsorted([30.0, 60.0, 90.0, 120.0, 250.0], x, side='left'))

def stream(ds, m, part):
    rows = []
    daily = []
    for day, ix in blocks(ds, part):
        for v, station in enumerate(ds['stations']):
            truth = ds['X'][ix[24:], v, 0]
            good = np.isfinite(truth)
            if good.sum() < 12:
                continue
            preds = predict(ds['Z'][ix], v, 24, m, ds, False)
            for method, pred in preds.items():
                raw = pred[:, 0] * ds['sd'][v, 0] + ds['mu'][v, 0]
                for j, t in enumerate(ix[24:]):
                    if good[j]:
                        rows.append({'part': part, 'day': day, 'timestamp': str(ds['times'][t]), 'station': str(station), 'method': method, 'prediction': raw[j], 'truth': truth[j], 'positive': truth[j] > 60})
                if good.sum() >= 18:
                    prior = ds['X'][ix[:24], v, 0]
                    previous = np.nanmean(prior) if np.isfinite(prior).sum() >= 18 else np.nan
                    daily.append({'day': day, 'station': str(station), 'method': method, 'truth_mean': np.mean(truth[good]), 'prediction_mean': np.mean(raw[good]), 'truth_category': category(np.mean(truth[good])), 'predicted_category': category(np.mean(raw[good])), 'previous_day_mean': previous, 'target_hours': int(good.sum()), 'aggregation': 'Common observed target hours; PM2.5-only category proxy, not official all-pollutant AQI'})
    return (pd.DataFrame(rows), pd.DataFrame(daily))

def main():
    for region in regions():
        ds = load(region)
        m = models(region)
        cal, _ = stream(ds, m, 3)
        test, daily = stream(ds, m, 2)
        choices = {}
        summary = []
        for method, g in cal.groupby('method'):
            cutoff = quantile(g.loc[~g.positive, 'prediction'], 0.05)
            choices[method] = {'cutoff': cutoff, 'validation_negatives': int((~g.positive).sum()), 'validation_positives': int(g.positive.sum())}
            q = test[test.method == method].copy()
            q['alarm'] = q.prediction > cutoff
            test.loc[q.index, 'alarm'] = q.alarm
            pos = q.positive
            summary.append({'method': method, 'cases': len(q), 'positive_cases': int(pos.sum()), 'negative_cases': int((~pos).sum()), 'sensitivity': q.loc[pos, 'alarm'].mean() if pos.any() else None, 'false_alarm_rate': q.loc[~pos, 'alarm'].mean() if (~pos).any() else None, 'precision': q.loc[q.alarm, 'positive'].mean() if q.alarm.any() else None})
        frame_save(region + '/hourly_alert_predictions.csv', test)
        frame_save(region + '/alert_summary.csv', pd.DataFrame(summary))
        save(region + '/alert_cutoffs.json', choices)
        save(region+'/alert_status.json', {'status':'completed','calibration_rows':len(cal),'test_rows':len(test)})
