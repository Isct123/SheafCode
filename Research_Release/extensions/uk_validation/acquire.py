from pathlib import Path
import sys, subprocess, json, hashlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Research_Release/uk_validation_results'
RAW = OUT / 'raw'
sys.path.insert(0, str(ROOT / 'audit_tools/uk_packages'))
import rdata

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x):
    p = OUT / p; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2), encoding='utf-8')

def fetch(name):
    p = RAW / name
    url = 'https://uk-air.defra.gov.uk/openair/R_data/' + name
    if not p.exists():
        temp = p.with_suffix('.download')
        subprocess.run(['curl.exe', '--fail', '--location', '--silent', '--show-error', '--max-time', '90', '--retry', '2', '--output', str(temp), url], check=True)
        rdata.read_rda(temp)
        temp.replace(p)
    return p, {'url': url, 'sha256': sha(p), 'bytes': p.stat().st_size}

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    identity = {'protocol_sha256': sha(HERE / 'PROTOCOL.md'), 'source_sha256': sha(Path(__file__))}
    lock = OUT / 'acquisition_lock.json'
    if lock.exists():
        previous = json.loads(lock.read_text())
        if previous['protocol_sha256'] != identity['protocol_sha256']: raise RuntimeError('UK protocol changed')
        if previous['source_sha256'] != identity['source_sha256']:
            if (OUT / 'uk_aurn/models.pkl').exists(): raise RuntimeError('UK acquisition changed after fitting')
            hp = OUT / 'acquisition_parser_history.json'
            history = json.loads(hp.read_text()) if hp.exists() else []
            if isinstance(history, dict): history = [history]
            history.append({'previous_lock': previous, 'correction': 'Pre-fit parser compatibility: ongoing end dates, epoch-second POSIXct and writable pandas arrays. No outcomes evaluated.'})
            save('acquisition_parser_history.json', history)
    save('acquisition_lock.json', identity)
    path, source = fetch('AURN_metadata.RData')
    meta = rdata.read_rda(path)['AURN_metadata']
    meta.to_csv(RAW / 'metadata.csv', index=False)
    active = meta[meta.latitude.between(51.2, 51.8) & meta.longitude.between(-.6, .35)
                  & (pd.to_datetime(meta.start_date) <= pd.Timestamp('2026-01-01'))
                  & (meta.end_date.isna() | (meta.end_date == 'ongoing')
                     | (pd.to_datetime(meta.end_date, errors='coerce') >= pd.Timestamp('2026-01-01')))]
    params = ['PM2.5', 'PM10', 'O3']
    candidates = [code for code, g in active.groupby('site_id') if set(params) <= set(g.parameter)]
    save('candidate_sites.json', {'codes': sorted(candidates), 'selection': 'metadata geography/active channels, before fitting'})
    times = pd.date_range('2026-01-01', '2026-06-30 23:00', freq='h')
    split = np.zeros(len(times), int)
    split[int(.6*len(times)):int(.7*len(times))] = 1
    split[int(.7*len(times)):int(.8*len(times))] = 3
    split[int(.8*len(times)):] = 2
    chosen = []; arrays = []; sources = [source]; eligibility = []; observations = []; quality = []
    for code in sorted(candidates):
        path, source = fetch(code + '_2026.RData'); sources.append(source)
        objects = rdata.read_rda(path)
        f = objects[code + '_2026'].copy(); f.columns = [str(c).lower() for c in f.columns]
        f['date'] = pd.to_datetime(f['date'], unit='s', utc=True).dt.tz_localize(None) if pd.api.types.is_numeric_dtype(f['date']) else pd.to_datetime(f['date'], utc=True).dt.tz_localize(None)
        if not (f.date.dt.year == 2026).any(): raise RuntimeError('UK timestamp parsing failed: ' + code)
        if f.date.duplicated().any(): raise RuntimeError('Duplicate source hours: ' + code)
        f = f.set_index('date').reindex(times)
        columns = ['pm2.5', 'pm10', 'o3']
        x = f.reindex(columns=columns).apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float, copy=True)
        x[~np.isfinite(x)] = np.nan
        x[x < 0] = np.nan
        training = x[split == 0]
        qualified = bool(np.all(np.isfinite(training).mean(0) >= .5) and np.all(np.isfinite(training).sum(0) >= 720))
        eligibility.append({'code': str(code), 'training_hours': np.isfinite(training).sum(0).tolist(), 'qualified': qualified})
        for j, parameter in enumerate(params):
            mm = active[(active.site_id == code) & (active.parameter == parameter)].iloc[0]
            cutoff = pd.to_datetime(mm.ratified_to, errors='coerce')
            quality.append({'station': str(code), 'channel': parameter, 'ratified_to': str(mm.ratified_to),
                            'period_fully_ratified': bool(pd.notna(cutoff) and cutoff >= pd.Timestamp('2026-06-30'))})
            for t in np.flatnonzero(np.isfinite(x[:, j])):
                observations.append({'timestamp': str(times[t]), 'station': str(code), 'pollutant': parameter,
                                     'value': float(x[t, j]), 'unit': 'ug/m3',
                                     'ratified': bool(pd.notna(cutoff) and times[t].floor('D') <= cutoff)})
        if qualified and len(chosen) < 5:
            chosen.append(str(code)); arrays.append(x)
        print('UK source parsed', code, 'training eligible', qualified, flush=True)
    if len(chosen) < 3: raise RuntimeError('Fewer than three eligible UK stations')
    pd.DataFrame(observations).to_csv(OUT / 'source_observations.csv', index=False)
    pd.DataFrame(quality).to_csv(OUT / 'ratification.csv', index=False)
    X = np.stack(arrays, axis=1); mu = np.nanmean(X[split == 0], axis=0); sd = np.nanstd(X[split == 0], axis=0, ddof=1)
    if (sd < 1e-8).any(): raise RuntimeError('Degenerate UK channel')
    folder = OUT / 'uk_aurn'; folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / 'dataset.npz', X=X, Z=(X-mu)/sd, mu=mu, sd=sd, split=split,
                        times=times.to_numpy(), stations=np.array(chosen), channels=np.array(params))
    save('uk_aurn/dataset_manifest.json', {'region': 'uk_aurn', 'stations': chosen, 'channels': params,
         'units': ['ug/m3']*3, 'dates': ['2026-01-01', '2026-06-30'], 'timezone': 'UTC',
         'timestamp_convention': 'hour beginning', 'split_counts': {str(i): int((split==i).sum()) for i in [0,1,3,2]},
         'training_only_selection': True, 'eligibility': eligibility, 'sources': sources,
         'protocol_sha256': identity['protocol_sha256'], 'source_sha256': identity['source_sha256']})
    save('cohorts.json', {'eligible': ['uk_aurn'], 'secondary': []})
    save('acquisition_status.json', {'status': 'completed', 'stations': chosen, 'raw_sources': len(sources)})

if __name__ == '__main__': main()
