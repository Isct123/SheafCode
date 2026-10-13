                                                                                         
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Research_Release/replication_20261012_results'
OLD = ROOT / 'Research_Release/expanded_validation_results'
RAW = OUT / 'uk_sources'
sys.path.insert(0, str(ROOT / 'audit_tools/uk_packages'))
import rdata
warnings.filterwarnings('ignore', message='Missing constructor for R class')
spec = importlib.util.spec_from_file_location('uk_shared_ingestion', ROOT / 'Research_Release/extensions/expanded_validation/acquire.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
a.OUT = OUT
a.HERE = HERE

def download(code):
    filename = code+'_2026.RData'
    p = RAW / filename
    url = 'https://uk-air.defra.gov.uk/openair/R_data/'+filename
    if not p.exists():
        cached = OLD / 'raw' / filename
        if cached.exists():
            import shutil
            shutil.copy2(cached, p)
        else:
            tmp = p.with_suffix('.download')
            subprocess.run(['curl.exe', '--fail', '--location', '--silent', '--show-error', '--max-time', '90', '--retry', '1', '--output', str(tmp), url], check=True)
            tmp.replace(p)
    return p, {'file': str(p.relative_to(ROOT)), 'url': url, 'bytes': p.stat().st_size, 'sha256': a.sha(p)}

def main():
    RAW.mkdir(exist_ok=True)
    metadata = OLD / 'raw/aurn_metadata.csv'
    identity = {'uk_protocol_sha256': a.sha(HERE / 'UK_PROTOCOL.md'), 'adapter_sha256': a.sha(Path(__file__)),
        'metadata_sha256': a.sha(metadata), 'prior_uk_manifest_hashes': {str(p.relative_to(ROOT)):a.sha(p) for p in sorted(OLD.glob('uk_*/dataset_manifest.json'))},
        'original_uk_manifest_sha256': a.sha(ROOT / 'Research_Release/uk_validation_results/uk_aurn/dataset_manifest.json')}
    lock = OUT / 'uk_acquisition_lock.json'
    if lock.exists() and json.loads(lock.read_text()) != identity:
        raise RuntimeError('UK acquisition lock changed')
    a.save(lock.name, identity)
    meta = pd.read_csv(metadata)
    active = meta[(pd.to_datetime(meta.start_date) <= pd.Timestamp('2026-01-01')) &
        (meta.end_date.eq('ongoing') | meta.end_date.isna() | (pd.to_datetime(meta.end_date, errors='coerce') >= pd.Timestamp('2026-01-01')))]
    used = set(json.loads((ROOT / 'Research_Release/uk_validation_results/uk_aurn/dataset_manifest.json').read_text())['stations'])
    for p in OLD.glob('uk_*/dataset_manifest.json'):
        used.update(json.loads(p.read_text())['stations'])
    candidates = [s for s,g in active.groupby('site_id') if {'PM2.5','PM10'} <= set(g.parameter) and s not in used]
    station_metadata = active[active.site_id.isin(candidates)].drop_duplicates('site_id')
    times, split = a.timeline('2026-01-01', '2026-06-30')
    selected, ledger = [], []
    for zone, group in station_metadata.groupby('zone', sort=True):
        codes = sorted(group.site_id.astype(str))
        record = {'zone': zone, 'metadata_candidates': codes, 'site_decisions': []}
        if zone == 'Greater London' or len(codes) < 3:
            record['status'] = 'excluded_original_london_area' if zone=='Greater London' else 'fewer_than_three_metadata_sites'
            ledger.append(record)
            continue
        arrays, chosen, sources, quality = [], [], [], []
        incomplete = False
        for code in codes:
            try:
                p, info = download(code)
                f = rdata.read_rda(p)[code+'_2026'].copy()
                f.columns = [str(c).lower() for c in f.columns]
                dates = pd.to_datetime(f.date, unit='s', utc=True).dt.tz_localize(None) if pd.api.types.is_numeric_dtype(f.date) else pd.to_datetime(f.date, utc=True).dt.tz_localize(None)
                if dates.duplicated().any():
                    raise RuntimeError('Duplicate hourly timestamps')
                f.index = dates
                X = f.reindex(times).reindex(columns=['pm2.5','pm10']).to_numpy(dtype=float, copy=True)
                X[~np.isfinite(X)] = np.nan
                X[X < 0] = np.nan
            except Exception as exc:
                record['site_decisions'].append({'site': code, 'status': 'acquisition_failed', 'error': str(exc)})
                incomplete = True
                continue
            sources.append(info)
            eligible = a.qualified(X, split)
            record['site_decisions'].append({'site': code, 'status': 'training_eligible' if eligible else 'training_ineligible',
                'training_observed_hours': np.isfinite(X[split==0]).sum(0).tolist()})
            if eligible:
                arrays.append(X)
                chosen.append(code)
                quality.extend([{'station': code, 'parameter': row.parameter, 'ratified_to': str(row.ratified_to)}
                    for _,row in active[(active.site_id==code)&active.parameter.isin(['PM2.5','PM10'])].iterrows()])
            if len(chosen) == 4:
                break
        record['chosen_sites'] = chosen
        if incomplete or len(chosen) < 3:
            record['status'] = 'unresolved_source_order' if incomplete else 'fewer_than_three_training_eligible_sites'
        else:
            import re
            name = 'uk_zone_'+re.sub('[^a-z0-9]+', '_', zone.lower()).strip('_')
            name = a.store(name, np.stack(arrays, axis=1), times, split, chosen, ['PM2.5','PM10'], ['ug/m3']*2,
                sources+[{'file':str(metadata.relative_to(ROOT)), 'sha256':a.sha(metadata), 'url':'https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData'}],
                {'provider':'AURN', 'source_zone':zone, 'timezone':'UTC', 'timestamp_convention':'hour beginning',
                 'uk_protocol_sha256':identity['uk_protocol_sha256'], 'quality':quality,
                 'previous_station_ids_excluded':sorted(used), 'scope':'Station-disjoint new zone cohort; common reporting program and dates'})
            record['status'] = 'selected'
            selected.append(name)
            print('UK new zone selected', name, chosen, flush=True)
        ledger.append(record)
        a.save('uk_candidates.json', {'decisions':ledger})
        a.save('UK_acquired.json', {'cohorts':selected, 'status':'running'})
        previous = json.loads((OUT / 'cohorts.json').read_text())
        previous['eligible'] = list(dict.fromkeys(previous['eligible']+selected))
        a.save('cohorts.json', previous)
    a.save('UK_acquired.json', {'cohorts':selected, 'status':'completed', 'errors_retained':True})

if __name__ == '__main__':
    main()
