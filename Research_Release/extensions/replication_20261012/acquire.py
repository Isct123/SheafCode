                                                                                        
from pathlib import Path
import importlib.util
import json
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Research_Release/replication_20261012_results'
OLD = ROOT / 'Research_Release/expanded_validation_results'
LEGACY = ROOT / 'Research_Release/extensions/expanded_validation'
sys.path.insert(0, str(LEGACY))
spec = importlib.util.spec_from_file_location('prior_acquisition', LEGACY / 'acquire.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
a.OUT = OUT
a.RAW = OLD / 'raw'
a.HERE = HERE

def preserved_download(name, url, cache=None):
    p = a.RAW / name
    if not p.exists():
        raise FileNotFoundError('Preserved source missing: '+str(p))
    return p, {'file': str(p.relative_to(ROOT)), 'url': url, 'sha256': a.sha(p),
               'bytes': p.stat().st_size, 'acquisition': 'Reused preserved original archive; no fresh download'}
a.download = preserved_download

def epa():
    seen = {tuple(map(int, p.name.split('_')[1:])) for p in OLD.glob('epa_*') if p.is_dir()}
    seen.add((6, 37))
    source = (LEGACY / 'acquire.py').read_text(encoding='utf-8')
    start = source.index('def epa():')
    stop = source.index('\ndef uk():', start)
    source = source[start:stop]
    source = source.replace('if (int(state),int(county))==(6,37):continue', 'if (int(state),int(county)) in PREVIOUS_COUNTIES:continue')
    source = source.replace('if len(selected)>=12:continue', "if len(selected)>=24:\n            excluded.append({'source':'EPA','state':int(state),'county':int(county),'eligible_stations':len(chosen),'reason':'Deferred beyond fixed 24-new-cohort cap; not evaluated'});continue")
    source = source.replace('First twelve qualifying state/county groups; no test-score selection', 'First 24 new qualifying counties, excluding prior cohorts; no test-score selection')
    scope = dict(a.__dict__)
    scope['PREVIOUS_COUNTIES'] = seen
    exec(compile(source, str(LEGACY / 'acquire.py'), 'exec'), scope)
    return scope['epa']()

def nsw():
                                                                                      
    spec = importlib.util.spec_from_file_location('prior_nsw', LEGACY / 'acquire_nsw.py')
    n = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n)
    n.a = a
    sites_path = a.RAW / 'nsw_sites.json'
    sites = json.loads(sites_path.read_text(encoding='utf-8'))
    groups = {}
    for s in sites:
        if s['Latitude'] is not None and s['Longitude'] is not None:
            groups.setdefault(s['Region'].strip(), []).append(int(s['Site_Id']))
    previous = set(json.loads((OLD / 'NSW_acquired.json').read_text())['cohorts'])
    times, split = a.timeline('2026-01-01', '2026-06-30')
    chosen_cohorts, decisions = [], []
    for region, ids in sorted(groups.items()):
        candidates = sorted(set(ids))
        if len(candidates) < 3:
            decisions.append({'region': region, 'reason': 'fewer_than_three_metadata_sites'})
            continue
        import re
        label = re.sub('[^a-z0-9]+', '_', region.lower()).strip('_')
        if 'nsw_'+label in previous:
            decisions.append({'region': region, 'reason': 'previously_evaluated_cohort'})
            continue
        req = a.RAW / ('nsw_'+label+'_request.json')
        response = a.RAW / ('nsw_'+label+'_response.json')
        if not req.exists() or not response.exists():
            decisions.append({'region': region, 'reason': 'preserved_response_unavailable'})
            continue
                                                                                     
                                                                                 
        import shutil
        private = OUT / 'nsw_sources'
        private.mkdir(exist_ok=True)
        for p in [req, response]:
            shutil.copy2(p, private / p.name)
        prior_raw = a.RAW
        a.RAW = private
        try:
            _, label, candidates, f, source = n.request(region, candidates)
        finally:
            a.RAW = prior_raw
        source['original_file'] = str(response.relative_to(ROOT))
        source['original_sha256'] = a.sha(response)
        arrays, station_ids = [], []
        for station in candidates:
            if f.empty:
                continue
            g = f[f.station == station]
            if g.duplicated(['timestamp', 'channel']).any():
                raise RuntimeError('Duplicate NSW station hours')
            X = g.pivot(index='timestamp', columns='channel', values='value').reindex(index=times, columns=['PM2.5', 'PM10']).to_numpy(dtype=float, copy=True)
            X[X < 0] = a.np.nan
            if a.qualified(X, split):
                station_ids.append(str(station))
                arrays.append(X)
        decisions.append({'region': region, 'eligible_sites': station_ids, 'reason': 'selected' if len(station_ids) >= 3 else 'fewer_than_three_training_eligible_sites'})
        if len(station_ids) < 3:
            continue
        name = a.store('nsw_'+label, a.np.stack(arrays[:4], axis=1), times, split, station_ids[:4], ['PM2.5', 'PM10'], ['ug/m3']*2,
            [source, {'file': str(sites_path.relative_to(ROOT)), 'sha256': a.sha(sites_path), 'url': 'https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails'}],
            {'provider': 'NSW', 'source_region': region, 'all_training_eligible_sites': station_ids,
             'timezone': 'Fixed AEST (UTC+10)', 'timestamp_convention': 'Hour 1 begins at 00:00; true hourly averages',
             'source_quality': 'Preserved official API response; ratification not asserted'})
        chosen_cohorts.append(name)
        print('NSW new cohort selected', name, station_ids[:4], flush=True)
    a.save('nsw_candidates.json', {'decisions': decisions})
    return chosen_cohorts

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    identity = {'protocol_sha256': a.sha(HERE / 'PROTOCOL.md'), 'acquisition_sha256': a.sha(Path(__file__)),
        'legacy_acquisition_sha256': a.sha(LEGACY / 'acquire.py'), 'legacy_nsw_sha256': a.sha(LEGACY / 'acquire_nsw.py'),
        'prior_cohorts_sha256': a.sha(OLD / 'cohorts.json'), 'user_date': '2026-10-12'}
    lock = OUT / 'acquisition_lock.json'
    if lock.exists() and json.loads(lock.read_text()) != identity:
        raise RuntimeError('Acquisition protocol/source changed after lock')
    a.save('acquisition_lock.json', identity)
    names = []
                                                                                     
    for provider, function in [('NSW', nsw), ('EPA', epa)]:
        marker = OUT / (provider+'_acquired.json')
        if marker.exists():
            cohorts = json.loads(marker.read_text())['cohorts']
        else:
            cohorts = function()
            a.save(marker.name, {'cohorts': cohorts})
        names.extend(cohorts)
        a.save('cohorts.json', {'eligible': names, 'scope': 'Additional recent geography cohorts; complete fixed-order batch, not test-score selection'})
    a.save('acquisition_status.json', {'status': 'completed', 'cohorts': names})

if __name__ == '__main__':
    main()
