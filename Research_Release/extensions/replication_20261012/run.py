                                                                                  
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Research_Release/replication_20261012_results'

def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)

def cohort(name):
    folder = OUT / name
    logs = folder / 'logs'
    logs.mkdir(exist_ok=True)
    import numpy as np
    with np.load(folder / 'dataset.npz', allow_pickle=False) as ds:
        test_cells = int(np.isfinite(ds['X'][ds['split']==2]).sum())
    sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
    import common
    common.OUT=OUT
    dataset=common.load(name)
    p=dataset['Z'].shape[2]
    eligible_cases=sum(int((np.isfinite(dataset['Z'][ix[24:]]).sum(axis=(0,2))>=12*p).sum())
                       for _,ix in common.blocks(dataset,2))
    if not eligible_cases:
        value = {'cohort': name, 'status': 'unavailable_data',
                 'eligible_test_station_days':0,'test_observed_cells':test_cells,
                 'reason': 'No test station-day meets the fixed 12p target-cell rule; not an empirical success'}
        write(folder / 'run_status.json', value)
        print(name, 'UNAVAILABLE: no eligible test station-day', flush=True)
        return value
    for stage, marker in [('fit', 'models.pkl'), ('evaluate', 'evaluation_status.json')]:
        if (folder / marker).exists():
            continue
        write(folder / 'run_status.json', {'cohort': name, 'status': 'running', 'stage': stage,
            'pid': os.getpid(), 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        print(name, stage, flush=True)
        with (logs / (stage+'.log')).open('a', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, '-u', str(HERE / 'stage.py'), name, stage],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            value = {'cohort': name, 'status': 'failed', 'stage': stage, 'exit_code': result.returncode,
                'log': str((logs / (stage+'.log')).relative_to(ROOT))}
            write(folder / 'run_status.json', value)
            print(name, 'FAILED', stage, result.returncode, flush=True)
            return value
    value = {'cohort': name, 'status': 'completed', 'test_observed_cells': test_cells}
    write(folder / 'run_status.json', value)
    summary = json.loads((folder / 'reconstruction_summary.json').read_text())
    print(name, 'COMPLETE', 'primary gain vs VAR:',
        summary['primary_every_third_day']['var_causal'].get('gain_percent'), flush=True)
    return value

def main():
    names = sys.argv[1:] or json.loads((OUT / 'cohorts.json').read_text())['eligible']
    if not names:
        raise RuntimeError('No acquired cohort available')
                                                                                    
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(cohort, names))
    write(OUT / 'analysis_status.json', {'status': 'failed' if any(r['status']=='failed' for r in results) else 'completed',
        'requested_cohorts': names, 'cohorts': results})
    if any(r['status']=='failed' for r in results):
        raise RuntimeError('Some cohorts failed; see their logs')

if __name__ == '__main__':
    main()
