import sys, subprocess, urllib.request, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

def execute(script, args=None):
    logfile = OUT / (script.stem + '_run.log')
    print('Running', script.name, flush=True)
    with logfile.open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, str(script), *(args or [])], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f'{script.name} failed; see {logfile}')

def main():
    sources = {p.name: sha(p) for p in HERE.glob('*.py')}
    lock('RUN_LOCK.json', {'sources': sources, 'protocol_sha256': sha(HERE / 'PROTOCOL.md')})
    stages = [('replay.py', 'replay_summary.json'), ('beijing.py', 'beijing_summary.json'), ('reconcile_beijing_baseline.py', 'beijing_corrected_summary.json'), ('fault_episodes.py', 'persistent_fault_scope.json'), ('raw_gain_faults.py', 'raw_gain_scope.json'), ('uncertainty_shift.py', 'uncertainty_scope.json'), ('reconcile_uncertainty_intervals.py', 'uncertainty_interval_reconciliation.json'), ('exact_math.py', 'exact_math_results.json')]
    state = {'status': 'running', 'completed': [], 'active': None}
    save('STAGE_STATUS.json', state)
    try:
        if not (OUT / 'beijing_original.zip').exists():
            url = 'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
            lock('ACQUISITION_LOCK.json', {'protocol_sha256': sha(HERE / 'PROTOCOL.md'), 'dataset_url': url, 'fixed_before_acquisition': True})
            target = OUT / 'beijing_original.zip'
            temp = target.with_suffix('.download')
            result = subprocess.run(['curl.exe', '--fail', '--location', '--max-time', '90', '--output', str(temp), url])
            if result.returncode:
                raise RuntimeError('Public Beijing download failed')
            temp.replace(target)
        for script, artifact in stages:
            state['active'] = script
            save('STAGE_STATUS.json', state)
            if not (OUT / artifact).exists():
                execute(HERE / script)
            else:
                print('Using completed stage', script, flush=True)
            state['completed'].append(script)
            save('STAGE_STATUS.json', state)
        if not (OUT / 'official_acquisition.json').exists():
            execute(HERE / 'official_records.py', ['--acquire'])
        if not (OUT / 'official_summary.json').exists():
            execute(HERE / 'official_records.py')
        state['active'] = 'unit tests'
        save('STAGE_STATUS.json', state)
        result = subprocess.run([sys.executable, '-m', 'pytest', str(HERE / 'test_online_validation.py'), '-q', '-p', 'no:cacheprovider'], capture_output=True, text=True, cwd=ROOT)
        (OUT / 'TEST_RESULTS.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('New extension unit tests failed')
        for script in ['audit.py']:
            state['active'] = script
            save('STAGE_STATUS.json', state)
            execute(HERE / script)
        state['status'] = 'completed'
        state['active'] = None
        save('STAGE_STATUS.json', state)
        print('All six extension tracks completed.', flush=True)
    except BaseException as error:
        state['status'] = 'failed_or_interrupted'
        state['error'] = str(error)
        save('STAGE_STATUS.json', state)
        raise
if __name__ == '__main__':
    main()
