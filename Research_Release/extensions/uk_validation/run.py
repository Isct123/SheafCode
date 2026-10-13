from pathlib import Path
import sys, subprocess, json, os, time
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Research_Release/uk_validation_results'

def status(value):
    p = OUT / 'status.json'
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(p)

def execute(script, args=()):
    stage = args[0] if args else script.stem
    status({'status': 'running', 'stage': stage, 'pid': os.getpid(), 'updated': time.strftime('%Y-%m-%d %H:%M:%S')})
    log = OUT / 'logs' / (stage + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    print('UK stage:', stage, flush=True)
    with log.open('a', encoding='utf-8') as f:
        result = subprocess.run([sys.executable, str(script), *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        status({'status': 'failed', 'stage': stage, 'exit_code': result.returncode, 'log': str(log)})
        raise RuntimeError('UK stage failed; see ' + str(log))

def main():
    OUT.mkdir(exist_ok=True)
    if not (OUT / 'acquisition_status.json').exists():
        execute(HERE / 'acquire.py')
    if not (OUT / 'uk_aurn/models.pkl').exists():
        execute(HERE / 'stage.py', ['fit'])
    for name, marker in [('evaluate', 'evaluation_status.json'), ('faults', 'fault_status.json'), ('fault_partial', 'available_fault_status.json'), ('uncertainty', 'uncertainty_status.json'), ('mathematics', 'mathematics_status.json'), ('extensions', 'extensions_status.json'), ('environment', 'environment_status.json'), ('additional', 'additional_status.json'), ('operator_support', 'operator_support_status.json')]:
        if not (OUT / 'uk_aurn' / marker).exists():
            execute(HERE / 'stage.py', [name])
    execute(HERE / 'stage.py', ['final_analysis'])
    execute(HERE / 'audit.py')
    status({'status': 'completed', 'all_required_uk_stages_done': True, 'remaining_long_runs': False, 'scope': 'All compatible non-neural UK analyses and independent inference replay; existing neural benchmark remains Indian-only'})
    print('UK external validation complete.', flush=True)
if __name__ == '__main__':
    main()
