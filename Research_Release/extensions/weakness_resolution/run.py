from pathlib import Path
import sys, subprocess, json, concurrent.futures, os
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'Research_Release/weakness_resolution_results'

def command(path, *args):
    subprocess.run([sys.executable, str(path), *args], cwd=ROOT, check=True)

def active_pid(pid):
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(1048576, False, pid)
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 258
    finally:
        kernel.CloseHandle(handle)

def split_replays(combined, selected):
    names = [item['cohort'] for item in selected]
    if combined.get('status') != 'passed' or len(set(names)) != len(names):
        raise RuntimeError('Invalid combined neural replay')
    records = combined.get('checks', [])
    if len(records) != 3 * len(names) or {r['cohort'] for r in records} != set(names):
        raise RuntimeError('Combined replay has missing or unexpected cohorts')
    result = {}
    for name in names:
        checks = [r for r in records if r['cohort'] == name]
        if sorted((r['seed'] for r in checks)) != [0, 1, 2] or not all((r['passed'] for r in checks)):
            raise RuntimeError('Incomplete neural seed replay: ' + name)
        result[name] = {'status': 'passed', 'vendor_commit': combined['vendor_commit'], 'checks': checks}
    return result

def main():
    OUT.mkdir(exist_ok=True)
    for progress in (OUT / 'neural').glob('**/neural_progress.json'):
        value = json.loads(progress.read_text())
        if value.get('status') == 'running' and value.get('pid') and active_pid(int(value['pid'])):
            raise RuntimeError('An existing neural worker is still running; do not duplicate it: ' + str(progress))
    selection = OUT / 'neural/SELECTION.json'
    if not selection.exists():
        command(HERE / 'neural_external.py')
        combined = json.loads((OUT / 'neural/REPLAY.json').read_text())
        selected = json.loads(selection.read_text())['selections']
        for name, replay in split_replays(combined, selected).items():
            target = OUT / 'neural' / (name + '_REPLAY.json')
            temp = target.with_suffix('.tmp')
            temp.write_text(json.dumps(replay, indent=2))
            temp.replace(target)
    else:
        selected = json.loads(selection.read_text())['selections']

        def worker(item):
            name = item['cohort']
            log = OUT / 'neural' / name / 'resume.log'
            with log.open('a') as stream:
                subprocess.run([sys.executable, str(HERE / 'neural_worker.py'), name], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, item) for item in selected]
            for future in futures:
                future.result()
    command(HERE / 'nonlinear.py')
    command(HERE / 'uncertainty_decisions.py')
    command(HERE / 'selection_guard.py')
    command(HERE / 'roc_uncertainty.py')
    command(HERE / 'fault_temporal.py')
    command(HERE / 'width_control.py')
    command(HERE / 'threshold_sensitivity.py')
    command(HERE / 'audit.py')
    print('Follow-up research and evidence audit completed.', flush=True)
if __name__ == '__main__':
    main()
