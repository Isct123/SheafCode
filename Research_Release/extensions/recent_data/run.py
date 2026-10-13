                                                                               
import sys,subprocess,time,ctypes
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def alive(pid):
    if os.name!='nt':return False
    kernel=ctypes.windll.kernel32;handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:return False
    code=ctypes.c_ulong();ok=kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle);return bool(ok and code.value==259)
def execute(name,args=()):
    logs=OUT/'logs';logs.mkdir(exist_ok=True);save('status.json',{'status':'running','stage':name,'runner_pid':os.getpid()})
    print('Running',name,flush=True)
    with (logs/(Path(name).stem+'.log')).open('a',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,str(HERE/name),*args],cwd=ROOT.parent,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:save('status.json',{'status':'failed','stage':name,'exit_code':r.returncode,'log':str(logs/(Path(name).stem+'.log'))});raise RuntimeError('Stage failed: '+name)
def finished(path):
    p=OUT/path
    return p.exists() and str(json.loads(p.read_text()).get('status','')).startswith('completed')
def main():
    if not finished('availability_expansion/acquisition_summary.json'):execute('acquire.py',['2026-01-01'])
    if not (OUT/'delhi/models.pkl').exists():execute('fit.py')
    for name,status in [('evaluate.py','evaluation_status.json'),('faults.py','fault_status.json'),('fault_partial.py','available_fault_status.json'),('uncertainty.py','uncertainty_status.json'),('mathematics.py','mathematics_status.json'),('extensions.py','extensions_status.json'),('environment.py','environment_status.json'),('additional.py','additional_status.json')]:
        if name=='faults.py' and (OUT/'delhi'/status).exists():continue
        if not finished('delhi/'+status):execute(name)
    if not finished('epa_la/availability_status.json'):execute('epa.py')
    if not finished('epa_la/secondary_status.json'):execute('secondary.py')
    if not finished('delhi/operator_support_status.json'):execute('operator_support.py')
    if not finished('delhi/official_summary.json'):execute('official.py')
    p=OUT/'neural_progress.json'
    if p.exists():
        progress=json.loads(p.read_text())
        if progress.get('status')=='running' and alive(progress.get('pid',0)):
            print('Waiting for the existing neural worker; no duplicate training launched.',flush=True)
            while alive(progress['pid']):
                save('status.json',{'status':'running','stage':'existing neural worker','runner_pid':os.getpid()});time.sleep(30)
    if not finished('neural_progress.json'):execute('neural.py')
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    with (logs/'tests.log').open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,'-m','pytest',str(HERE/'test_recent.py'),'-q'],cwd=ROOT.parent,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Recent-data tests failed')
    if not finished('epa_la/operator_support_status.json'):execute('operator_support.py')
    if not (OUT/'exact_math_results.json').exists():execute('exact_math.py')
    execute('final_analysis.py');execute('final_confirmation.py')
    if not (OUT/'NEURAL_CONFIRMATION_REPLAY.json').exists():execute('neural_confirmation.py')
    execute('audit.py');execute('report.py');execute('package.py');save('status.json',{'status':'completed','all_required_stages_done':True,'report_folder':'IRIS_Submission_Recent_Data','remaining_long_runs':False});print('Recent-data experiments and documents completed.',flush=True)
if __name__=='__main__':main()
