                                                                                   
from pathlib import Path
import subprocess,sys,os,json,contextlib,time
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];RESEARCH=ROOT/'Research_Release'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(RESEARCH/'extensions/online_validation'))
results=[]
followup_only='--followup-only' in sys.argv
for name,relative in [('repair_optima','extensions/math_integration/independent_optima.py'),('exact_rank','extensions/math_integration/exact_graph_rank.py')]:
    if followup_only:continue
    source=RESEARCH/relative;run=subprocess.run([sys.executable,str(source),'--check-only'],cwd=ROOT,capture_output=True,text=True)
    (OUT/('whole_'+name+'.log')).write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
    assert run.returncode==0,(name,run.stderr[-1000:])
    parsed=[json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
    results.append({'name':name,'status':'passed','summary':parsed[-1]});print(name,'passed',flush=True)

for name,relative,before,after in [
 ('joint','extensions/math_integration/replay_joint.py',"(OUT/'INDEPENDENT_REPLAY.json')","(CHECK_OUT/'WHOLE_JOINT_REPLAY.json')"),
 ('archive_transfer','extensions/decision_repair/replay_heldout.py',"(directory/'INDEPENDENT_REPLAY.json')","(CHECK_OUT/'WHOLE_ARCHIVE_REPLAY.json')"),
 ('forests_and_intervals','extensions/weakness_resolution/audit.py',"(OUT/'AUDIT.json')","(CHECK_OUT/'WHOLE_FOLLOWUP_REPLAY.json')")]:
    if followup_only and name!='forests_and_intervals':continue
    source=RESEARCH/relative;code=source.read_text(encoding='utf-8');assert code.count(before)==1
    code=code.replace(before,after);ns={'__file__':str(source),'__name__':'read_only_verifier','CHECK_OUT':OUT}
    sys.argv=[str(source)];sys.path.insert(0,str(source.parent));started=time.time()
    with (OUT/('whole_'+name+'.log')).open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        exec(compile(code,str(source),'exec'),ns);ns['main']()
    results.append({'name':name,'status':'passed','elapsed_seconds':time.time()-started});print(name,'passed',round(time.time()-started,1),'seconds',flush=True)
(OUT/('WHOLE_FOLLOWUP_STATUS.json' if followup_only else 'WHOLE_REPLAY_STATUS.json')).write_text(json.dumps({'status':'passed','verifications':results},indent=2)+'\n',encoding='utf-8')
