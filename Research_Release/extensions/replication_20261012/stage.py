                                                                               
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECENT = ROOT / 'Research_Release/extensions/recent_data'
sys.path.insert(0, str(RECENT))
import common
common.OUT = ROOT / 'Research_Release/replication_20261012_results'
region, task = sys.argv[1:3]
common.regions = lambda: [region]
source = (RECENT / (task+'.py')).read_text(encoding='utf-8')
if task == 'evaluate':
                                                                                  
                                                                           
    cut = source.index('        # Snapshot and partial-channel controls')
    source = source[:cut]+"        save(region+'/evaluation_status.json',{'status':'completed','outage_rows':len(f),'all_outcomes_retained':True,'scope':'Seven-method 24-hour outage experiment only; other families not run'})\nif __name__=='__main__':main()\n"
module = types.ModuleType('replication_'+task)
module.__file__ = str(RECENT / (task+'.py'))
exec(compile(source, module.__file__, 'exec'), module.__dict__)
module.regions = lambda: [region]
if task == 'fit':
    module.prepare = lambda: None
common.lock(region+'/'+task+'_batch_lock.json', {
    'protocol_sha256': common.sha(HERE / 'PROTOCOL.md'), 'adapter_sha256': common.sha(Path(__file__)),
    'kernel_sha256': common.sha(RECENT / (task+'.py')), 'effective_kernel_sha256': common.hashlib.sha256(source.encode()).hexdigest(),
    'dataset_sha256': common.sha(common.OUT / region / 'dataset.npz')})
module.main()
