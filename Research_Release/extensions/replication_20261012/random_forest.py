                                                                                    
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY = ROOT / 'Research_Release/extensions/weakness_resolution/nonlinear.py'
OUT = ROOT / 'Research_Release/replication_20261012_results'
source = LEGACY.read_text(encoding='utf-8').replace('audit_tools/weakness_packages', 'audit_tools/replication_packages')
source = source.replace("folders=sorted(p.parent for p in SOURCE.glob('*/outage_cases.csv'))", "folders=sorted(p.parent for p in SOURCE.glob('*/outage_cases.csv') if (p.parent/'evaluation_status.json').exists())")
module = types.ModuleType('replication_random_forest')
module.__file__ = str(LEGACY)
exec(compile(source, str(LEGACY), 'exec'), module.__dict__)
module.SOURCE = OUT
module.OUT = OUT / 'random_forest'
identity = {'protocol_sha256': module.sha(HERE / 'PROTOCOL.md'), 'adapter_sha256': module.sha(Path(__file__)),
    'original_kernel_sha256': module.sha(LEGACY), 'effective_kernel_sha256': module.hashlib.sha256(source.encode()).hexdigest(),
    'dependencies': {'scikit-learn': '1.9.1', 'joblib': '1.6.0', 'threadpoolctl': '3.7.0'}}
import json
lock = OUT / 'random_forest_batch_lock.json'
if lock.exists() and json.loads(lock.read_text()) != identity:
    raise RuntimeError('RF source/protocol changed after lock')
lock.write_text(json.dumps(identity, indent=2), encoding='utf-8')
if __name__ == '__main__':
    module.main()
