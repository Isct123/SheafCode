                                                                           
from pathlib import Path
import sys, importlib, types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECENT = ROOT / 'Research_Release/extensions/recent_data'
sys.path.insert(0, str(RECENT))
import common
common.OUT = ROOT / 'Research_Release/uk_validation_results'
common.regions = lambda: ['uk_aurn']

def main():
    name = sys.argv[1]
    if name in ['final_analysis', 'fault_partial']:
        source = (RECENT / (name + '.py')).read_text().replace("regions()+['epa_la']", 'regions()')
        if name == 'fault_partial':
            source = source.replace('Availability extension after strict 24h complete-network criterion yielded zero episodes;', 'Available-candidate evaluation, separate from the strict whole-network episode study;')
        frozen = HERE / ('adapted_' + name + '.py')
        frozen.write_text(source, encoding='utf-8')
        module = types.ModuleType('uk_' + name); module.__file__ = str(frozen)
        exec(compile(source, str(frozen), 'exec'), module.__dict__)
    else: module = importlib.import_module(name)
    module.regions = lambda: ['uk_aurn']
    if name == 'fit': module.prepare = lambda: None
    module.main()

if __name__ == '__main__': main()
