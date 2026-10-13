                                                                          
import sys,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def main():
    if not (OUT/'epa_la/models.pkl').exists():save('epa_la/secondary_status.json',{'status':'inapplicable_no_eligible_network'});return
    for name,status in [('faults','fault_status.json'),('fault_partial','available_fault_status.json'),('uncertainty','uncertainty_status.json'),('mathematics','mathematics_status.json'),('extensions','extensions_status.json'),('environment','environment_status.json'),('additional','additional_status.json')]:
        p=OUT/'epa_la'/status
        if p.exists() and (str(json.loads(p.read_text()).get('status')).startswith('completed') or name=='faults'):continue
        module=importlib.import_module(name);module.regions=lambda:['epa_la'];module.main()
    save('epa_la/secondary_status.json',{'status':'completed','scope':'Recent US network uses the same compatible secondary task families; no US regulatory decision claim. GRIN is a Delhi-only neural benchmark.'})
if __name__=='__main__':main()
