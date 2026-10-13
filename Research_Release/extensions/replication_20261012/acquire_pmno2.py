                                                                           
from pathlib import Path
import importlib.util
import json
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'Research_Release/replication_20261012_results'
OLD=ROOT/'Research_Release/expanded_validation_results'
LEGACY=ROOT/'Research_Release/extensions/expanded_validation/acquire.py'
spec=importlib.util.spec_from_file_location('pmno2_shared_ingestion',LEGACY)
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
a.OUT=OUT
a.HERE=HERE
RAW=OUT/'extra_epa_sources'

def download(name,url,cache=None):
    candidates=[OLD/'raw'/name,ROOT/'Research_Release/recent_data_results/epa_la/raw'/name,RAW/name]
    p=next((p for p in candidates if p.exists()),None)
    if p is None:
        raise FileNotFoundError('Acquire official source before ingestion: '+name)
    return p,{'file':str(p.relative_to(ROOT)),'url':url,'sha256':a.sha(p),'bytes':p.stat().st_size}
a.download=download

def main():
    RAW.mkdir(exist_ok=True)
    seen={tuple(map(int,p.name.split('_')[1:])) for parent in [OLD,OUT] for p in parent.glob('epa_*') if p.is_dir() and re.fullmatch(r'epa_\d+_\d+',p.name)}
    seen.add((6,37))
    identity={'protocol_sha256':a.sha(HERE/'PMNO2_PROTOCOL.md'),'acquisition_sha256':a.sha(Path(__file__)),
        'legacy_kernel_sha256':a.sha(LEGACY),'excluded_prior_counties':sorted(seen)}
    lock=OUT/'pmno2_acquisition_lock.json'
                                                                                   
    identity=json.loads(json.dumps(identity))
    if lock.exists() and json.loads(lock.read_text())!=identity:
        raise RuntimeError('PMNO2 acquisition lock changed')
    a.save(lock.name,identity)
    marker=OUT/'PMNO2_acquired.json'
    if marker.exists():
        selected=json.loads(marker.read_text())['cohorts']
    else:
        source=LEGACY.read_text(encoding='utf-8')
        source=source[source.index('def epa():'):source.index('\ndef uk():')]
        source=source.replace('[88101,88502,44201]','[88101,88502,42602]')
        source=source.replace("([44201],'Parts per million')","([42602],'Parts per billion')")
        source=source.replace('if (int(state),int(county))==(6,37):continue','if (int(state),int(county)) in PREVIOUS_COUNTIES:continue')
        source=source.replace('if len(selected)>=12:continue',"if len(selected)>=24:\n            excluded.append({'source':'EPA','state':int(state),'county':int(county),'reason':'Deferred beyond fixed 24-new-cohort cap'});continue")
        source=source.replace("name=f'epa_", "name=f'epa_pmno2_")
        source=source.replace("['PM2.5','O3_ppm'],['ug/m3','ppm']","['PM2.5','NO2_ppb'],['ug/m3','ppb']")
        source=source.replace("'epa_candidates.json'","'pmno2_candidates.json'")
        source=source.replace('First twelve qualifying state/county groups; no test-score selection','First 24 new PM2.5/NO2 county groups; no test-score selection')
        source=source.replace('PM mass and ozone volume units preserved; no conversion between them','Native PM2.5 ug/m3 and NO2 ppb preserved; no unit conversion')
        source=source.replace("'provider':'EPA'","'provider':'EPA','additional_protocol_sha256':ADDITIONAL_PROTOCOL_HASH")
        scope=dict(a.__dict__)
        scope['PREVIOUS_COUNTIES']=seen
        scope['ADDITIONAL_PROTOCOL_HASH']=identity['protocol_sha256']
        exec(compile(source,str(LEGACY),'exec'),scope)
        selected=scope['epa']()
        a.save(marker.name,{'cohorts':selected,'status':'completed'})
    cohorts=json.loads((OUT/'cohorts.json').read_text())
    cohorts['eligible']=list(dict.fromkeys(cohorts['eligible']+selected))
    a.save('cohorts.json',cohorts)
    print('Additional PM2.5/NO2 cohorts:',selected,flush=True)

if __name__=='__main__':
    main()
