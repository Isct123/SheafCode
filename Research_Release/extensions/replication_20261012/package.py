                                                                                    
from pathlib import Path
import hashlib
import json
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'Research_Release/replication_20261012_results'
DEST=ROOT/'Research_Release/Additional_Recent_Data_20261012_Review.zip'

def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()

def main():
    audit=json.loads((OUT/'audit.json').read_text())
    assert audit['status']=='passed', 'Finish the audit before packaging.'
    notice=(
        'Start with Research_Release/replication_20261012_results/RESULTS.md.\n'
        'This review archive includes the new results, models, tensors, acquisition records, '
        'protocols and extension scripts. All completed comparisons, including losses, are retained.\n'
        'It is not a standalone software distribution. The original shared EPA archives, '
        'historical inference kernels and Python environment remain in the existing SheafCode '
        'workspace. Source paths, URLs and hashes are preserved in dataset manifests.\n'
        'Run_Additional_Recent_Data_Research.ps1 resumes the frozen batch in that workspace; '
        'it does not continuously discover new periods or datasets.\n'
        'Ancillary spatial CCA controls, where present, are separate from the main VAR comparison.\n'
    )
    inputs=list(HERE.rglob('*'))+list(OUT.rglob('*'))+[ROOT/'Run_Additional_Recent_Data_Research.ps1']
    inputs=sorted(p for p in inputs if p.is_file() and '__pycache__' not in p.parts
                  and p.suffix not in {'.tmp','.download','.pyc'} and p.name!='package_manifest.json')
    members=[{'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in inputs]
    manifest={'purpose':'Review bundle; full reproduction uses the existing workspace',
              'audit_status':audit['status'],'cohorts_audited':len(audit['checks']),
              'all_outcomes_retained':True,'files':members}
    (OUT/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        archive.writestr('READ_ME_FIRST.txt',notice)
        archive.writestr('package_manifest.json',json.dumps(manifest,indent=2)+'\n')
        for path,item in zip(inputs,members):
            archive.write(path,item['file'])
    with zipfile.ZipFile(DEST) as archive:
        assert archive.testzip() is None
        for item in members:
            assert hashlib.sha256(archive.read(item['file'])).hexdigest()==item['sha256']
    print(json.dumps({'archive':str(DEST.relative_to(ROOT)),'bytes':DEST.stat().st_size,
                      'sha256':sha(DEST),'files':len(members),'cohorts':len(audit['checks'])},indent=2))

if __name__=='__main__':main()
