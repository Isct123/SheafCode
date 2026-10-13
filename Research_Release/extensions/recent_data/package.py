                                                                                  
import sys,zipfile,importlib.metadata,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def main():
    package=ROOT.parent/'IRIS_Submission_Recent_Data';review=package/'Review_Notes'
    evidence=json.loads((OUT/'FINAL_EVIDENCE_AUDIT.json').read_text());document=json.loads((review/'DOCUMENT_CHECKS.json').read_text())
    if evidence['status']!='passed' or document['status']!='passed':raise RuntimeError('Do not freeze failed audits')
    tests=(OUT/'logs/tests.log').read_text();
    if '6 passed' not in tests:raise RuntimeError('Recent tests are incomplete')
    versions={}
    for name in ['numpy','pandas','scipy','matplotlib','pymupdf','pytest']:
        versions[name]=importlib.metadata.version(name)
    torch_root=ROOT.parent/'audit_tools/torch_packages';sys.path.insert(0,str(torch_root));versions['torch']=importlib.metadata.version('torch')
    if not versions['torch']:
        distributions=list(torch_root.glob('torch-*.dist-info'))
        if len(distributions)!=1:raise RuntimeError('Ambiguous torch distribution version')
        versions['torch']=distributions[0].name[len('torch-'):-len('.dist-info')]
    versions['python']=sys.version;save('REPRODUCTION_ENVIRONMENT.json',{'versions':versions,'torch_location':'audit_tools/torch_packages; archived vendor architecture and checkpoints preserved','platform':sys.platform,'scope':'Pinned outputs/checkpoints, not a bundled Python distribution'})
    requirements='--extra-index-url https://download.pytorch.org/whl/cpu\n'+'\n'.join(f'{k}=={v}' for k,v in versions.items() if k!='python')+'\n'
    (HERE/'requirements.txt').write_text(requirements,encoding='utf-8')
    (review/'REPRODUCTION.md').write_text('''# Reproduction and release contents

The ZIP preserves the repository-relative layout needed by the recent-data scripts: Research_Release/extensions/recent_data, the referenced mathematical kernels and independent reference decoder, the pinned GRIN vendor source, Research_Release/recent_data_results, document sources and the report builder. Original OpenAQ compressed daily objects, EPA hourly ZIPs, CPCB PDFs, model/config locks, epoch checkpoints, predictions, case summaries, all plotted data, exact witnesses and audit outputs are included.

The Windows launcher in the original workspace uses its bundled Python. On an extracted release with a standard Python installation, install Research_Release/extensions/recent_data/requirements.txt and run `python Research_Release/extensions/recent_data/run.py`. The launcher expects audit_tools/python/python.exe, which is not bundled into the scientific ZIP. PDF generation requires Tectonic and its already-cached fonts/packages under audit_tools/tectonic and audit_tools/tex_cache; those are environmental prerequisites, not part of the scientific ZIP. Code/data/checkpoint identities must match for resume. Internet is needed only if original sources are missing; source URLs and hashes are preserved. Stored own-output pickle/checkpoints are trusted local model artifacts and should not be confused with arbitrary downloaded pickles.

Historical observation archives are not included in this recent-only release. Their validated research package and full manuscript remain untouched in the original project. A data-independent proof source is reused with recorded provenance, without implying that its original empirical datasets train recent models. The internal audit still references the historical manuscript hash in the full workspace; an external extraction can inspect that recorded hash but requires that separate manuscript for that optional preservation check.

Review code/parameter choices and study scopes before changing settings. The primary test is fixed; a further test-selected variant must not replace it under the same confirmatory description. This package captures completed evidence, not a guarantee of cross-machine bitwise-identical neural optimization or organiser eligibility.
''',encoding='utf-8')
    files=set()
    for directory in [OUT,HERE,package]:
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix not in ['.tmp','.download','.pyc','.zip'] and '__pycache__' not in p.parts and 'plot_cache' not in p.parts:files.add(p)
                                                                                         
    files.update((OUT/'epa_la/raw').glob('*.zip'))
    for directory in ['online_validation','structured','functional','followup','joint_decisions','validation']:
        for p in (ROOT/'extensions'/directory).glob('*.py'):files.add(p)
    for p in (ROOT/'extensions/validation/vendor/grin').rglob('*'):
        if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts:files.add(p)
    files.update((ROOT/'sheafcode').glob('*.py'))
    files.add(ROOT.parent/'audit_tools/build_iris_submission.py');files.add(ROOT.parent/'Final_Manuscript/Extended_Validation/references.bib');files.add(ROOT.parent/'Final_Manuscript/Extended_Validation/sections/10_proofs.tex');files.add(ROOT.parent/'Final_Manuscript/SheafCode_Validated_Manuscript.pdf');files.add(ROOT.parent/'Run_Recent_Data_Research.cmd')
    files.add(ROOT/'validation_results/external/observations.csv');files.add(ROOT/'validation_results/external/station_mapping.json')
    files={p for p in files if p.name not in ['RELEASE_MANIFEST.json','PACKAGE_VERIFICATION.json','FINAL_REVIEW.md','status.json']}
    entries=[{'path':p.relative_to(ROOT.parent).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
    save('RELEASE_MANIFEST.json',{'date':'2026-10-07','files':entries,'evidence_checks':evidence['check_count'],'exact_rational_checks':37678,'code_tests':6,'document_checks':len(document['checks']),'scope':'Frozen recent-only empirical evidence plus reusable mathematical source; organiser acceptance and unrestricted priority are not certified'})
    files.add(OUT/'RELEASE_MANIFEST.json');target=package/'Supporting_Reference/SheafCode_Recent_Reproducibility.zip';temp=target.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT.parent).as_posix())
    temp.replace(target)
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('ZIP CRC failure: '+bad)
        for item in entries:
            if hashlib.sha256(z.read(item['path'])).hexdigest()!=item['sha256']:raise RuntimeError('ZIP hash mismatch: '+item['path'])
    verification={'status':'passed','zip':str(target.relative_to(ROOT.parent)),'bytes':target.stat().st_size,'sha256':sha(target),'members':len(files),'file_hashes_checked':len(entries),'crc_checked':True}
    save('PACKAGE_VERIFICATION.json',verification);(review/'PACKAGE_VERIFICATION.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    (review/'FINAL_REVIEW.md').write_text(f'''# Final recent-data review

Completed 7 October 2026. Both 2026-only empirical networks, all compatible task families, every planned neural seed and the document build are complete. Six recent code tests passed; {evidence['check_count']:,} evidence checks passed; 37,678 exact finite rational checks passed; {len(document['checks']):,} manuscript/synopsis/reference document checks passed. The archive passed CRC and {len(entries):,} exact member-hash checks.

The fixed primary recent result is Delhi 16.48% normalized RMSE reduction against causal tuned VAR (77 paired cases, 16 dates; exploratory weekly-bin interval 12.62–20.16%). Secondary Los Angeles County gives 8.68%, with too few bins for an interval. All-case, physical-unit, neural, uncertainty, fault, environmental-proxy and negative ablation outcomes are preserved. Full mathematical assumptions/proofs and all case-table references accompany the paper. No old evidence was deleted or silently relabelled recent.

Final recheck independently reproduced all seven reconstruction methods across all 310 eligible station-days (2,172 inference checks). Every validation-chosen neural checkpoint reproduced all 242 cases for each of three seeds. Mathematical statements explicitly define finite state boxes, the column-gain convention and positive noisy fault counts. Prior source overlap is quantified and disclosed; no blinded new-observation validation is claimed. Upload networks are anonymized A/B without changing data or outcomes.

This is an internal scientific/reproducibility review. It cannot certify first-ever mathematical priority, complete physical uncertainty, deployment impact, independent authorship or IRIS acceptance. Existing specialized proofs have a useful role; adding theorem labels without a new substantive result would not improve the current study. Further research remains possible, but no additional theorem is necessary to make this documented protocol complete. The recent core uses observations recorded within twelve months, but signed-in application requirements, student/guide records and video remain the submitter's actions. No further long run is required to obtain the results in this package.
''',encoding='utf-8')
    print('Recent release archive verified:',len(entries),'file hashes,',target.stat().st_size,'bytes',flush=True)
if __name__=='__main__':main()
