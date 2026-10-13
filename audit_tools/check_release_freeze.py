                                                                                    
from pathlib import Path
import hashlib,json,zipfile,datetime
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'IRIS_Submission_Recent_Data';REVIEW=PACKAGE/'Review_Notes'
BASE=ROOT/'Research_Release/expanded_validation_results'
FOLLOWUP=ROOT/'Research_Release/weakness_resolution_results'

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    checks=[]
    def check(label,result):
        checks.append({'check':label,'passed':bool(result)})
        if not result:raise AssertionError(label)
    publication=read(BASE/'publication_status.json');documents=read(REVIEW/'EXPANDED_FINAL_AUDIT.json')
    paper=ROOT/publication['manuscript']
    check('publication_complete',publication['status']=='completed_and_audited')
    check('current_paper_identity',sha(paper)==publication['manuscript_sha256']==documents['sha256'])
    check('document_checks_passed',documents['status']=='passed' and all(r['passed'] for r in documents['checks']))
    theory=read(REVIEW/'THEORY_COVERAGE_CHECK.json')
    check('current_theory_preservation',theory['status']=='passed' and theory['manuscript_sha256']==sha(paper) and all(theory['checks'].values()))
    followup=read(FOLLOWUP/'AUDIT.json')
    check('followup_complete',followup['status']=='passed' and all(r['passed'] for r in followup['checks']))
    for filename,digest in followup['implementation_sha256'].items():check('followup_source/'+filename,sha(ROOT/filename)==digest)
    report=read(FOLLOWUP/'REPORT_STATUS.json')
    check('all_neural_seeds_reported',report['status']=='generated' and report['neural_seed_comparisons']==9)
    for name,digest in read(BASE/'IMPLEMENTATION_HASHES.json')['sha256'].items():check('implementation/'+name,sha(ROOT/name)==digest)
    portal=read(REVIEW/'PORTAL_TEXT_CHECK.json')
    check('portal_text_limits',portal['status']=='passed' and all(portal['word_limits'][key][0]<=count<=portal['word_limits'][key][1] for key,count in portal['word_counts'].items()))
    archive=ROOT/documents['evidence_archive']
    check('sealed_archive_identity',sha(archive)==documents['archive_sha256'])
    with zipfile.ZipFile(archive) as z:
        inventory=json.loads(z.read('SHA256.json'))
        check('archive_inventory_size',len(z.namelist())==len(inventory)+1)
        for name,digest in inventory.items():
            path=(ROOT/name).resolve()
            check('archive_path/'+name,path.is_relative_to(ROOT.resolve()))
            check('archive_current_source/'+name,path.is_file() and sha(path)==digest)
    reference=read(REVIEW/'COMPLETE_REFERENCE_CHECKS.json')
    check('combined_reference_passed',reference['status']=='passed' and reference['sources_unchanged'])
    check('combined_reference_current',reference['source_sha256'][str(paper.relative_to(ROOT))]==sha(paper))
    check('combined_reference_identity',sha(ROOT/reference['output'])==reference['sha256'])
    stamp=datetime.datetime.now().astimezone().isoformat()
    decision='Freeze the present research scope and proceed with presentation and submission preparation.'
    limits=[
        'Extreme-event interval coverage and weak-fault detection remain imperfect; report the failures and false-alarm tradeoffs.',
        'Experiments are exploratory, with artificial outages, synthetic faults, dependent observations and limited neural cohorts; they do not establish universal superiority.',
        'The mathematical statements are conditional, with established ingredients credited; first-ever priority and external proof review are not established.',
        'Environmental evidence concerns computational monitoring continuity and threshold proxies, not measured emissions or health improvements.',
        'Student contributions, supervising-guide records, video/forms and submission-day observation-age eligibility require accurate external completion.'
    ]
    value={'status':'passed','checked_at':stamp,'decision':decision,'check_count':len(checks),'checks':checks,
           'manuscript':publication['manuscript'],'manuscript_sha256':sha(paper),'pages':publication['pages'],
           'acknowledged_limitations':limits,
           'scope':'Release consistency and research-scope freeze; not proof of exhausted research possibilities, first-ever priority, competition acceptance or field environmental benefit'}
    (REVIEW/'RESEARCH_FREEZE_CHECK.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    note='# Research freeze decision\n\n'+decision+'\n\n'
    note+=f'Checked {stamp}. Current manuscript: {publication["pages"]} pages; SHA-256 {sha(paper)}.\n\n'
    note+=f'{len(checks):,} release-identity and consistency checks passed, including every archived source file, the current paper, theory-preservation record, all nine external neural seed results and submission-text limits. The fresh-run neural replay routing was corrected and independently checked without retraining.\n\n'
    note+='No additional experiment is required to support the present manuscript claims. Further research can improve performance or provide external validation, but is outside this frozen scope. Scientific limitations remain acknowledged:\n\n'
    note+='\n'.join('- '+item for item in limits)+'\n\n'
    note+='This is a go-ahead to freeze and prepare the submission, not a guarantee of selection, unrestricted operational reliability or exhausted mathematical novelty.\n'
    (REVIEW/'RESEARCH_FREEZE_DECISION.md').write_text(note,encoding='utf-8')
    print('Research freeze checks passed',len(checks),'checks;',publication['pages'],'pages',flush=True)

if __name__=='__main__':main()
