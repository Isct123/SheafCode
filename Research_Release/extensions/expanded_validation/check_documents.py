                                                                               
from pathlib import Path
import json,re,hashlib,zipfile
import numpy as np
import pandas as pd
import pymupdf as fitz
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'Research_Release/expanded_validation_results';PACKAGE=ROOT/'IRIS_Submission_Recent_Data';SRC=PACKAGE/'Expanded_Evidence_Source';REVIEW=PACKAGE/'Review_Notes'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    import runpy
    runpy.run_path(str(HERE/'check_theory_continuity.py'),run_name='__main__')
    runpy.run_path(str(HERE/'prepare_portal.py'),run_name='__main__')
    checks=[]
    def check(label,result):
        checks.append({'check':label,'passed':bool(result)})
        if not result:raise AssertionError(label)
    meta=json.loads((OUT/'publication_status.json').read_text());path=ROOT/meta['manuscript'];table=pd.read_csv(OUT/'COHORT_RESULTS.csv');external=table[table.provider!='OpenAQ']
    theory=json.loads((REVIEW/'THEORY_COVERAGE_CHECK.json').read_text())
    check('theory_preservation',theory['status']=='passed' and all(theory['checks'].values()))
    check('theory_current_manuscript',theory['manuscript_sha256']==sha(path))
    followup=ROOT/'Research_Release/weakness_resolution_results'
    followup_audit=json.loads((followup/'AUDIT.json').read_text())
    check('weakness_followup_audit',followup_audit['status']=='passed')
    for name,digest in followup_audit['implementation_sha256'].items():check('weakness_source/'+name,sha(ROOT/name)==digest)
    followup_report=json.loads((followup/'REPORT_STATUS.json').read_text())
    check('external_neural_all_seeds',followup_report['neural_seed_comparisons']==9)
    check('followup_report_complete',followup_report['status']=='generated')
    grid_audit=json.loads((followup/'threshold_grid/AUDIT.json').read_text())
    check('concentration_grid_audit',grid_audit['status']=='passed')
    portal=json.loads((REVIEW/'PORTAL_TEXT_CHECK.json').read_text())
    check('portal_field_word_limits',portal['status']=='passed' and all(portal['word_limits'][k][0]<=v<=portal['word_limits'][k][1] for k,v in portal['word_counts'].items()))
    check('cohort_count',len(external)==meta['new_external_cohorts']);check('external_denominator',int(external.cases.sum())==meta['new_external_station_days']);check('sign_count',int((external.gain>0).sum())==meta['positive_var_gain_cohorts'])
    for r in table.itertuples():
        folder=OUT/r.cohort;audit=json.loads((folder/'AUDIT.json').read_text());check(r.cohort+'/audit',audit['status']=='passed');check(r.cohort+'/denominator',audit['all_method_station_days_replayed']==r.cases)
        ds=np.load(folder/'dataset.npz',allow_pickle=False);check(r.cohort+'/recent_age',pd.Timestamp(ds['times'][0])>=pd.Timestamp('2025-10-14'));check(r.cohort+'/no_future_observations',pd.Timestamp(ds['times'][-1])<pd.Timestamp('2026-10-08'))
        for cases,summary,alarm,countflag in [('fault_episode_cases.csv','fault_summary.csv','alarms',None),('available_fault_episode_cases.csv','available_fault_summary.csv','alarm_rate','count_correct'),('available_fault_hour_cases.csv','available_fault_hour_summary.csv','alarm_rate','count_correct')]:
            if not (folder/summary).exists():continue
            cf=pd.read_csv(folder/cases);sf=pd.read_csv(folder/summary)
            for row in sf.itertuples():
                g=cf[cf.kind.eq(row.kind)&cf.true_count.eq(row.true_count)]
                check(r.cohort+'/'+summary+'/'+row.kind+str(row.true_count)+'/count',len(g)==row.cases)
                check(r.cohort+'/'+summary+'/'+row.kind+str(row.true_count)+'/alarm',abs(g.alarm.mean()-getattr(row,alarm))<1e-12)
                check(r.cohort+'/'+summary+'/'+row.kind+str(row.true_count)+'/exact',abs(g.exact.mean()-row.exact_support)<1e-12)
                if countflag:check(r.cohort+'/'+summary+'/'+row.kind+str(row.true_count)+'/unknown_count',abs(g[countflag].mean()-row.count_accuracy)<1e-12)
        alerts=pd.read_csv(folder/'hourly_alert_predictions.csv');summary=pd.read_csv(folder/'alert_summary.csv')
        for row in summary.itertuples():
            g=alerts[alerts.method.eq(row.method)];positive=g[g.positive];negative=g[~g.positive]
            check(r.cohort+'/alerts/'+row.method+'/denominators',len(g)==row.cases and len(positive)==row.positive_cases and len(negative)==row.negative_cases)
            if len(positive):check(r.cohort+'/alerts/'+row.method+'/sensitivity',abs(positive.alarm.mean()-row.sensitivity)<1e-12)
            else:check(r.cohort+'/alerts/'+row.method+'/undefined_sensitivity',pd.isna(row.sensitivity))
            if len(negative):check(r.cohort+'/alerts/'+row.method+'/false_alarms',abs(negative.alarm.mean()-row.false_alarm_rate)<1e-12)
        masks=pd.read_csv(folder/'matched_missing_mask_cases.csv');ms=json.loads((folder/'matched_missing_mask_summary.json').read_text());check(r.cohort+'/actual_mask/denominator',len(masks)==ms['cases'])
        if len(masks):
            for method in ['sheaf','city']:check(r.cohort+'/actual_mask/'+method+'_rmse',abs(np.sqrt(np.mean((masks[method+'_estimate']-masks.truth)**2))-ms[method+'_pm25_rmse'])<1e-10)
    abstract=(PACKAGE/'Upload_Files/Project_Abstract_Expanded.txt').read_text();check('abstract_word_limit',len(abstract.split())<=250)
    check('manuscript_size',path.stat().st_size<20*1024*1024);check('manuscript_hash',sha(path)==meta['manuscript_sha256'])
    synopsis=PACKAGE/'Upload_Files/SheafCode_Full_Empirical_Synopsis.pdf'
    check('synopsis_size',synopsis.stat().st_size<5*1024*1024)
    with fitz.open(synopsis) as syn:
        for i,page in enumerate(syn):
            check(f'synopsis_page_{i+1}/encoding','\ufffd' not in page.get_text())
            for j,b in enumerate(page.get_text('dict')['blocks']):
                x0,y0,x1,y1=b['bbox'];check(f'synopsis_page_{i+1}/block_{j}/bounds',x0>=-.5 and y0>=-.5 and x1<=page.rect.width+.5 and y1<=page.rect.height+.5)
    for name,digest in json.loads((OUT/'IMPLEMENTATION_HASHES.json').read_text())['sha256'].items():check('implementation/'+name,sha(ROOT/name)==digest)
    log=(SRC/'expanded_full_manuscript.log').read_text(errors='replace');check('resolved_references',not re.search(r'(?:Citation|Reference).{0,180}undefined|multiply defined|undefined references',log,re.I))
    texts=[];previews=[]
    with fitz.open(path) as doc:
        pages=len(doc)
        for i,page in enumerate(doc):
            text=page.get_text();texts.append(text);check(f'page_{i+1}/encoding','\ufffd' not in text);check(f'page_{i+1}/no_unresolved_reference','??' not in text)
            for j,b in enumerate(page.get_text('dict')['blocks']):
                x0,y0,x1,y1=b['bbox'];check(f'page_{i+1}/block_{j}/bounds',x0>=-.5 and y0>=-.5 and x1<=page.rect.width+.5 and y1<=page.rect.height+.5)
            if 'All added cohorts' in text or 'No-feedback PM2.5 intervals' in text or 'Two-fault injected' in text or 'Broad external empirical validation' in text:previews.append((i,page.get_pixmap(matrix=fitz.Matrix(.7,.7))))
        contact=fitz.open()
        for offset in range(0,len(previews),6):
            page=contact.new_page(width=900,height=650)
            for j,(idx,pix) in enumerate(previews[offset:offset+6]):
                x=j%3*300;y=j//3*325;page.insert_text((x+5,y+12),'Manuscript page '+str(idx+1),fontsize=9);page.insert_image(fitz.Rect(x+5,y+20,x+295,y+320),pixmap=pix)
        if len(contact):
            contact.save(REVIEW/'EXPANDED_DOCUMENT_CONTACT.pdf')
            for i,p in enumerate(contact):p.get_pixmap().save(REVIEW/('EXPANDED_DOCUMENT_CONTACT_'+str(i+1)+'.png'))
    text='\n'.join(texts)
    check('all_codes_in_manuscript',all(str(x) in text for x in table.code))
    check('negative_results_retained','negative' in text.lower());check('limitations_present','first-ever' in text and 'fourteen' in text and 'not independent' in text)
    check('no_iteration_labels',not re.search(r'\bv[1-9]\b',re.sub(r'https?://\S+','',text)))
    check('original_manuscript_preserved',sha(ROOT/'Final_Manuscript/SheafCode_Validated_Manuscript.pdf')=='90ad9d477dae3b6e5421f2b315e328b715ba9e155a49fc5009b0e483a2ea2b81')
    archive=PACKAGE/'Supporting_Reference/SheafCode_Broad_Empirical_Evidence.zip';files=list(OUT.rglob('*'))+list(HERE.rglob('*'))+list(SRC.rglob('*'))
    for directory in ['recent_data','online_validation','structured','functional','followup','joint_decisions','operator_uncertainty','validation']:
        files+=list((ROOT/'Research_Release/extensions'/directory).rglob('*.py'))
    files.append(ROOT/'Run_Broad_Empirical_Validation.cmd')
    files+=list(followup.rglob('*'))
    files+=list((ROOT/'Research_Release/extensions/weakness_resolution').rglob('*'))
    files+=list((ROOT/'Research_Release/extensions/validation/vendor/grin').rglob('*'))
    files.append(ROOT/'Run_Weakness_Resolution.cmd')
    files.append(REVIEW/'WEAKNESS_RESOLUTION_REVIEW.md')
    files.append(ROOT/'audit_tools/build_iris_submission.py')
    files.append(ROOT/'Final_Manuscript/SheafCode_Validated_Manuscript.pdf')
    files+=list((PACKAGE/'Three_Network_Source').rglob('*'))
    files+=[path,synopsis,PACKAGE/'Upload_Files/Project_Abstract_Expanded.txt',REVIEW/'EXPANDED_EVIDENCE_REVIEW.md']
    files+=[REVIEW/'THEORY_COVERAGE_CHECK.md',REVIEW/'THEORY_COVERAGE_CHECK.json']
    files.append(REVIEW/'THEOREM_SELECTION_AND_PROGRESSION.md')
    files+=[PACKAGE/'Upload_Files/Portal_Text.txt',PACKAGE/'FINAL_SUBMISSION_README.md',REVIEW/'PORTAL_TEXT_CHECK.json']
    for manifest in OUT.glob('*/dataset_manifest.json'):
        for source in json.loads(manifest.read_text())['sources']:files.append(ROOT/source['file'])
    files=[p for p in sorted(set(files)) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.download','.tmp','.log'] and not p.name.startswith('weakness_layout_preview') and not p.name.endswith('_preview.png')]
    files=[p for p in files if p!=OUT/'publication_status.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)).replace('\\','/'))
        z.writestr('SHA256.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(archive) as z:check('archive_crc',z.testzip() is None);check('archive_inventory',len(z.namelist())==len(files)+1)
    value={'status':'passed','check_count':len(checks),'checks':checks,'manuscript':meta['manuscript'],'pages':pages,'sha256':sha(path),'evidence_archive':str(archive.relative_to(ROOT)),'archive_sha256':sha(archive),'scope':'Internal source/inference/document/archive audit; external proof review, field validation and competition outcome not certified'}
    (REVIEW/'EXPANDED_FINAL_AUDIT.json').write_text(json.dumps(value,indent=2));meta['status']='completed_and_audited';meta['document_audit_checks']=len(checks);meta['pages']=pages;(OUT/'publication_status.json').write_text(json.dumps(meta,indent=2))
    guide=f'''# Manuscript to review\n\nUse [SheafCode_Full_Empirical_Manuscript.pdf](../Upload_Files/SheafCode_Full_Empirical_Manuscript.pdf): {pages} pages, one manuscript with the complete recent-data theory and expanded empirical evidence.\n\n{meta['new_external_cohorts']} added external regional cohorts; {meta['new_external_station_days']:,} new paired outage station-days. Original benchmark results remain included, and longer Indian sensitivity is explicitly non-independent.\n\nRead [EXPANDED_EVIDENCE_REVIEW.md](EXPANDED_EVIDENCE_REVIEW.md) for all favorable and unfavorable results and remaining limitations. [EXPANDED_FINAL_AUDIT.json](EXPANDED_FINAL_AUDIT.json) records document checks; each cohort has its own raw-source and independent-inference audit.\n\nNo additional long run is required for this expansion. Run_Broad_Empirical_Validation.cmd resumes reproducibly if needed; it does not make unfinished field/priority questions disappear.\n\nThe oldest included record is 14 October 2025. Recheck observation-age eligibility if actual submission is after 13 October 2026. Historical benchmark PDFs remain preserved for reference; the manuscript here does not treat historical observations as recent.\n'''
    guide+='\nThe isolated follow-up is included: causal forest, three external GRIN cohorts with all seeds, risk intervals, matched-FPR alert analysis, validation-only model choice and temporal fault detection. Read [WEAKNESS_RESOLUTION_REVIEW.md](WEAKNESS_RESOLUTION_REVIEW.md) for the results and limits. Run_Weakness_Resolution.cmd resumes its checkpoints and rebuilds the current submission.\n'
    (REVIEW/'EXPANDED_START_HERE.md').write_text(guide,encoding='utf-8');(PACKAGE/'README.md').write_text(guide.replace('../Upload_Files/','Upload_Files/').replace('(EXPANDED_','(Review_Notes/EXPANDED_'),encoding='utf-8')
    print('Document audit passed',len(checks),'checks;',pages,'pages; archive',len(files),'files',flush=True)
if __name__=='__main__':main()
