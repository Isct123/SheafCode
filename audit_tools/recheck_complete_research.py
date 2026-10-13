                                                                               

                                                                
   
import ast,hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'Research_Release'
OUT=BASE/'Final_Manuscript/Complete_Research/audit'
OUT.mkdir(parents=True,exist_ok=True)
cache={};checks=[];issues=[]


def digest(p):
    p=Path(p)
    if p not in cache:cache[p]=hashlib.sha256(p.read_bytes()).hexdigest()
    return cache[p]


def check(name,condition,detail=''):
    if not bool(condition):raise AssertionError(name+': '+str(detail))
    checks.append({'check':name,'passed':True,'detail':str(detail)})


def main():
    for folder,source in [('validation_results','extensions/validation'),('joint_decision_results','extensions/joint_decisions')]:
        a=json.loads((ROOT/folder/'audit.json').read_text())
        check(folder+'/status',a['status']=='complete')
        for path,sha in a['source_sha256'].items():check(folder+'/source/'+path,digest(ROOT/source/path)==sha)
        for path,sha in a.get('completed_result_sha256',a.get('result_sha256',{})).items():check(folder+'/result/'+path,digest(ROOT/folder/path)==sha)
        for path,sha in a.get('input_sha256',{}).items():check(folder+'/input/'+path,digest(ROOT/path)==sha)
    regions=['delhi','ncr','mumbai','kolkata','hyderabad','bengaluru','india','delhi2025','bengaluru_raw']
    reviewed=pd.read_csv(ROOT/'expanded_docs/reviewed_comparisons.csv')
    for region in regions:
        folder=ROOT/'results_expanded'/region
        m=json.loads((folder/'manifest.json').read_text())
        check(region+'/completion',m['status'] in ['complete','excluded'])
        for path,sha in m['code_sha256'].items():check(region+'/source/'+path,digest(ROOT/path)==sha)
        for path,sha in m['data_sha256'].items():check(region+'/data/'+path,digest(ROOT/path)==sha)
        if m['status']=='excluded':continue
        frame=pd.read_csv(folder/'prolonged_with_gaussian.csv')
        check(region+'/unique_cases',not frame.duplicated(['day','station','mode','method']).any())
        for mode,g in frame.groupby('mode'):
            wide=g.pivot(index=['day','station'],columns='method',values='mse')
            for ref in ['stable_var','gaussian','spatial_sheaf','seasonal_persistence']:
                pair=wide[['temporal_sheaf',ref]].dropna()
                gain=1-np.sqrt(pair.temporal_sheaf.mean()/pair[ref].mean())
                row=reviewed[(reviewed.region==region)&(reviewed['mode']==mode)&(reviewed.reference==ref)].iloc[0]
                check(f'{region}/{mode}/{ref}/count',len(pair)==row.cases)
                check(f'{region}/{mode}/{ref}/gain',abs(gain-row.rmse_gain)<1e-12)
    for region in ['delhi2025','bengaluru_raw']:
        f=pd.read_csv(ROOT/f'extension_results/strong_baselines/{region}_cases.csv')
        saved=json.loads((ROOT/'extension_results/strong_baselines/summary.json').read_text())['comparisons'][region]
        for mode,g in f.groupby('mode'):
            w=g.pivot(index=['day','station'],columns='method',values='mse')[['temporal_sheaf','full_covariance_var']].dropna()
            check(f'{region}/{mode}/strong_gain',abs(1-np.sqrt(w.iloc[:,0].mean()/w.iloc[:,1].mean())-saved[mode]['rmse_relative_gain'])<1e-12)
    faults=pd.read_csv(ROOT/'validation_results/fault_cases.csv')
    summary=pd.read_csv(ROOT/'validation_results/fault_summary.csv')
    groupkeys=['region','fault_count','direction','magnitude','method']
    for keys,g in faults.groupby(groupkeys):
        r=summary
        for k,v in zip(groupkeys,keys):r=r[r[k]==v]
        check('fault/single_summary/'+str(keys),len(r)==1)
        r=r.iloc[0]
        check('fault/cases/'+str(keys),len(g)==r.cases)
        for field,col in [('alarm_rate','alarm'),('exact_end_to_end','exact'),('count_accuracy','count_correct')]:
            check('fault/'+field+'/'+str(keys),abs(g[col].mean()-r[field])<1e-12)
        check('fault/exact_requires_alarm/'+str(keys),not ((g.fault_count>0)&g.exact&~g.alarm).any())
    bounded=pd.read_csv(ROOT/'constrained_results/cases.csv')
    check('conditional_containment',bounded.loc[bounded.assumptions_met,'covered'].all())
    count=json.loads((ROOT/'constrained_results/manifest.json').read_text())['conditional_containment_checks']
    check('conditional_count',int(bounded.assumptions_met.sum())==count)
                                                                                 
    sensitivity=pd.read_csv(ROOT/'joint_decision_results/comparison_sensitivity.csv')
    check('sensitivity_family',len(sensitivity)==30)
    check('sensitivity_nested_quantiles',((sensitivity.family_sensitivity_low<=sensitivity.nominal_95_low)&(sensitivity.family_sensitivity_high>=sensitivity.nominal_95_high)).all())
    jc=pd.read_csv(ROOT/'joint_decision_results/joint_cases.csv')
    check('joint_lower_dominates_marginal',(jc.joint_lower>=jc.independent_lower-1e-6).all())
                                                                              
    structured=json.loads((ROOT/'extension_results/structured/summary.json').read_text())
    check('actual_map_rank_agreements',structured['rank_agreements']==5394)
    for region in structured['results']:
        a=structured['results'][region]
        check(region+'/zero_actual_kernel',a['kernel_dimension']==0)
        check(region+'/infinite_distance',a['block_distance']=='infinity')
                                                                                
    for directory in [ROOT/'sheafcode',ROOT/'scripts',ROOT/'extensions']:
        for p in directory.rglob('*.py'):
            if 'vendor' in p.parts:continue
            ast.parse(p.read_text(encoding='utf-8-sig'));check('syntax/'+str(p.relative_to(ROOT)),True)
    old=json.loads((BASE/'Final_Manuscript/audit/numerical_audit.json').read_text())
    check('supplied_numeric_audit/all_checks',all(x['passed'] for x in old['checks']))
                                                                            
                                                                         
    snapshot=pd.read_csv(BASE/'Final_Manuscript/data/snapshot_summary.csv')
    check('supplied_snapshot/table_available',len(snapshot)>0)
    for path in [ROOT/'validation_results/Supplemental_Research_Results.zip',ROOT/'joint_decision_results/Additional_Results.zip',ROOT/'Final_Research_Results.zip']:
        with zipfile.ZipFile(path) as z:check('zip/'+path.name,z.testzip() is None)
    inventory=[]
                                                                         
                                                                         
    derived={BASE/'Final_Manuscript/SheafCode_Complete_Manuscript.pdf',
             BASE/'Final_Manuscript/SheafCode_Complete_Research_Package.zip',
             BASE/'Final_Manuscript/READ_COMPLETE_MANUSCRIPT.md'}
    for directory in [BASE/'evidence',BASE/'Final_Manuscript',ROOT]:
        for p in directory.rglob('*'):
            if not p.is_file() or p in derived or 'vendor' in p.parts or '__pycache__' in p.parts or 'Complete_Research' in p.parts:continue
            if p.suffix.lower() not in ['.md','.tex','.pdf','.json','.csv','.py','.zip']:continue
            inventory.append({'path':str(p.relative_to(BASE)),'bytes':p.stat().st_size,'sha256':digest(p)})
    for p in BASE.glob('*.zip'):inventory.append({'path':p.name,'bytes':p.stat().st_size,'sha256':digest(p)})
    pd.DataFrame(inventory).drop_duplicates('path').to_csv(OUT/'SOURCE_INVENTORY.csv',index=False)
    result={'status':'passed','checks':checks,'total_checks':len(checks),'audited_files':len(cache),
            'scope':'Frozen-source and saved-result arithmetic audit, syntax, support/containment invariants and archive integrity. No complete independent retraining or replay of every historic trial.',
            'limitations':['The 89 non-neural/core extension tests and one actual published-architecture integration test were rerun separately.',
                          'Mathematical review is an internal proof/assumption review, not independent peer review or originality certification.',
                          'Earlier report prose contains stale pending statuses. The integrated manuscript uses completed artifact status, not those stale statements.',
                          'Synthetic and conditional guarantees do not establish physical sensor fault truth, emissions reduction or health benefit.']}
    (OUT/'RECHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Independent frozen-result recheck passed:',len(checks),'checks;',len(cache),'hashed files',flush=True)


if __name__=='__main__':main()
