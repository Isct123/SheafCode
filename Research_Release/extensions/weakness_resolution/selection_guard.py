                                                                                
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'Research_Release/expanded_validation_results'
OUT=ROOT/'Research_Release/weakness_resolution_results/selection_guard'
RF=OUT.parent/'nonlinear'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
import pandas as pd

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for folder in sorted(p.parent for p in RF.glob('*/status.json')):
        status=json.loads((folder/'status.json').read_text())
        if status['status']!='completed':continue
        name=folder.name;choices=json.loads((SOURCE/name/'model_choices.json').read_text())
        if choices['validation_cases']!=status['selection_station_days']:
            raise AssertionError('Selection case sets require explicit reconciliation')
        scores={'sheaf_causal':choices['sheaf']['validation_mse'],'var_causal':choices['var']['validation_mse'],
                'random_forest_causal':status['selection_mse']}
        chosen=min(scores,key=lambda key:(scores[key],key))
        ref=pd.read_csv(SOURCE/name/'outage_cases.csv',dtype={'station':str})
        candidate=pd.read_csv(folder/'cases.csv',dtype={'station':str}) if chosen=='random_forest_causal' else ref[ref.method.eq(chosen)].copy()
        candidate=candidate.copy();candidate['method']='validation_model_selector'
        candidate.to_csv(OUT/(name+'_cases.csv'),index=False)
        result={'cohort':name,'chosen_method':chosen,'validation_scores':scores,
                'case_count':len(candidate),'test_used_for_choice':False,
                'summary':common.paired(pd.concat([candidate,ref]),'validation_model_selector','var_causal'),
                'physical_summary':common.paired(pd.concat([candidate,ref]),'validation_model_selector','var_causal',value='pm25_raw_mse'),
                'scope':'Exploratory validation-only selection using common selection-case definition; does not guarantee superiority under test shift'}
        rows.append(result)
    (OUT/'SUMMARY.json').write_text(json.dumps(common.clean(rows),indent=2))
    print('Validation-only selector evaluated',len(rows),'cohorts',flush=True)

if __name__=='__main__':main()
