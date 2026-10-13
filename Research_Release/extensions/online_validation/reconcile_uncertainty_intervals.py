                                                                                   
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from uncertainty_shift import cluster_ci

def main():
    original=OUT/'uncertainty_shift_summary.csv';summary=pd.read_csv(original)
                                                                           
                                                                          
    for (region,rate),g in summary[summary.weeks<5].groupby(['region','extra_missing_rate']):
        cases=pd.read_csv(OUT/f'uncertainty_{region}_{rate:g}_cases.csv')
        for index,row in g.iterrows():
            q=cases[cases.method==row.method]
            if row.group_type=='quarter':q=q[q.calendar_quarter==row['group']]
            if row.group_type=='train_extreme':q=q[q.train_defined_extreme==(row['group']=='True')]
            ci=cluster_ci(q);summary.loc[index,['coverage_ci_low','coverage_ci_high']]=ci
    summary.to_csv(original,index=False)
    corrected=summary.copy();corrected.loc[corrected.weeks<5,['coverage_ci_low','coverage_ci_high']]=np.nan
    corrected.to_csv(OUT/'uncertainty_shift_corrected_summary.csv',index=False)
    save('uncertainty_interval_reconciliation.json',{'source_sha256':sha(__file__),'original_summary_sha256':sha(original),
        'corrected_summary_sha256':sha(OUT/'uncertainty_shift_corrected_summary.csv'),
        'suppressed_groups':int((corrected.weeks<5).sum()),
        'scope':'Display correction only: no uncertainty interval reported for fewer than five weekly clusters. All case outcomes, point coverage, widths and thresholds unchanged; sparse groups remain reported with sample counts.'})

if __name__=='__main__':main()
