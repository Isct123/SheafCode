                                                                                   
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Research_Release/replication_20261012_results'

def main():
    table=pd.read_csv(OUT/'ALL_COHORT_RESULTS.csv')
    completed=table[table.status.eq('completed')]
    positive=completed[completed.normalized_gain_vs_var.gt(0)].copy()
    positive=positive.sort_values(['scope','normalized_gain_vs_var'],ascending=[False,False])
    positive.to_csv(OUT/'POSITIVE_RESULTS.csv',index=False)
    lines=['# Positive sheaf-versus-VAR reconstruction results','',
        'Selected cohorts with lower normalized sheaf RMSE than causal VAR. '
        'The filter is normalized gain greater than zero, applied separately to the primary and secondary protocols. '
        'These are exploratory highlights selected from the completed batch. '
        'The complete comparison set is in [RESULTS.md](RESULTS.md) and [ALL_COHORT_RESULTS.csv](ALL_COHORT_RESULTS.csv).','',
        'EPA observations cover 14 October 2025 to 31 March 2026; UK and NSW observations cover '
        '1 January to 30 June 2026. All observations used are within the preceding twelve months as of '
        '12 October 2026. Each experiment simulates a 24-hour station outage using causal inputs.','',
        'Positive gains mean lower RMSE. Intervals are paired seven-day 95% bootstrap intervals, '
        'without adjustment for multiple comparisons. A positive point estimate can have an interval crossing zero. '
        'Physical PM2.5 gains are reported separately and may be negative even when normalized gain is positive.','']
    for scope,title in [('primary_every_third_day','Primary comparison: every third eligible test day'),
                        ('all_eligible_test_days','Secondary comparison: all eligible test days')]:
        subset=positive[positive.scope.eq(scope)].sort_values('normalized_gain_vs_var',ascending=False)
        total=len(completed[completed.scope.eq(scope)])
        above=int(subset.ci_low.gt(0).sum())
        lines += [f'## {title}','',
                  f'{len(subset)} of {total} evaluable cohorts have positive normalized gains; '
                  f'{above} have intervals entirely above zero. '
                  'The primary and secondary tables reuse observations and are separate summaries.','',
                  '| Cohort | Cases / dates | Normalized gain vs VAR (%) | 95% interval (%) | Physical PM2.5 gain (%) |',
                  '|---|---:|---:|---:|---:|']
        for _,row in subset.iterrows():
            interval=f'[{row.ci_low:.2f}, {row.ci_high:.2f}]' if pd.notna(row.ci_low) else 'Insufficient calendar bins'
            pm=f'{row.pm25_gain_vs_var:.2f}' if pd.notna(row.pm25_gain_vs_var) else 'Unavailable'
            lines.append(f'| {row.cohort} | {int(row.cases)} / {int(row.test_dates)} | '
                         f'{row.normalized_gain_vs_var:.2f} | {interval} | {pm} |')
        lines.append('')
    lines += ['[POSITIVE_RESULTS.csv](POSITIVE_RESULTS.csv) retains the exact numerical values and metadata '
              'for these selected rows, including the random forest comparisons.']
    (OUT/'POSITIVE_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
                                                                                             
    saved=pd.read_csv(OUT/'POSITIVE_RESULTS.csv')
    assert saved.normalized_gain_vs_var.gt(0).all()
    assert set(zip(saved.cohort,saved.scope))==set(zip(positive.cohort,positive.scope))
    print(saved.groupby('scope').size().to_string())
    print('Created POSITIVE_RESULTS.md and POSITIVE_RESULTS.csv.')

if __name__=='__main__':main()
