                                                                                       
from pathlib import Path
import csv,json,hashlib,re,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'audit_tools/plot_config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Research_Release/operational_validation_results'
WORK=ROOT/'SUBMISSION_WORKSPACE'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    study=json.loads((OUT/'SUMMARY.json').read_text());replay=json.loads((OUT/'INDEPENDENT_REPLAY.json').read_text())
    if study['status']!='completed_and_checked' or replay['status']!='passed':raise RuntimeError('Study/replay incomplete')
    banks=json.loads((OUT/'FROZEN_BANKS.json').read_text());roles=[c for c in study['cohorts'] if c['status']=='complete']
    rows=list(csv.DictReader((OUT/'TRIALS.csv').open(newline='',encoding='utf-8')))
    aggregate={}
    for faults in ['0','1']:
        subset=[r for r in rows if r['faults']==faults];common=[r for r in subset if r['direct_prediction']!='']
        a={}
        for kind in ['point','box','direct']:
            issued=[r for r in subset if r[kind+'_prediction']!=''];wrong=[r for r in issued if r[kind+'_prediction']!=r['truth']]
            positives=[r for r in subset if r['truth']=='1']
            a[kind]={'trials':len(subset),'issued':len(issued),'wrong':len(wrong),'abstained':len(subset)-len(issued),
                     'issued_fraction':len(issued)/len(subset),'precision':1-len(wrong)/len(issued) if issued else None,
                     'positive_trials':len(positives),'sensitivity':sum(r[kind+'_prediction']=='1' for r in positives)/len(positives) if positives else None,
                     'wrong_outside_model':sum(r.get(kind+'_member')=='0' for r in wrong) if kind!='direct' else 0}
        full=sum(int(r['full_queries']) for r in subset);direct=sum(int(r['direct_queries']) for r in subset);simple=sum(int(r['training_order_queries']) for r in subset)
        a.update({'full_queries':full,'direct_queries':direct,'training_order_queries':simple,'query_saving_all':1-direct/full,
                  'query_saving_common_certified':1-sum(int(r['direct_queries']) for r in common)/sum(int(r['full_queries']) for r in common),
                  'extra_queries_vs_training_order':direct-simple})
        aggregate[faults]=a
    hours=sum(c['complete_hours']['2'] for c in roles);covered=sum(c['test_box_members'] for c in roles)
    rows_plans=[p for b in banks.values() for ps in b['plans'].values() for p in ps.values()]
    feasible=[p for p in rows_plans if p['feasible']]
    exact=sum(len(p['stations']) for p in feasible);greedy=sum(p['greedy_queries'] for p in feasible)
    stats={'status':'completed_and_replayed','attempted_roles':len(study['cohorts']),'evaluable_roles':len(roles),'test_hour_role_memberships':hours,
           'box_member_hours':covered,'box_coverage_pooled':covered/hours,'box_coverage_min':min(c['test_box_members']/c['complete_hours']['2'] for c in roles),
           'box_coverage_max':max(c['test_box_members']/c['complete_hours']['2'] for c in roles),'fault_budgets':aggregate,
           'planning':{'attempted':len(rows_plans),'feasible':len(feasible),'exact_query_sum':exact,'greedy_query_sum':greedy,
                       'max_exact_seconds':max(p['seconds'] for p in rows_plans),'max_greedy_seconds':max(p['greedy_seconds'] for p in rows_plans),
                       'station_counts':sorted({len(b['stations']) for b in banks.values()})},
           'scope':'Role/hour and noise/corruption trials are dependent; this is exploratory retrospective evaluation, not independent field exposure or cost evidence.'}
    (OUT/'AGGREGATES.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    with (OUT/'COHORT_TRANSFER.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['cohort','test_hours','box_member_hours','box_coverage','calibration_hours','calibration_box_coverage','query_saving_no_corruption','query_saving_one_corruption']);w.writeheader()
        for c in roles:
            role_rows=[r for r in rows if r['cohort']==c['cohort']]
            record={'cohort':c['cohort'],'test_hours':c['complete_hours']['2'],'box_member_hours':c['test_box_members'],'box_coverage':c['test_box_members']/c['complete_hours']['2'],
                    'calibration_hours':c['complete_hours']['3'],'calibration_box_coverage':c['calibration_box_members']/c['complete_hours']['3']}
            for f,label in [('0','no_corruption'),('1','one_corruption')]:
                trial=[r for r in role_rows if r['faults']==f];record['query_saving_'+label]=1-sum(int(r['direct_queries']) for r in trial)/sum(int(r['full_queries']) for r in trial)
            w.writerow(record)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    coverage=[100*c['test_box_members']/c['complete_hours']['2'] for c in roles]
    axes[0].bar(range(len(roles)),coverage,color='#456e91');axes[0].axhline(90,color='gray',linestyle='--',label='Calibration rank target; not a guarantee')
    axes[0].set_xticks(range(len(roles)),[c['cohort'] for c in roles],rotation=80,fontsize=7);axes[0].set_ylim(0,100);axes[0].set_ylabel('Observed box membership (%)');axes[0].set_title('Later-date model coverage, all 18 roles');axes[0].legend(fontsize=7)
    x=np.arange(2);axes[1].bar(x-.17,[100*aggregate[f]['direct']['issued_fraction'] for f in ['0','1']],width=.34,label='Direct certificate issued')
    axes[1].bar(x+.17,[100*aggregate[f]['query_saving_all'] for f in ['0','1']],width=.34,label='Query saving vs full measurement')
    axes[1].set_xticks(x,['No corruption','One corruption']);axes[1].set_ylim(0,100);axes[1].set_ylabel('Stress-trial percentage (%)');axes[1].set_title('Direct guard: zero wrong issued certificates');axes[1].legend(fontsize=8)
    fig.text(.5,.01,'Dependent retrospective trials; conditional on the specified reading-error/fault budgets. No field costs or emissions measured.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(OUT/'HELDOUT_TRANSFER.pdf');plt.close(fig)
    a0=aggregate['0'];a1=aggregate['1']
    report=f'''# Operational follow-up: results and boundaries

Completed 8 October 2026. No new theorem or prediction-model refit was needed. The chronological repair follow-up uses training scenarios, a separate calibration block and later test dates. It is exploratory: those archives had already been accessed for other analyses.

## What was evaluated

All 23 existing cohort roles were attempted; 18 have sufficient complete training/calibration/test PM2.5 vectors. The other five are excluded explicitly, not silently. All {hours:,} eligible test hour-role memberships are retained. Roles include a longer Indian sensitivity and original networks with possible overlap; neither roles nor stress trials are independent samples. Naturally missing target truth remains excluded and counted.

At most 24 training-only vectors define each bank. Calibration-only nearest-centre distances set a 90%-rank radius, then each centre's radius is capped to preserve its threshold label. The concentration reference is an illustrative hourly network maximum above 35 ug/m3. Noise is bounded by 1 ug/m3. No-corruption cases test negative, zero and positive clean errors; one-corruption cases test each station in turn with both low/high corrupted readings and both endpoint error directions.

## The domain gap is measured, not hidden

Box membership covers {covered:,}/{hours:,} test hour-role memberships ({100*covered/hours:.2f}%), ranging from {100*stats['box_coverage_min']:.2f}% to {100*stats['box_coverage_max']:.2f}% by role. The calibrated quantile therefore is not a field-coverage guarantee. Label-preserving caps and temporal shift matter.

Without corruption, box-only inference issues {a0['box']['issued']:,} decisions and gets {a0['box']['wrong']:,} wrong. With one corrupted query, it issues {a1['box']['issued']:,} and gets {a1['box']['wrong']:,} wrong. Every wrong point/box decision is outside its model membership; no in-model guarantee failed. A scenario-only certificate must not be treated as unconditional operational reliability.

## Direct-reading guard and matched controls

The direct policy stops at exceedance only after more than f queried readings have clean lower bounds above the reference. With no corruption it can certify safe only after every station's upper bound is below or at the reference. With one arbitrary corrupted reading, a safe label cannot generally be certified without additional physical bounds, so it abstains. This is a simple interval/counting control, not an added theorem.

| Query faults | Stress trials | Direct decisions issued | Wrong issued | Queries used / full-network queries | Saving across all trials |
|---|---:|---:|---:|---:|---:|
| 0 | {a0['direct']['trials']:,} | {a0['direct']['issued']:,} | 0 | {a0['direct_queries']:,} / {a0['full_queries']:,} | {100*a0['query_saving_all']:.2f}% |
| 1 | {a1['direct']['trials']:,} | {a1['direct']['issued']:,} | 0 | {a1['direct_queries']:,} / {a1['full_queries']:,} | {100*a1['query_saving_all']:.2f}% |

The sequential guard and full-network direct certificate give exactly the same decision or abstention on each trial. Zero wrong issued certificates is conditional on the declared error/fault model; it is not an estimate of real-world infallibility. Under one corruption, certification is only {100*a1['direct']['issued_fraction']:.2f}% of trials, so the large abstention fraction must accompany its correctness claim.

The simpler training-frequency order uses {a0['training_order_queries']:,} and {a1['training_order_queries']:,} queries respectively. Prioritizing the box-repair plan costs {a0['extra_queries_vs_training_order']} and {a1['extra_queries_vs_training_order']} extra queries. There is no demonstrated operational query-cost advantage from that prioritization. The new evidence supports an elementary conservative control, not universal learned-model dominance.

## Planning and verification

Actual fitted banks have {stats['planning']['station_counts']} stations. {len(feasible)}/{len(rows_plans)} point/box/fault planning settings are feasible, including trivial one-label banks. Exact plans total {exact} queries and greedy plans {greedy} on those same settings. Measured maximum exact and greedy planning times are {stats['planning']['max_exact_seconds']:.6f}s and {stats['planning']['max_greedy_seconds']:.6f}s on this machine; small-bank measurements do not establish large-network scalability.

The finalized run passed {study['checks']:,} checks. Independent replay passed {replay['checks']:,} checks, reconstructing every direct certificate and query cost, verifying bank/calibration choices and checking {replay['sampled_exact_model_checks']:,} model inferences in exact arithmetic. Four focused boundary/corruption tests pass. These checks are verification tasks, not additional scientific trials or formal proof certification.

Inputs, protocol and code are hashed in PROTOCOL_LOCK.json. Full outputs are TRIALS.csv, HOUR_MEMBERSHIP.csv, FROZEN_BANKS.json, COHORT_TRANSFER.csv, AGGREGATES.json and INDEPENDENT_REPLAY.json. The original reconstruction evidence and frozen manuscript remain unchanged. The submission working copy receives this evidence as an explicitly exploratory addition.

## What is fixed and what remains

The prior unmeasured model-transfer gap now has later-date membership/error measurements. A model-free guard provides a conservative alternative at fewer simulated queries than full measurement. Positive monitoring relevance is computational; field savings, real sensor-error distributions, emissions reduction and health effects remain unmeasured. No script can supply those facts from archived observations alone. Student ownership still requires an accurate personal contribution record; it cannot be certified by an automated audit.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8');(WORK/'Guides/Operational_Validation.md').write_text(report,encoding='utf-8')
    import shutil
    shutil.copy2(OUT/'HELDOUT_TRANSFER.pdf',WORK/'Guides/Operational_Validation_Figure.pdf')
    snippet=rf'''
\subsection{{Chronological repair transfer and a direct-reading guard}}
An exploratory follow-up attempts all 23 existing cohort roles, including the original networks and the longer Indian sensitivity. Eighteen provide enough complete PM2.5 training/calibration/test vectors; five lack complete later vectors. All {hours:,} eligible test hour-role memberships are evaluated. These are dependent source views, not eighteen independent replications. The scenario bank uses at most 24 equally spaced training vectors. A separate calibration block supplies the 90\%-rank nearest-centre distance; each box radius is capped at 95\% of its centre-to-threshold margin to preserve the decision label. Test outcomes choose neither bank nor radius. The reference is an illustrative hourly network maximum above 35 $\mu$g/m$^3$, with clean-reading error at most 1 $\mu$g/m$^3$; this is not official AQI or GRAP.

\textbf{{Transfer limitation measured.}} Only {covered:,}/{hours:,} later hour-role vectors ({100*covered/hours:.2f}\%) belong to the boxes, with role coverage {100*stats['box_coverage_min']:.2f}--{100*stats['box_coverage_max']:.2f}\%. Without corruption, box-only inference gets {a0['box']['wrong']:,} of {a0['box']['issued']:,} issued decisions wrong; with one corrupted query, {a1['box']['wrong']:,} of {a1['box']['issued']:,}. Every such error is outside model membership. The theorem remains conditional; the calibration quantile does not establish future model coverage.

\textbf{{An elementary operational control.}} Query the box plan first when feasible, then remaining stations in training-only exceedance-frequency order. A model-free guard certifies exceedance after more than $f$ queried readings have lower clean-error bounds above the reference: at least one must be clean. With $f=0$, safe requires every station queried and every upper bound at or below the reference. With one arbitrary corrupted reading and no physical concentration bounds, safe is generally unidentifiable, so the guard abstains. This counting/interval control adds no new theorem. Noise trials use the negative, zero and positive endpoints; one-corruption stress trials vary every faulty station, both high/low corrupt values and both clean-error signs. They reuse hours and are not independent fault observations.

\begin{{table}}[htbp]\centering\small
\caption{{Later-date query simulation. Sequential and full-network direct certificates return identical decisions or abstentions on matched trials. Correctness is conditional on the simulated error/fault budgets.}}
\begin{{tabular}}{{rrrrr}}\toprule
Fault budget & Trials & Issued / wrong & Query saving & Abstentions\\\midrule
0 & {a0['direct']['trials']:,} & {a0['direct']['issued']:,} / 0 & {100*a0['query_saving_all']:.2f}\% & {a0['direct']['abstained']:,}\\
1 & {a1['direct']['trials']:,} & {a1['direct']['issued']:,} / 0 & {100*a1['query_saving_all']:.2f}\% & {a1['direct']['abstained']:,}\\
\bottomrule\end{{tabular}}
\par\smallskip Query saving uses all trials, including abstentions, against measuring every station under the same responses. It measures query counts, not field expenditure.
\end{{table}}

With one corruption, certificates are issued on only {100*a1['direct']['issued_fraction']:.2f}\% of trials. A simpler training-frequency order is slightly cheaper: box-plan prioritization adds {a0['extra_queries_vs_training_order']} and {a1['extra_queries_vs_training_order']} queries across the two budgets. No operational query-cost superiority over that simple order is established. On the actual small banks, exact/greedy multicover totals are {exact}/{greedy} across {len(feasible)} feasible point/box/fault settings, including trivial banks; this does not resolve exponential scaling. Independent replay reconstructs all {len(rows):,} direct trials and checks {replay['sampled_exact_model_checks']:,} model inferences in exact arithmetic. The finding is retrospective monitoring reliability and simulated query reduction, not prospective field validation or reduced emissions.

'''
    source=WORK/'Source/manuscript.tex';text=source.read_text(encoding='utf-8')
    frozen=json.loads((ROOT/'IRIS_Submission_Recent_Data/Review_Notes/CONCISE_MANUSCRIPT_AUDIT.json').read_text(encoding='utf-8-sig'))
    if sha(source)!=frozen['source_sha256']:raise RuntimeError('Working manuscript has personal edits; report was saved but automatic insertion refused to protect them.')
    marker=r'\section{Reliability, limitations and conclusion}'
    if text.count(marker)!=1:raise RuntimeError('Insertion marker missing/ambiguous')
    abstract_match=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',text,re.S);old_abstract=abstract_match.group(1).strip()
    extra=rf' Without query corruption, a later-date direct-reading simulation saves {100*a0["query_saving_all"]:.2f}\% of queries against full measurement, with no wrong issued certificates under the tested error budget.'
    new_abstract=old_abstract+extra
    if len(new_abstract.replace(r'\%','%').split())>250:raise RuntimeError('Abstract exceeds 250 words')
    text=text.replace(old_abstract,new_abstract,1).replace(marker,snippet+marker,1)
    source.write_text(text,encoding='utf-8')
    plain_old=old_abstract.replace(r'\%','%').replace('--','\u2013');plain_new=new_abstract.replace(r'\%','%').replace('--','\u2013')
    (WORK/'Abstract.txt').write_text(plain_new+'\n',encoding='utf-8')
    portal=WORK/'Portal_Text.txt';portal_text=portal.read_text(encoding='utf-8-sig')
    if plain_old not in portal_text:raise RuntimeError('Portal abstract is not the original draft; synchronization needs manual review')
    portal.write_text(portal_text.replace(plain_old,plain_new,1),encoding='utf-8')
    synopsis=WORK/'Source/synopsis.tex';synopsis_text=synopsis.read_text(encoding='utf-8')
    synopsis_extra=rf' An exploratory training/calibration-fitted repair follow-up evaluates {hours:,} later complete hour-role vectors. Box membership is {100*covered/hours:.2f}\%, exposing wrong scenario-only decisions outside the model. A direct-reading control retains abstentions and gives no wrong issued certificates in the specified noise/corruption trials; without corruption it saves {100*a0["query_saving_all"]:.2f}\% of queries against full measurement. The simple training-frequency order is slightly cheaper than prioritizing the box plan. No field savings were measured.'
    synopsis_text=synopsis_text.replace(r'\section*{Acknowledgements and reference links}',synopsis_extra+'\n'+r'\section*{Acknowledgements and reference links}',1)
    synopsis_text=synopsis_text.replace(r'\textit{SheafCode Concise Manuscript}',r'\textit{SheafCode Manuscript}')
    synopsis.write_text(synopsis_text,encoding='utf-8')
    (OUT/'MANUSCRIPT_ADDITION.tex').write_text(snippet,encoding='utf-8')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
