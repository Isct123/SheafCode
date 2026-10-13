from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Final_Manuscript'
for path in list(OUT.glob('*.tex')) + list((OUT/'sections').glob('*.tex')):
    text = path.read_text(encoding='utf-8')
    replacements = {
        'Prospectively designated': 'Pre-ranked',
        'Prospectively assigned': 'Pre-ranked',
        'prospectively assigned': 'pre-ranked',
        'Prospective risk groups': 'Pre-ranked risk groups',
        'Prospective risk grouping': 'Pre-ranked risk grouping',
        'prospective risk-group': 'pre-ranked risk-group',
        'prospective risk-group comparisons': 'pre-ranked risk-group comparisons',
        'Prospective decile tests': 'Pre-ranked decile tests',
        'reported prospective ranking': 'reported pre-scoring ranking',
        'prospective held-out validation': 'pre-ranked held-out evaluation',
        'leakage-free artificial outages, strong statistical and machine-learning baselines': 'masked artificial outages, implemented statistical and machine-learning baselines',
        'competitively with strong statistical and graph-based baselines': 'competitively with the implemented statistical and graph-based baselines',
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    if path.name == '09_results.tex':
        before = 'Before the main test-error loop, the code writes the frozen model\'s $C2^*$ ordering and top/bottom decile memberships.'
        text = text.replace(before, before + ' This establishes ordering before scoring in the recorded execution, not a time-stamped preregistration or proof that the held-out period was never inspected. The comparison is therefore described as a pre-ranked held-out test; a genuinely untouched later window would be needed for prospective confirmation.')
        text = text.replace('Raw replication 2 is descriptive. This figure does not combine outage sizes.', 'The right column gives the relative mean gain over Gaussian and the within-network Holm-adjusted paired-test value; confidence-interval overlap is not a significance test. Raw replication 2 is descriptive. This figure does not combine outage sizes.')
        text = text.replace('targets within one station are not fully independent.', 'the right column gives the gain over Gaussian and the adjusted paired-test value. Targets within one station are not fully independent.')
        text = text.replace('The sheaf and Kalman methods can use later surviving-station observations inside each full window.', 'The right column gives relative mean gain over Kalman and the supplementary run-based adjusted test value. The sheaf and Kalman methods can use later surviving-station observations inside each full window.')
        text = text.replace('Group membership may differ in outage size and geometry,', 'The right column gives the high/low error ratio and the across-network Holm-adjusted test value. The companion detail panel expands the non-national comparisons while retaining the full national interval in the overview. Group membership may differ in outage size and geometry,')
    if path.name == '10_discussion.tex':
        text = text.replace('Pre-ranked risk grouping is useful because test errors do not determine the ranking.', 'Pre-ranked risk grouping is useful because the recorded scoring loop does not determine the ranking. This code ordering does not establish an untouched prospective validation.')
    if path.name == '11_limitations.tex':
        text = text.replace('Static artificial-outage certificates omit changing natural-gap masks.', 'The recorded ranking precedes scoring, but its execution order is not evidence of an untouched, formally preregistered holdout. Static artificial-outage certificates omit changing natural-gap masks.')
    if path.name == 'portal_fields.tex':
        text = text.replace('Its conclusions apply to the implemented models and selected public monitoring datasets.', 'Its conclusions apply to the implemented models and selected public monitoring datasets. The intended research use is to compare reconstructions and diagnose vulnerable outage configurations before considering operational deployment.')
        text = text.replace('temporal training offsets need not equal elapsed hours.', 'temporal training offsets need not equal elapsed hours. Risk groups are ranked from the frozen training operator before scoring; this execution order does not establish an untouched prospective validation.')
    if path.name == 'synopsis.tex':
        text = text.replace('strong distribution shift may dominate structural conditioning.', 'distributional differences may weaken the association between conditioning and error.')
        text = text.replace('and limited external source verification constrain inference and generalization.', 'and limited external source verification constrain inference and generalization. Pre-scoring group assignment is not proof of an untouched prospective holdout.')
    path.write_text(text, encoding='utf-8')

aud = json.loads((OUT/'audit/numerical_audit.json').read_text())['results']
source = ROOT/'evidence/sheafcode_iris_2026_final/sheafcode_iris_2026_final/results/FINAL_RESULTS.json'
record = json.loads(source.read_text())['regions']
rows=[]
for region, values in aud.items():
    for plot, key, ref, comparator, family in [
        ('snapshot','single_snapshot','vector_sheaf','gaussian_cond','P1_singles'),
        ('channel','channel','vector_sheaf','gaussian_cond','P2'),
        ('prolonged','prolonged','st_sheaf','kalman_var','P3_1_24h'),
    ]:
        m = values[key]
        gain = 100*(1-m[ref]['mean']/m[comparator]['mean'])
        test = record[region][family].get(comparator,{})
        p = values['cluster_sensitivity'][comparator]['p_holm'] if plot=='prolonged' else test.get('p_holm',float('nan'))
        rows.append(dict(plot=plot,network=values['label'],n=m[ref]['n'],runs=values['n_runs'] if plot=='prolonged' else 0,gain_percent=gain,p_holm=p,test_unit='run' if plot=='prolonged' else 'target'))
    if values['decile']:
        d=values['decile']
        rows.append(dict(plot='certificates',network=values['label'],n=d['n_safe'],n_high=d['n_risky'],runs=0,gain_percent=float('nan'),ratio=d['ratio'],p_holm=d['p_holm_across_networks'],test_unit='pattern'))
pd.DataFrame(rows).to_csv(OUT/'data/figure_annotations.csv',index=False)

def words(text):
    text=re.sub(r'\\url\{([^}]+)\}',r'\1',text)
    text=re.sub(r'\\[A-Za-z]+\*?(?:\[[^]]*\])?', '', text)
    return len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b",text))

portal=(OUT/'portal_fields.tex').read_text()
counts={}
for match in re.finditer(r'\\section\*\{([^}]+)\}\s*(.*?)(?=\\section\*|\\end\{document\})',portal,re.S):
    counts[match[1].replace(r'\&','&')]=words(match[2])
video=(OUT/'video_script.tex').read_text().split(r'\section*{Spoken reference script}')[1].split(r'\section*{Visual storyboard}')[0]
counts['spoken_script']=words(video)
(OUT/'audit/word_counts.json').write_text(json.dumps(counts,indent=2)+'\n',encoding='utf-8')
print(json.dumps(counts,indent=2))
