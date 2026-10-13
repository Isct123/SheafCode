                                                                                
from pathlib import Path
from collections import Counter
import re
import json
import difflib

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'audit_tools/final_iris_review'
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Compact_Readable_Final'
OLD = REVIEW / 'before_repetition_edit/SheafCode_IRIS_Final.tex'

def indexed(path):
    return {int(m[1]):m[2].strip() for m in re.finditer(
        r'^\[(\d+)\][ \t]*(.*?)\s*(?=^\[\d+\]|\Z)',
        path.read_text(encoding='utf-8'), re.M | re.S)}

old = OLD.read_text(encoding='utf-8')
new = old
before = indexed(REVIEW / 'natural_prose_replacements.txt')
after = indexed(REVIEW / 'repetition_replacements.txt')
numeric_changes = []
for k, value in after.items():
    assert new.count(before[k]) == 1, (k, before[k][:80])
    new = new.replace(before[k], value)
    nums = lambda s:Counter(re.findall(r'(?<![A-Za-z])\d+(?:[.,]\d+)*',s))
    removed = nums(before[k])-nums(value)
    added = nums(value)-nums(before[k])
    if removed or added:
        numeric_changes.append({'paragraph':k,'removed':dict(removed),'added':dict(added)})

captions = [
    "The fitted Indian graph compares transformed station vectors in edge spaces such as $C_a$. Positions are schematic, and the orange hidden station is illustrative rather than a recorded outage.",
    "Causal input includes available history and current surviving readings. Hidden target truth, future observations and naturally missing context are excluded.",
    "The segment $a+b=2$ avoids the box where both values are at most 0.75, although either can be zero. These concentrations are illustrative.",
    "B separates both pairs when $f=0$, while $f=1$ requires three separators per pair. Expanded reading intervals meet at the strict stability boundary. Both panels use mathematical examples.",
    "Native units are retained for physical concentrations. A dash marks an unused channel.",
    "Chronological period lengths include missing hourly positions.",
    "Selection chooses among these model settings.",
    "Positive gain means lower causal sheaf normalized RMSE against each matched control. India's 77 station-days support seven-day intervals, while the US comparison's 24 cases have insufficient bins. Networks are not pooled.",
    r"Gains (\%) and paired seven-day intervals cover every evaluable external cohort on matched station-days. Negative gains indicate higher sheaf error.",
    r"Normalized retrospective sheaf gains (\%) include every tested GRIN seed.",
    "The top panels show exact localization of injected faults. The bottom panels show reconstruction gains across outage lengths, history and context loss. These sensitivity tests are separate from the fixed benchmark and do not establish natural-fault recovery.",
    r"No-feedback PM2.5 snapshot intervals target 90\% coverage, with no additional context loss.",
    "Finite archive settings are dependent. Pattern-impossibility counts concern feasible decision settings, and query totals compare settings where both tasks are feasible.",
    "Points above the diagonal improve stability at the same minimum cost. All feasible settings are shown, with guarantees valid strictly inside each radius.",
    "Sequential and full-network guards give identical decisions and abstentions on matched later-date responses. Savings count queries, include abstentions and do not estimate field expenditure.",
    "Implementation checks are dependent computational validations rather than independent field trials.",
    "Earlier studies retain their own protocols and negative findings.",
]
matches = list(re.finditer(r'\\caption\{',new))
assert len(matches) == len(captions) == 17
for m, value in reversed(list(zip(matches,captions))):
    start = m.end(); end = start; depth = 1
    while depth:
        if new[end] == '{': depth += 1
        elif new[end] == '}': depth -= 1
        end += 1
    new = new[:start]+value+new[end-1:]

extra = [
    (r"The VAR and RF gains use normalized multichannel error, while PM2.5 gains use physical concentration error. Brackets give paired seven-day bootstrap intervals. A dash means there are too few calendar bins for an interval. EPA codes identify the state and county, and UK and NSW names identify the regions. We exclude EPA 06-073 and 06-111 because they lack usable test observations.",
     r"VAR and RF gains use normalized multichannel error, while PM2.5 uses physical error. Brackets give paired seven-day intervals, and a dash indicates insufficient bins. EPA codes identify state and county, while UK and NSW names identify regions. EPA 06-073 and 06-111 lack usable test observations and are excluded."),
    (r"A is the original Indian all-day comparison, D1 is EPA 04-013, E1 is AURN central, and F1 is NSW Illawarra. All gains use sheaf as the numerator. The seeds share the same truth observations, so they are not independent datasets.",
     r"A denotes the original Indian all-day comparison, D1 EPA 04-013, E1 AURN central, and F1 NSW Illawarra. Gains use sheaf as numerator. Seeds share truth observations and are not independent datasets."),
    (r"Macro coverage gives each cohort equal weight across the 20 cohorts overall and the 19 with extreme readings. Pooled coverage gives each cell equal weight. Width is the average of the cohort-level finite interval widths in $\mu$g/m$^3$. We exclude infinite intervals from this average. The final column gives the percentage of cells whose intervals have no finite bounds.",
     r"Macro coverage weights cohorts equally, with 20 overall and 19 containing extremes. Pooled coverage weights cells equally. Width averages cohort-level finite widths in $\mu$g/m$^3$, excluding infinite intervals. The last column gives the percentage of cells with infinite intervals."),
    (r"Sheaf & We choose the decoder weight $\lambda$ from 0.1 or 1 and the temporal scale from 0.5 or 2.",
     r"Sheaf & The decoder weight $\lambda$ is 0.1 or 1, and temporal scale is 0.5 or 2."),
    (r"VAR \citep{lutkepohl2005var} & We choose ridge from 1 or 100, noise scale from 0.05 or 0.5, and covariance shrinkage from 0, 0.2 or 1.",
     r"VAR \citep{lutkepohl2005var} & Ridge is 1 or 100, noise scale is 0.05 or 0.5, and covariance shrinkage is 0, 0.2 or 1."),
    (r"RF \citep{breiman2001} & We use 64 trees with a maximum depth of 12. We choose a minimum leaf size of 5 or 20 training examples and sample at most 80 training days.",
     r"RF \citep{breiman2001} & The forest has 64 trees with depth at most 12, minimum leaf size 5 or 20 examples, and at most 80 sampled training days."),
    (r"GRIN \citep{cini2022grin} & We test three random seeds and choose a hidden-layer width of 32 or 64. Training lasts at most 30 epochs and stops after seven epochs without selection improvement.",
     r"GRIN \citep{cini2022grin} & Three seeds use hidden-layer width 32 or 64. Training lasts at most 30 epochs and stops after seven without selection improvement."),
]
for a,b in extra:
    assert new.count(a) == 1,a[:80]
    new=new.replace(a,b)

                                                                           
                                                                        
wrapper = re.compile(r'\\par\\noindent\\begin\{minipage\}\{\\linewidth\}\s*'
                     r'(\\begin\{(?:proposition|theorem)\}.*?'
                     r'\\end\{(?:proposition|theorem)\})\s*'
                     r'\\end\{minipage\}\\par', re.S)
freed_blocks = []
def release(m):
    if any(label in m[1] for label in [r'\label{prop:search-bounds}',r'\label{thm:repair-radius}']):
        freed_blocks.append(re.search(r'\\label\{(.*?)\}',m[1])[1])
        return m[1]
    return m[0]
new=wrapper.sub(release,new)
assert freed_blocks == ['prop:search-bounds','thm:repair-radius']

for env in ['proposition','theorem','equation','thebibliography','tikzpicture']:
    pat=rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}'
    assert re.findall(pat,new,re.S)==re.findall(pat,old,re.S),env
assert new[new.index(r'\appendix'):]==old[old.index(r'\appendix'):]
assert re.findall(r'\\\[.*?\\\]',new,re.S)==re.findall(r'\\\[.*?\\\]',old,re.S)
assert re.findall(r'\\(?:sub)?section\*?\{.*?\}',new)==re.findall(r'\\(?:sub)?section\*?\{.*?\}',old)
for macro in ['label','citep','citet','ref','eqref']:
    assert Counter(re.findall(rf'\\{macro}\{{(.*?)\}}',new))==Counter(re.findall(rf'\\{macro}\{{(.*?)\}}',old)),macro
tables=lambda s:re.findall(r'\\begin\{table\}.*?\\end\{table\}',s,re.S)
assert len(tables(new))==len(tables(old))==10
for a,b in zip(tables(old),tables(new)):
    body=lambda s:re.search(r'\\begin\{tabularx?\}.*?\\end\{tabularx?\}',s,re.S)[0]
    assert Counter(re.findall(r'\d+(?:[.,]\d+)*',body(a)))==Counter(re.findall(r'\d+(?:[.,]\d+)*',body(b)))
assert len(re.findall(r'\\begin\{figure\}',new))==7

def prose(s):
    s=s[s.index(r'\begin{abstract}'):s.index(r'\begingroup\footnotesize\raggedright')]
    return re.sub(r'\\begin\{(figure|table|proposition|theorem)\}.*?\\end\{\1\}','',s,flags=re.S)
words=lambda s:len(re.findall(r'\b[A-Za-z]+\b',prose(s)))
we=lambda s:len(re.findall(r'\bwe\b',prose(s),re.I))
stats={'rewritten_paragraphs':len(after),'captions_tightened':17,'long_blocks_allowed_to_break':freed_blocks,'old_body_words':words(old),'new_body_words':words(new),'old_we_mentions':we(old),'new_we_mentions':we(new),'numeric_wording_changes':numeric_changes}
(OUT/'SheafCode_IRIS_Final.tex').write_text(new,encoding='utf-8')
(OUT/'repetition_edit.diff').write_text(''.join(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),fromfile='before_repetition_edit.tex',tofile='SheafCode_IRIS_Final.tex')),encoding='utf-8')
(REVIEW/'repetition_edit_stats.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
print(json.dumps(stats,indent=2))
