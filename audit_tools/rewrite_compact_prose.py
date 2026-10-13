                                                                                
from pathlib import Path
from collections import Counter
import difflib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'audit_tools/final_iris_review'
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Compact_Readable_Final'
OLD = REVIEW / 'before_natural_prose_rewrite/SheafCode_IRIS_Final.tex'


def indexed(path):
    s = path.read_text(encoding='utf-8')
    return {int(m[1]): m[2].strip() for m in re.finditer(
        r'^\[(\d+)\][ \t]*(.*?)\s*(?=^\[\d+\]|\Z)', s, re.M | re.S)}


original = OLD.read_text(encoding='utf-8')
text = original
inventory = indexed(REVIEW / 'prose_inventory.txt')
replacements = indexed(REVIEW / 'natural_prose_replacements.txt')
numeric_review = []
for index, after in replacements.items():
    before = inventory[index]
    assert text.count(before) == 1, (index, before[:100])
    text = text.replace(before, after)
    numbers = lambda s: Counter(re.findall(r'(?<![A-Za-z])\d+(?:[.,]\d+)*', s))
    removed = numbers(before) - numbers(after)
    added = numbers(after) - numbers(before)
    if removed or added:
        numeric_review.append({'paragraph': index, 'removed': dict(removed), 'added': dict(added)})

captions = [
    "The diagram shows the fitted five-station Indian graph and the maps on one comparison edge. The positions are schematic. Orange marks an illustrative hidden station rather than a recorded outage. The endpoint maps transform the station vectors into the edge's comparison space, labelled $C_a$ for edge $a$, before comparing them.",
    "The diagram shows which observations can enter the estimate at the first hidden hour. We can use available history and current surviving readings. We cannot use hidden target readings or future observations, and we do not assume that naturally missing context is available.",
    "This example uses illustrative concentrations rather than measurements. The segment $a+b=2$ stays outside the box where both values are at most 0.75, even though either individual value can be zero.",
    "The left example shows how B separates both opposite pairs when $f=0$. With $f=1$, each pair needs three separators. The right example shows expanded reading intervals meeting at the strict stability boundary. Both panels are mathematical examples rather than measured pollution outcomes.",
    "The table lists the native pollutant units in the original networks. A dash marks a channel that we do not use.",
    "The table gives the chronological period lengths for the original networks in hours, including missing positions.",
    "The table lists the model settings considered during selection. We use selection data to choose among the alternatives.",
    "We compare causal sheaf reconstruction with five controls on matched cases. Positive gain means that sheaf has lower normalized RMSE. India has 77 station-days with seven-day intervals. The secondary US comparison has 24 station-days and too few calendar bins for intervals. We do not pool the networks.",
    r"The table reports gains (\%) and paired seven-day intervals for every evaluable external cohort on matched station-days. Negative values mean that sheaf has higher error.",
    r"The table reports the normalized retrospective sheaf gain (\%) for every tested GRIN seed.",
    "The top panels report exact localization across injected fault types and counts. The bottom panels show matched reconstruction gains as outage length, history and context loss change. These settings are separate from the fixed benchmark and do not establish recovery from naturally occurring faults.",
    r"We evaluate no-feedback PM2.5 snapshot intervals with a nominal coverage target of 90\% and no additional context loss.",
    "The table compares decision repair with full-pattern identification in finite archives. The settings are dependent. We count full-pattern impossibility only among settings with feasible decision plans, and compare query totals where both tasks are feasible.",
    "We compare the radius of a minimum-cost plan with the best radius available at the same cost. Points above the diagonal allow more scenario change without extra queries. Every feasible setting is plotted, and guarantees hold strictly inside the radius.",
    "The table reports the direct-reading stress simulation on later dates. Sequential and full-network guards give identical decisions and abstentions on matched responses. Savings include trials that end in abstention and count queries rather than field expenditure.",
    "The table groups implementation checks by their purpose. The counts describe dependent computational checks rather than independent field trials.",
    "The table reports earlier studies under their own protocols, including comparisons where sheaf loses.",
]
matches = list(re.finditer(r'\\caption\{', text))
assert len(matches) == len(captions) == 17
for m, caption in reversed(list(zip(matches, captions))):
    start = m.end()
    end = start
    depth = 1
    while depth:
        if text[end] == '{': depth += 1
        elif text[end] == '}': depth -= 1
        end += 1
    text = text[:start] + caption + text[end-1:]

extra = [
    (r'\emph{Proof in Appendix~', r'\emph{The proof is in Appendix~'),
    (r"VAR/RF gains use normalized multichannel error; PM2.5 gains use physical concentration error. Brackets give paired seven-day bootstrap intervals; a dash means insufficient calendar bins. EPA codes identify state/county; UK/NSW names identify regions. EPA 06-073 and 06-111 lack usable test observations and are excluded, not successes.",
     r"The VAR and RF gains use normalized multichannel error, while PM2.5 gains use physical concentration error. Brackets give paired seven-day bootstrap intervals. A dash means there are too few calendar bins for an interval. EPA codes identify the state and county, and UK and NSW names identify the regions. We exclude EPA 06-073 and 06-111 because they lack usable test observations."),
    (r"A = original Indian all-day comparison; D1 = EPA 04-013; E1 = AURN central; F1 = NSW Illawarra. All gains use sheaf as numerator. Seeds share truth observations and are not independent datasets.",
     r"A is the original Indian all-day comparison, D1 is EPA 04-013, E1 is AURN central, and F1 is NSW Illawarra. All gains use sheaf as the numerator. The seeds share the same truth observations, so they are not independent datasets."),
    (r"Macro coverage gives each cohort equal weight (20 overall and 19 with extreme readings). Pooled coverage gives each cell equal weight. Width averages each cohort's finite interval widths in $\mu$g/m$^3$; infinite intervals are excluded. Infinite (\%) is the percentage of cells given an interval with no finite bounds.",
     r"Macro coverage gives each cohort equal weight across the 20 cohorts overall and the 19 with extreme readings. Pooled coverage gives each cell equal weight. Width is the average of the cohort-level finite interval widths in $\mu$g/m$^3$. We exclude infinite intervals from this average. The final column gives the percentage of cells whose intervals have no finite bounds."),
    (r"Sheaf & Decoder weight $\lambda$: 0.1 or 1; temporal scale: 0.5 or 2.",
     r"Sheaf & We choose the decoder weight $\lambda$ from 0.1 or 1 and the temporal scale from 0.5 or 2."),
    (r"VAR \citep{lutkepohl2005var} & Ridge: 1 or 100; noise scale: 0.05 or 0.5; covariance shrinkage: 0, 0.2 or 1.",
     r"VAR \citep{lutkepohl2005var} & We choose ridge from 1 or 100, noise scale from 0.05 or 0.5, and covariance shrinkage from 0, 0.2 or 1."),
    (r"RF \citep{breiman2001} & 64 trees; depth at most 12; minimum leaf size 5 or 20 training examples; at most 80 sampled training days.",
     r"RF \citep{breiman2001} & We use 64 trees with a maximum depth of 12. We choose a minimum leaf size of 5 or 20 training examples and sample at most 80 training days."),
    (r"GRIN \citep{cini2022grin} & Three random seeds; hidden-layer width 32 or 64; at most 30 epochs; stop after seven epochs without selection improvement.",
     r"GRIN \citep{cini2022grin} & We test three random seeds and choose a hidden-layer width of 32 or 64. Training lasts at most 30 epochs and stops after seven epochs without selection improvement."),
    ("All nominal full operators have zero kernel; separate nonzero-kernel and singular controls test other cases.",
     "All nominal full operators have zero kernel. We test other cases with separate nonzero-kernel and singular controls."),
    ("All 40 positive truth cases abstain; there is no measured above-reference joint certificate.",
     "The checker abstains on all 40 positive truth cases and issues no measured above-reference joint certificate."),
    ("All 414 fixed configurations, 122 stability frontiers (594 budget points) and 76 adaptive optima (72,950 subproblems) agree with checks.",
     "Checks agree for all 414 fixed configurations, 122 stability frontiers covering 594 budget points, and 76 adaptive optima covering 72,950 subproblems."),
    ("Sixty synthetic multicover cases reach 128 stations; greedy matches 19/24 small optima and otherwise needs at most one extra query.",
     "Sixty synthetic multicover cases reach 128 stations. The greedy method matches 19/24 small optima and otherwise needs at most one extra query."),
    ("For the search with the largest initial gap, the upper bound falls from 3.58 to 2.86 after seven nodes; a feasible lower bound of 2.86 appears after eight (linear objective, offset excluded). Median times are 0.025 seconds for support LPs and 0.077 for verified search; verification adds overhead.",
     "For the search with the largest initial gap, the upper bound falls from 3.58 to 2.86 after seven nodes. A feasible lower bound of 2.86 appears after eight nodes. These values use the linear objective without its offset. Median times are 0.025 seconds for support LPs and 0.077 seconds for verified search. Verification adds overhead."),
    (r"Median gain 3.1\% over dense Gaussian conditioning; 7/8 favour sheaf, 2/8 pass within-network Holm correction.",
     r"The median gain over dense Gaussian conditioning is 3.1\%. Sheaf has lower error in 7/8 comparisons, and 2/8 pass within-network Holm correction."),
    (r"Median gain 34.0\% over the implemented Kalman/VAR smoother across nine configurations; 5/9 supplementary run-based comparisons pass Holm.",
     r"The median gain over the implemented Kalman/VAR smoother is 34.0\% across nine configurations. Of the supplementary run-based comparisons, 5/9 pass Holm correction."),
    (r"Gain 23.2\% over full-covariance VAR on 399 station-days and 58 dates.",
     r"The gain over full-covariance VAR is 23.2\% on 399 station-days and 58 dates."),
    (r"PM-only gain 9.5\% on 74 station-days and 19 dates; not an independent replication.",
     r"The PM-only gain is 9.5\% on 74 station-days and 19 dates. This is not an independent replication."),
    (r"All-pollutant RMSE 5.93\% higher than VAR on 502 station-days.",
     r"All-pollutant RMSE is 5.93\% higher than VAR on 502 station-days."),
    ("Historical Delhi favours GRIN on every seed; weather controls establish no improvement.",
     "GRIN has lower error in historical Delhi on every seed. The weather controls establish no improvement."),
]
for before, after in extra:
    assert text.count(before) == (7 if before.startswith(r'\emph{Proof') else 1), before
    text = text.replace(before, after)

                                                                            
for env in ['proposition', 'theorem', 'equation', 'thebibliography', 'tikzpicture']:
    pattern = rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}'
    assert re.findall(pattern, text, re.S) == re.findall(pattern, original, re.S), env
assert text[text.index(r'\appendix'):] == original[original.index(r'\appendix'):]
assert re.findall(r'\\\[.*?\\\]', text, re.S) == re.findall(r'\\\[.*?\\\]', original, re.S)
assert re.findall(r'\\(?:sub)?section\*?\{.*?\}', text) == re.findall(r'\\(?:sub)?section\*?\{.*?\}', original)
for macro in ['label', 'citep', 'citet', 'ref', 'eqref']:
    pattern = rf'\\{macro}\{{(.*?)\}}'
    assert Counter(re.findall(pattern, text)) == Counter(re.findall(pattern, original)), macro
old_tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', original, re.S)
new_tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', text, re.S)
assert len(old_tables) == len(new_tables) == 10
for before, after in zip(old_tables, new_tables):
    body = lambda s: re.search(r'\\begin\{tabularx?\}.*?\\end\{tabularx?\}', s, re.S)[0]
    assert Counter(re.findall(r'\d+(?:[.,]\d+)*', body(before))) == Counter(re.findall(r'\d+(?:[.,]\d+)*', body(after)))
assert len(re.findall(r'\\begin\{figure\}', text)) == 7
(OUT / 'SheafCode_IRIS_Final.tex').write_text(text, encoding='utf-8')
(OUT / 'natural_prose_rewrite.diff').write_text(''.join(difflib.unified_diff(
    original.splitlines(keepends=True), text.splitlines(keepends=True),
    fromfile='before_natural_prose_rewrite.tex', tofile='SheafCode_IRIS_Final.tex')), encoding='utf-8')
(REVIEW / 'natural_prose_numeric_review.json').write_text(json.dumps(numeric_review, indent=2), encoding='utf-8')
print(json.dumps({'rewritten_paragraphs': len(replacements), 'numeric_wording_changes': numeric_review}, indent=2))
