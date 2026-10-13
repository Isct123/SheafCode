from pathlib import Path
import re,json,difflib,hashlib

OUT=Path(__file__).resolve().parent
source=OUT/'Original_Submitted_Manuscript.tex'
text=source.read_text(encoding='utf-8');original=text;changes=[]

def fix(label,before,after,reason,count=1):
    global text
    actual=text.count(before)
    assert actual==count,(label,actual,count)
    line=original[:original.index(before)].count('\n')+1
    text=text.replace(before,after)
    changes.append({'id':label,'original_line':line,'occurrences':count,'before':before,'after':after,'reason':reason})

fix('normalization','We normalize each pollutant by subtracting its training mean and dividing by its training standard deviation.',
    'We normalize each pollutant at each station by subtracting its training mean and dividing by its training standard deviation.',
    'The saved means and standard deviations have a separate station/channel entry.')
fix('threshold_boundary','(all are below/all are high)','(all are at or below/all are above)',
    'The decision convention is non-exceedance at equality and exceedance strictly above the reference.')
fix('actual_graph_comment','d 17-8235','d 10485-8235','Edge d was assigned to the wrong station pair in the diagram comment.')
fix('actual_graph_edge',r'\draw[seed, overlay] (s17) to[bend right=28] node[lab, pos=0.45, left, text=black] {d} (s8235);',
    r'\draw[seed] (s10485) to[bend left=18] node[lab, pos=0.72, below right=3pt, text=black] {d} (s8235);',
    'The stored fitted graph connects hidden station 8235 to 10485, not to 17.')
fix('actual_graph_incidence','{17/d, 8235/d, 50/f, 8235/f, 8235/i, 8239/i}',
    '{10485/d, 8235/d, 50/f, 8235/f, 8235/i, 8239/i}',
    'The station-to-edge panel must have the same incidences as the fitted graph.')
fix('weighted_counterexample',r'as the separate possibilities $(2,0)$ and $(0,2)$ show;',
    r'as the separate possibilities $(2,0)$ and $(0,2)$ with reference levels of 1.5 show;',
    'With the earlier 0.75 references the shared equal weights already certify both states. With 1.5 references each joint score is 4/3, while every common convex weighting has minimum at most 2/3.')
fix('monotone_search','The upper bound $U$ also need not increase if child bounds are limited by their parent\'s valid bound.',
    'The upper bound $U$ also cannot increase if child bounds are limited by their parent\'s valid bound.',
    'Under the stated parent cap the covering-tree bound is monotone, as proved in the appendix.')
fix('nonnegative_budget','the total available budget is $B$.',r'the total available budget is $B\geq0$.',
    'This ensures the feasible query-set family contains the empty plan and its maximum is defined.')
fix('adaptive_compatible_history',r'\begin{proposition}[Exact adaptive repair]\label{prop:adaptive}'+'\n\n',
    r'\begin{proposition}[Exact adaptive repair]\label{prop:adaptive}'+'\n\nConsider a compatible history, so at least one pattern remains.\n\n',
    'An empty pattern bank is inconsistent with the model; vacuous label agreement must not be interpreted as a certificate.')
fix('benchmark_start','Records begin on 14 October 2025.','The original-network records begin on 1 January 2026.',
    'The three original-network hourly grids start on 1 January 2026. The expanded EPA roles start on 14 October 2025; these are different datasets and must not be conflated.')
fix('expanded_start','The expanded EPA data extend through March 2026 and the UK and NSW data through June 2026,',
    'The expanded EPA data begin on 14 October 2025 and extend through March 2026; the expanded UK and NSW data begin on 1 January 2026 and extend through June 2026,',
    'The expanded-role manifests use 14 October 2025 for EPA and 1 January 2026 for AURN, NSW and the longer Indian split. State provider-specific starts rather than treating them as the original-network dates.')
fix('certification_two_variants',
    r'We test original readings and shifts of $\pm1$ normalized unit, for 272 cases total, using only spatial comparisons in $D$.',
    'At each hour, we test original readings and one variant with all channels at one surviving station shifted by $+1$ or $-1$ normalized unit, alternating signs across the sampled hours, for 272 cases total, using only spatial comparisons in $D$.',
    'The generator creates clean and unit_shift scenarios per hour, not clean/up/down. It alters a single surviving station; hidden target truth is unchanged.')
fix('empty_bank_configuration_count','produce 414 settings.',
    'produce 414 settings, including 90 empty-bank settings from the five groups without complete test patterns.',
    '23 attempted groups Ãƒâ€” 18 configurations = 414; 18 nonempty banks Ãƒâ€” 18 = 324; the remaining 90 configurations are recorded as trivial_or_empty.')
fix('abstract_positive_certification',
    'The method still performs worse on some networks and struggles with fault localization and prediction intervals during extreme pollution.',
    'The method still performs worse on some networks and struggles with fault localization and prediction intervals during extreme pollution; the joint checker abstains on all 40 test exceedances.',
    'The general certification theorem is not evidence of measured positive-alert certification: all 40 positive-truth archive cases abstained.')
fix('primary_vs_ablation',
    'These ablations do not isolate every component, and more elaborate maps are not always better.',
    'These ablations do not isolate every component, and more elaborate maps are not always better. The primary benchmark retains the CCA-based model specified in the protocol; the identity-map result is a separate ablation, not a replacement chosen after testing.',
    'The protocol specifies CCA maps before outcome evaluation. Switching the headline to the better ablation after seeing test results would change the declared primary comparison.')
fix('grin_point_comparison',
    'SheafCode beats GRIN across three seeds in India, US D1 and UK E1, but GRIN wins on Illawarra F1',
    'SheafCode has lower error than GRIN across three seeds in India, US D1 and UK E1, but GRIN has lower error on Illawarra F1',
    'These are numerical comparisons; the intervals include zero for D1 seed 0 and all three F1 seeds, so statistical superiority is not established.')
fix('guard_equality','confirming all upper bounds are below the reference.',
    'confirming all upper bounds are at or below the reference.',
    'The implemented and proved safe decision uses y+eta <= threshold, including equality.')
fix('test_case_scope','In all 40 real-data hidden-target exceedances,','In all 40 test-case hidden-target exceedances,',
    'The archive target values are real observations, but the outage and some surviving readings are simulated or deliberately modified.')
fix('fault_scaled_label','{raw gain}','{scaled readings}',
    'The fault is multiplication of physical concentration by 1.5, called scaled readings in the prose.',count=2)
fix('sensitivity_scope',
    'These sensitivity tests are separate from the fixed benchmark and do not establish natural-fault recovery.',
    'These sensitivity tests use a separately sampled history/horizon/context protocol, so their 24-hour cells do not reproduce the fixed benchmark, and they do not establish natural-fault recovery. Blue indicates a positive reconstruction gain or localization rate; orange indicates a negative reconstruction gain.',
    'The sensitivity grid uses different selected day windows and noise/context tests. This distinguishes its 18.4% cell from the primary 16.48% gain and defines the color encodings.')
fix('holm_citation',"we adjust multiple comparisons by Holm's method.",
    "we adjust multiple comparisons by Holm's method \\citep{holm1979}.",
    'Supply the primary source for the named multiple-testing procedure.')
fix('hall_attribution',r'with the remaining edge constraints determining the degrees of freedom \citep{hall1935}.',
    'with the remaining edge constraints determining the degrees of freedom.',
    'Hall\'s matching theorem is not a source for the actual numerical sheaf rank claim. The preceding Ghrist/Hansen citations and the provided spanning-tree proof remain.')

                                                                       
for key in ['gibbs2021','hall1935']:
    pattern=r'\\bibitem\[[\s\S]*?\]\{'+key+r'\}[\s\S]*?(?=\n\\bibitem|\n\\end\{thebibliography\})'
                                                                                          
    entries=list(re.finditer(r'\\bibitem[\s\S]*?(?=\n\\bibitem|\n\\end\{thebibliography\})',text))
    entry=next(m.group(0) for m in entries if re.search(r'\]\{'+key+r'\}',m.group(0)))
    fix('unused_reference_'+key,entry,'',f'The {key} entry has no remaining citation in this manuscript.')

holm=r'''\bibitem[Holm(1979)]{holm1979}
Sture Holm.
\newblock A simple sequentially rejective multiple test procedure.
\newblock \emph{Scandinavian Journal of Statistics}, 6\penalty0
  (2):\penalty0 65--70, 1979.
\newblock URL \url{https://www.jstor.org/stable/4615733}.

'''
fix('holm_reference',r'\bibitem[Hotelling(1936)]{hotelling1936relations}',
    holm+r'\bibitem[Hotelling(1936)]{hotelling1936relations}',
    'Primary publication verified against the original Holm paper.')
fix('search_trace_rounding', 'the upper bound falls from 3.58 to 2.86 after seven nodes. A feasible lower bound of 2.86 appears after eight nodes.', 'the upper bound falls from 3.58 to 2.85 after seven nodes. A feasible lower bound of 2.85 appears after eight nodes.', 'The outcome-selected saved trace has upper/lower value 2.8456435500471424, which rounds to 2.85; the objective offset is already excluded as stated.')
fix('l1_reference_notation', 'dictionaries via l1 minimization.', r'dictionaries via $\ell_1$ minimization.', 'Render the mathematical norm symbol in the published title correctly.')
fix('chemical_intro',r'$NO_2$','NO$_2$','Use upright chemical element symbols in the pollutant name.')
fix('chemical_table','Network & PM2.5 & PM10 & Ozone & NO2\\',
    'Network & PM2.5 & PM10 & Ozone & NO$_2$\\',
    'Use proper chemical subscript notation in the channel heading.')

target=OUT/'SheafCode_IRIS_Natural_Writing_Technically_Corrected.tex'
target.write_text(text,encoding='utf-8')
(OUT/'TECHNICAL_CHANGES.json').write_text(json.dumps(changes,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(OUT/'TECHNICAL_CHANGES.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='Original_Submitted_Manuscript.tex',tofile=target.name)),encoding='utf-8')
print(json.dumps({'corrected_file':str(target),'changes':len(changes),'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'corrected_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'abstract_words':len(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',text,re.S)[1].split())},indent=2))
