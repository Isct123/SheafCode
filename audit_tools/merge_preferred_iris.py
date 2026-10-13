                                                                             
from pathlib import Path
import re
import difflib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS'
path = OUT / 'SheafCode_IRIS_Final.tex'
audited = path.read_text(encoding='utf-8')
preferred_path = Path(r'C:\Users\rishi\Downloads\SheafCode_Preferred_Writing_New_Figures.tex')
preferred = preferred_path.read_text(encoding='utf-8-sig')
s = preferred

def between(text, start, end):
    return text[text.index(start):text.index(end,text.index(start))]

def replace(before, after):
    global s
    assert s.count(before)==1, before[:100]
    s=s.replace(before, after)

def replace_between(start,end,new):
    replace(between(s,start,end),new)

                                                                                           
replace(r'\usepackage[authoryear,round]{natbib}\bibliographystyle{plainnat}',r'\usepackage[numbers,sort&compress]{natbib}')
replace(r'\begingroup\small\bibliography{references}\endgroup',
        between(audited,r'\begingroup\footnotesize\raggedright',r'\end{document}').replace(r'\begingroup\footnotesize',r'\begingroup\small'))
replace('normalized root mean squared error is','normalized root mean squared error (RMSE) is')
replace('The demonstrated benefit concerns computational monitoring continuity, not measured reductions in emissions, pollution or health risk.',
        'The demonstrated benefit is computational reconstruction under simulated outages and conditional measurement planning; we did not measure reductions in emissions, pollution or health risk.')
replace('Air-quality stations measure pollutants such as particulate matter, nitrogen dioxide and ozone.',
        'Air-quality stations measure pollutants such as particulate matter, nitrogen dioxide and ozone. PM2.5 and PM10 denote particulate matter with diameters up to 2.5 and 10 micrometres respectively.')
replace('Suppose it falls just below an alert reference. Another state allowed by the observations and error limits might lie above it.',
        'For example, an estimated PM2.5 reading of 30 might coexist with an allowed reading of 40. At an illustrative reference of 35, these would give opposite decisions. Concentrations here are in micrograms per cubic metre ($\\mu$g/m$^3$).')
replace('If so, we can certify that decision within the model; otherwise we report uncertainty.',
        'If so, we can certify that decision within the model; otherwise we report uncertainty. A \\emph{certificate} is a conclusion justified for every state allowed by the stated assumptions.')
replace('These are applications and specialized results built on established sheaf, optimization and test-selection theory, not inventions of those foundations.',
        'These are applications and specialized results built on established sheaf, optimization and test-selection theory, not inventions of those foundations.\n\nReconstruction, certification and query planning use distinct experimental protocols. They describe stages of one monitoring workflow, but have not yet been tested together as a deployed system.')
replace('GRIN learns spatiotemporal relationships through graph message passing',
        'GRIN, a graph neural imputation method, learns relationships across stations and time by passing information along graph edges')
replace('Canonical correlation analysis (CCA), used to learn comparison maps, is also established',
        'Canonical correlation analysis (CCA), which finds correlated linear combinations of two sets of readings, is also established')
replace('Epigraph optimization, verified bounds and branch and bound supply credited tools',
        'We use established tools: epigraph optimization introduces a variable bounding the largest score, verified inequalities check numerical bounds, and branch and bound divides fault possibilities into smaller searches')
replace('Adaptive conformal inference concerns revealed prediction errors',
        'Adaptive conformal inference adjusts uncertainty intervals using revealed prediction errors')

                                                                                 
replace_between('Station $v$ carries', 'Training means and standard deviations',
                between(audited,'Let $p$ be the number', 'Each station selects'))
replace('Training means and standard deviations set each station/channel\'s scale. ', '')
replace('Stacking the spatial discrepancies gives a coboundary $\\delta$.',
        'Let $x$ stack all station vectors. Stacking the spatial discrepancies gives a linear operator, the coboundary $\\delta$.')
replace_between('Let $D$ stack spatial', 'For hour $t$, the causal solve',
                between(audited,'Let $D$ be the matrix', 'For hour $t$, the solve'))

                                                                                            
statement_pattern=r'\\begin\{(?:proposition|theorem)\}.*?\\end\{(?:proposition|theorem)\}'
proof_pattern=r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}'
for pattern in (statement_pattern,proof_pattern):
    source_blocks=re.findall(pattern,audited,re.S)
    target_blocks=re.findall(pattern,s,re.S)
    assert len(source_blocks)==len(target_blocks)==7
    for target, source in zip(target_blocks,source_blocks): replace(target,source)
replace('Choose a root and a spanning tree in each connected component $C$.',
        'A connected component $C$ is a group of stations linked by paths. Choose a root station and a spanning tree, which connects the component without cycles.')
replace('written $x_v=T_va$.',
        'written $x_v=T_va$, where $T_v$ is the product of comparison transports along that tree path.')
replace('generic matching cannot establish rank for tied learned maps',
        'graph matching alone cannot establish matrix rank when learned coefficients have algebraic relationships')
replace_between('To state the uncertainty model explicitly,', 'Let $s_k(z)=',
                between(audited,'A surviving reading may itself be faulty.', 'For pollutant $k$,'))
replace_between('Let $s_k(z)=',r'\begin{proposition}[A joint threshold',
                between(audited,'For pollutant $k$,',r'\begin{proposition}[A joint threshold'))
replace('This uses the established epigraph principle',
        'Introducing $y$ above the pollutant scores is the established epigraph principle')
replace('The next result uses known variable ranges to account for that leftover explicitly.',
        'The next result uses known variable ranges to account for that leftover explicitly. Here $x$ denotes a general vector of optimization variables, and the inequality weights $\\lambda_i$ are separate from the decoder weight $\\lambda$.')
replace_between('For this search, introduce a binary activation',r'\begin{proposition}[Anytime bounds',
                between(audited,'To bound a linear score, introduce',r'\begin{proposition}[Anytime bounds'))
replace_between('Write $Z_-$ and $Z_+$',r'\begin{proposition}[Minimum-cost decision',
                between(audited,'Let $Z_-$ and $Z_+$',r'\begin{proposition}[Minimum-cost decision'))
replace('Near-threshold states may be inseparable at positive noise, so a margin or abstention region is necessary.',
        'Near-threshold states may be inseparable at positive noise, so a margin or abstention region is necessary. The right panel of Figure~\\ref{fig:repair-intuition} previews the next question: $\\rho$ is the amount by which a true state may differ from an archived example.')
replace('For the network-maximum reference $\\tau$, a sufficient label-preserving margin is',
        'Write $a_{iv}$ for station $v$\'s reading in archived scenario $i$, and $\\ell_i$ for its decision label. For a network-maximum reference $\\tau$, scenario $i$ exceeds if $\\max_v a_{iv}>\\tau$. A sufficient label-preserving margin is')
replace('The dynamic program chooses the next station to minimize the worst-case remaining query cost.',
        'Write $b=(b_i)$ for the budgets of surviving candidates, and let $V(U,b)$ be the least possible worst-case cost of the remaining queries. A dynamic program computes this value by solving the smaller problems reached after each response.')

                                                                                     
replace_between('The original networks use',r'\section{Results and environmental relevance}',
                between(audited,'We use three original networks,',r'\section{Results and environmental relevance}'))

                                                                                   
tables=re.findall(r'\\begin\{table\}.*?\\end\{table\}',audited,re.S)
table_by_label={re.search(r'\\label\{([^}]+)\}',t).group(1):t for t in tables}
assert {'tab:external','tab:grin','tab:intervals'} <= set(table_by_label)
external_figure=re.search(r'\\begin\{figure\}.*?\\label\{fig:external\}.*?\\end\{figure\}',s,re.S)
                                                                             
figures={re.search(r'\\label\{([^}]+)\}',f).group(1):f for f in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',s,re.S)}
replace(figures['fig:external'],table_by_label['tab:external']+'\n\n'+figures['fig:external'])
replace('Complete region/code mappings, physical PM2.5 outcomes and split counts are in the companion reference.',
        'Table~\\ref{tab:external} gives region/code mappings, case counts and physical PM2.5 outcomes.')
replace('All seeds and intervals are retained in the reference.',
        'Table~\\ref{tab:grin} retains every tested seed and interval.')
replace('These comparisons do not establish a causal neural advantage.',
        'Both methods receive later context, so these comparisons do not establish performance against a causal neural method during a live outage.\n\n'+table_by_label['tab:grin'])
replace('Injected equal opposite two-fault offsets yield exact support recovery',
        'Fault localization depends strongly on the failure type. Exact support recovery means identifying precisely the faulty stations, with no extras or omissions. Injected equal opposite two-fault offsets yield exact support recovery')
replace('The reference retains cohort-macro coverage, widths, sparse-calibration infinite intervals and exclusions.',
        'Table~\\ref{tab:intervals} gives cohort-average coverage, interval widths and the frequency of infinite intervals.')
replace(figures['fig:risk'],table_by_label['tab:intervals']+'\n\n'+figures['fig:risk'])
replace('Without query corruption, all 61 two-label finite-bank settings admit a decision plan;',
        'The decision can be identifiable even when the full pattern is not. A two-label setting contains both exceedance and non-exceedance scenarios. Without query corruption, all 61 such finite-bank settings admit a decision plan;')
replace_between('Adaptive repair lowers summed',r'\begin{figure}[htbp]\centering'+'\n'+r'\resizebox{.90\linewidth}',
                between(audited,'Adaptivity gives modest savings',r'\begin{figure}[htbp]\centering'+'\n'+r'\resizebox{.90\linewidth}'))
replace_between('Across 18 chronological roles, only',r'\begin{table}[htbp]\centering\small'+'\n\n'+r'\caption{Later-date direct-reading',
                between(audited,'The main transfer problem is',r'\begin{table}[htbp]\centering\small'+'\n\n'+r'\caption{Later-date direct-reading'))
replace_between('With one corruption, only 20.53',r'\begin{figure}[htbp]\centering'+'\n'+r'\resizebox{.98\linewidth}'+'\n'+r'{%',
                '') if False else None
replace('With one corruption, only 20.53\\% of trials receive certificates. Box-plan prioritization costs 41/15 more queries than simple training-frequency order. Validation-selected ordering improves on that simple order by only 0.219\\%/0.080\\%, with losses in 7/4 of eighteen roles. We establish fewer queries than full measurement under simulated budgets, not superiority to simple orders.',
        between(audited,'The guard issued no wrong certificates',r'\begin{figure}[htbp]\centering'+'\n'+r'\resizebox{.98\linewidth}').strip())
replace('The joint-decision result deserves special attention: the checker did not certify an above-reference state in the tested hidden-target cases, even though some true readings exceeded the reference. The correct response in those cases was to abstain. Separate scalar searches and planted positive controls are documented in the technical reference. The other checks support the algorithms\' internal correctness, but they do not establish physical error bounds, polynomial scaling or independently reproduced performance.',
        'The joint-decision result deserves special attention: the checker did not certify an above-reference decision in the tested hidden-target cases, and all 40 truly positive cases received abstentions. Two planted joint-only positive controls separately check that the algorithm can issue such a certificate when the model permits it.\n\nA separate scalar check, involving one score rather than the joint maximum, gives 216 below-reference decisions, 44 uncertain cases and 12 inconsistent cases (Figure~\\ref{fig:math-integration}). These are different endpoints. The other checks support the algorithms\' internal correctness, but do not establish physical error bounds, polynomial scaling or independently reproduced performance.')
replace('Table~\\ref{tab:verification-summary} groups the checks by the question they answer.',
        'Table~\\ref{tab:verification-summary} groups the checks by the question they answer. A matrix\'s kernel consists of vectors it maps to zero. Exact modular checks use arithmetic modulo primes to certify matrix rank.')
replace('only two pass a within-network Holm correction.',
        'only two pass a within-network Holm correction, which accounts for testing several comparisons.')
replace('At an illustrative hourly PM2.5 reference of 60',
        'The alert results show a tradeoff between finding high-pollution hours and raising false alarms. Sensitivity is the fraction of truly above-reference hours flagged; the false-positive rate is the fraction of below-reference hours incorrectly flagged. At an illustrative hourly PM2.5 reference of 60')
replace('References and test-label ROC curves do not implement official regulatory alerts.',
        'The illustrative references and receiver operating characteristic (ROC) curves, which compare detection and false-alarm rates across cutoffs, do not implement official regulatory alerts.')
replace('The research archive and companion reference retain supporting protocols, failures and detailed implementation checks.',
        'Supporting protocols, failures and implementation checks remain in the authors\' research archive; the mathematical statements, proofs and principal comparisons are included here.')
replace('The strongest supported application is computational monitoring continuity.',
        'The experiments demonstrate computational reconstruction under simulated outages and conditional measurement planning. Monitoring continuity during real outages remains the intended application.')
replace_between(r'\section*{Data, code and supplementary material}',r'\section*{Acknowledgements}',
                between(audited,r'\section*{Data and code availability}',r'\section*{Acknowledgements}'))
replace('the Central Pollution Control Board for its bulletins,',
        'the Central Pollution Control Board (CPCB) for its bulletins,')
replace('Fourteen-day sensitivities are retained in the reference.',
        'Fourteen-day sensitivity results are retained in the research archive.') if 'Fourteen-day sensitivities are retained in the reference.' in s else None

                                                                                      
replace(r'\resizebox{.43\linewidth}',r'\resizebox{.52\linewidth}')
for pattern in (statement_pattern,proof_pattern):
    assert re.findall(pattern,s,re.S)==re.findall(pattern,audited,re.S)
assert len(re.findall(r'\\begin\{figure\}',s))==11
assert not re.search(r'companion|technical.reference|retained in the reference|in the companion',s,re.I)
assert not any(token in s for token in [r'\bibliography{',r'\includegraphics',r'\input{'])
path.write_text(s,encoding='utf-8')
(OUT/'readability_changes.diff').write_text(''.join(difflib.unified_diff(preferred.splitlines(True),s.splitlines(True),fromfile='preferred_writing',tofile='standalone_final')),encoding='utf-8')
report=json.loads((OUT/'final_check.json').read_text())
report.update(scope='Preferred prose with definitions before use and standalone details; no experiments rerun.',
              mathematical_statements_unchanged=True,core_proofs_unchanged=True,
              preferred_prose_source=str(preferred_path),main_text_font_points=11,
              clean_compile_pending=True,fresh_directory_single_file_compile_passed=False,
              previous_16_page_version_archived=True)
(OUT/'final_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Preferred writing merged; 7 statements, 7 proofs and all 11 diagrams retained.')
