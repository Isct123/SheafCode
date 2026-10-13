from pathlib import Path
import re
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]/'Final_Manuscript'
path=ROOT/'sections/09_results.tex'
original=path.read_text(encoding='utf-8')
parts=re.split(r'(?=\\subsection\{)',original)
sections={re.match(r'\\subsection\{([^}]+)\}',p)[1]:p for p in parts[1:]}
order=[
    'Prolonged retrospective reconstruction',
    'Pre-ranked risk groups and limitations of the certificate',
    'Primary single-station snapshots and secondary mixed outages',
    'Single-channel outages and matched-information comparisons',
    'C1 under controlled graph thinning',
    'National graph construction: limited improvement',
    'Secondary fault and network-design studies',
]
assert set(order)==set(sections)
intro=r'''\section{Results}
The results are presented in order of their role in the research argument: prolonged reconstruction gives the clearest numerical advantage; pre-ranked outage groups test the proposed sensitivity diagnostic; snapshot and channel comparisons establish the scope of spatial competitiveness; and structural and exploratory studies explain where the framework helps or fails. This presentation order does not change the prespecified comparison families or their statistical units.

\begin{table}[H]\centering\small
\caption{Main findings and evidence map. Numerical direction and corrected inference are separate. Networks overlap, so these denominators are not counts of independent replications.}\label{tab:mainfindings}
\begin{tabularx}{\linewidth}{@{}p{.22\linewidth}XX@{}}
\toprule
Finding & Numerical result & Inference and detailed evidence \\
\midrule
\textbf{Prolonged reconstruction} & Lower first-day mean error than Kalman/VAR in 9/9 configurations; median gain \textbf{34.0\%}. & 5/9 supplementary run-based tests pass within-network Holm correction. Retrospective full-window reconstruction (\cref{tab:prolonged,fig:prolonged,fig:horizons}). \\
\textbf{Outage risk ranking} & High-C2$^*$ groups have higher mean error in \textbf{7/7} evaluable networks. & 5/7 tests pass supplementary across-network Holm correction. Group composition and natural gaps limit interpretation (\cref{tab:cert,fig:cert,fig:riskassociation}). \\
\textbf{Spatial competitiveness} & Primary single-station gains in 7/8 testable networks; median gain \textbf{3.1\%}. & 2/8 comparisons pass within-network Holm correction. Channel gains are heterogeneous (\cref{tab:snapshot,fig:snapshot,tab:channel}). \\
\bottomrule
\end{tabularx}
\end{table}
'''
path.write_text(intro+'\n'.join(sections[s] for s in order),encoding='utf-8')
preserved={s:hashlib.sha256(sections[s].encode()).hexdigest() for s in order}
updated=path.read_text(encoding='utf-8')
assert all(updated.count(section)==1 for section in sections.values())
assert sorted(re.findall(r'\\includegraphics[^\n]+',original))==sorted(re.findall(r'\\includegraphics[^\n]+',updated))
(ROOT/'audit/results_preservation.json').write_text(json.dumps({'all_seven_sections_retained_verbatim':True,'subsection_sha256':preserved,'all_existing_result_figures_retained':True},indent=2)+'\n',encoding='utf-8')

old="For primary single-station snapshots, the spatial sheaf has lower mean error than Gaussian conditional prediction in seven of eight testable networks, with median relative improvement 3.1 percent; only two comparisons pass within-network Holm correction. For prolonged retrospective reconstruction, first-24-hour mean error is lower than the implemented Kalman/VAR smoother in all nine networks, with median improvement 34.0 percent. Grouping failures by run supports fewer significance claims than treating each run-station failure separately. Pre-ranked high-C2$^*$ groups have higher held-out error in all seven evaluable networks."
new="The clearest advantage is prolonged retrospective reconstruction: first-day mean error is lower than the implemented Kalman/VAR smoother in all nine networks, with median improvement 34.0 percent. Five supplementary run-based comparisons pass within-network Holm correction. Pre-ranked high-C2$^*$ groups have higher held-out error in seven evaluable networks; five tests pass supplementary across-network Holm correction. Primary single-station snapshots show smaller gains: lower mean error than Gaussian conditional prediction in seven of eight testable networks, median improvement 3.1 percent, and two within-network Holm-significant comparisons."
for name in ['sections/01_abstract.tex','portal_fields.tex']:
    p=ROOT/name;t=p.read_text(encoding='utf-8');assert old in t,name;p.write_text(t.replace(old,new),encoding='utf-8')
old_results="For primary single-station snapshots, the sheaf has lower mean error than Gaussian conditional prediction in seven of eight testable networks, with median relative improvement 3.1 percent; two comparisons pass within-network Holm correction. Prolonged retrospective reconstruction has lower first-day mean error than the implemented Kalman/VAR smoother in all nine networks, with median improvement 34.0 percent. A run-based sensitivity analysis supports fewer corrected comparisons. Pre-ranked high-C2$^*$ groups have higher error in all seven evaluable networks; six nominal tests and five supplementary across-network Holm tests pass 0.05."
new_results="Prolonged retrospective reconstruction has lower first-day mean error than the implemented Kalman/VAR smoother in all nine networks, with median improvement 34.0 percent; five supplementary run-based comparisons pass within-network Holm correction. Pre-ranked high-C2$^*$ groups have higher error in seven evaluable networks, with five supplementary across-network Holm-significant tests. Primary single-station snapshot gains are smaller: seven of eight testable networks improve over Gaussian conditional prediction, median gain 3.1 percent, and two within-network Holm-significant comparisons."
for name in ['synopsis.tex','portal_fields.tex']:
    p=ROOT/name;t=p.read_text(encoding='utf-8');assert old_results in t,name;p.write_text(t.replace(old_results,new_results),encoding='utf-8')

p=ROOT/'sections/12_conclusion.tex'
t=p.read_text(encoding='utf-8')
start="SheafCode connects missing-measurement reconstruction with rank and residual-sensitivity analysis through one learned operator. Primary single-station snapshots are competitive with dense Gaussian prediction, with median improvement 3.1\\% across eight testable networks and corrected positive evidence in two. Prolonged retrospective reconstruction gives lower first-day mean error than the implemented Kalman/VAR smoother in all nine networks, with median relative improvement 34.0\\%; the strength of significance depends on the replicate definition."
replacement="SheafCode connects missing-measurement reconstruction with rank and residual-sensitivity analysis through one learned operator. Its clearest empirical advantage is prolonged retrospective reconstruction: lower first-day mean error than the implemented Kalman/VAR smoother in all nine networks, with median relative improvement 34.0\\%. Five supplementary run-based comparisons pass within-network Holm correction; the original run--station analysis supports more significant comparisons. Primary single-station snapshots are competitive with dense Gaussian prediction, with median improvement 3.1\\% across eight testable networks and corrected positive evidence in two."
assert start in t;p.write_text(t.replace(start,replacement),encoding='utf-8')

def words(text):
    text=re.sub(r'\\url\{([^}]+)\}',r'\1',text)
    text=re.sub(r'\\[A-Za-z]+\*?(?:\[[^]]*\])?', '', text)
    return len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b",text))
counts={}
portal=(ROOT/'portal_fields.tex').read_text()
for m in re.finditer(r'\\section\*\{([^}]+)\}\s*(.*?)(?=\\section\*|\\end\{document\})',portal,re.S):counts[m[1].replace(r'\&','&')]=words(m[2])
video=(ROOT/'video_script.tex').read_text().split(r'\section*{Spoken reference script}')[1].split(r'\section*{Visual storyboard}')[0]
counts['spoken_script']=words(video)
(ROOT/'audit/word_counts.json').write_text(json.dumps(counts,indent=2)+'\n',encoding='utf-8')
print(json.dumps(counts,indent=2))
