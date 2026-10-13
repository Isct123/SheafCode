from pathlib import Path
import json,re,hashlib
r=Path(__file__).resolve().parent.parent;w=r/'SUBMISSION_WORKSPACE';p=w/'Source/manuscript.tex'
before=p.read_text(encoding='utf-8');b=r/'audit_tools/writing_review_backup';b.mkdir(exist_ok=True)
assert not (b/'manuscript.tex').exists();(b/'manuscript.tex').write_text(before,encoding='utf-8')
after=before
def change(old,new):
    global after
    assert old in after,old;after=after.replace(old,new,1)
change('This project asks two connected questions: can those surviving readings and pre-outage history improve a causal reconstruction, and what pollution information remains justified when reconstruction is ambiguous or some stations are faulty?','This project asks two connected questions: can those surviving readings and pre-outage history improve a causal reconstruction, and what pollution information remains justified when reconstruction is ambiguous or some stations are faulty? Here, repair means selecting additional station measurements to resolve a decision.\n\nThe fixed Indian benchmark finds a 16.48\\% reduction in normalized RMSE against the tuned causal VAR; Section~\\ref{sec:results} presents the matched comparisons and transfer tests.')
change('Spatial statistics models dependence across locations','Spatial statistical models describe dependence across locations')
change('Importantly, \\citet{pham2022airquality} already applied','\\citet{pham2022airquality} previously applied')
change('Selection chooses its hyperparameters.','Selection, also called validation in the comparator descriptions, chooses its hyperparameters.')
change('For a normalized station vector $x_v\\in\\R^p$,','Here RMSE denotes root mean squared error. For a normalized station vector $x_v\\in\\R^p$,')
change('Let $Z_-$ and $Z_+$ be the declared safe and exceedance explanations, with all candidate states labelled accordingly.','Let $n$ be the number of candidate query stations, and let $Z_-$ and $Z_+$ be the declared safe and exceedance explanations, with all candidate states labelled accordingly.')
change('\\section{Results and environmental relevance}','\\section{Results and environmental relevance}\\label{sec:results}')
discussion=r'''
The reconstruction results answer the first research question with a qualified positive result. The fixed Indian causal comparison improves both normalized multichannel error and physical PM2.5 error, while the external gains depend on the cohort and comparator. The identity-map ablation also prevents assigning that advantage specifically to learned CCA maps. Together, these findings support the tested reconstruction procedure without establishing a universal ranking of methods or isolating the cause of its gains.

The decision experiments answer a separate question about what can be established from incomplete information. Archive-based repair resolves some decisions with fewer queries than full-pattern identification, and its stability radius quantifies a conditional tolerance. Later-date model-membership failures show why this guarantee cannot be applied solely because a point estimate is plausible. The direct-reading guard supplies decisions under the simulated reading-error and corruption budgets, but its abstentions and the slightly cheaper simple query order limit operational claims. Better reconstruction, a valid conditional certificate and lower measurement cost are distinct outcomes.

'''
change('\\section{Discussion and limitations}\n','\\section{Discussion and limitations}\n'+discussion)
availability=r'''
\section*{Data and code availability}
Public observation providers and access resources are cited in Methods. Acquisition requests, station identifiers, unit conventions, source hashes, fitted-model records, case-level results and verification code are retained in the accompanying research archive. In the supplied project workspace, \path{Research_Release/} contains the research artifacts and \path{audit_tools/} contains the audit and replay tools; the complete technical reference is retained in \path{IRIS_Submission_Recent_Data/Supporting_Reference/}. The Overleaf handoff contains manuscript sources, bibliography and figures. A public repository URL for the research archive is not provided in this manuscript.

'''
change('\\appendix\n',availability+'\\appendix\n')
change('\\section{Supporting evidence and completeness of reporting}','\\section{Supporting evidence under separate protocols}')
change('The main text reports the designated recent comparisons and the linked decision-repair results. It does not reproduce every saved case table. This appendix restores supporting outcomes that would otherwise be represented only by a general reference to the evidence archive. Case counts below describe their own protocols; they are not extra independent replications or additions to the primary station-day denominator.','The main text reports the designated recent comparisons and the linked decision-repair results. This appendix summarizes additional operator checks and earlier reconstruction studies under their own protocols. Complete case tables remain in the research archive. The counts below are not extra independent replications or additions to the primary station-day denominator.')
change('The UK safe decisions must not be erased by a blanket claim that every original-network trial abstained.','The original-network UK trial therefore has a different decision distribution from the Indian and US trials.')
change('The following selected findings are retained from the audited technical reference, not represented as new reruns in this document revision.','The following selected findings summarize the audited technical reference; they are separate from the recent benchmark.')
change('\\subsection{What the concise manuscript does and does not reproduce}','\\subsection{Additional analyses in the technical reference}')
change('They are not additional demonstrated first-ever theorems. Every result family and its reporting location is mapped in the accompanying completeness audit; underlying case tables are retained for reproduction rather than inserted wholesale into the manuscript.','These derivations use established mathematical foundations. The accompanying completeness audit maps result families to their reporting locations; underlying case tables are retained for reproduction.')
for env in ['proposition','theorem','abstract','figure','table']:
    pat=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
    assert re.findall(pat,before,re.S)==re.findall(pat,after,re.S),env
pat=r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}'
assert re.findall(pat,before,re.S)==re.findall(pat,after,re.S)
assert set(re.findall(r'\\cite\w*\{([^}]+)\}',before))==set(re.findall(r'\\cite\w*\{([^}]+)\}',after))
p.write_text(after,encoding='utf-8')
report=dict(status='passed',date='2026-10-10',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),existing_formal_statements_proofs_tables_figures_abstract_unchanged=True,existing_citations_retained=True,changes=['Restored a one-sentence introduction headline preview with Results cross-reference','Defined informational repair and selection/validation terminology','Clarified wording and defined the candidate-station count outside the formal proposition','Added interpretation to Discussion using existing results and limitations','Added honest archive/data/code availability statement without inventing a public repository','Replaced editing-history language in appendix with research reporting'],remaining_personal_items=['Author field is blank; determine whether this is required anonymity or add actual authors before submission','Confirm school-teacher acknowledgement and contribution/assistance details','Public sharing/access arrangements for the full research archive are not specified'],scope='Writing and research-paper structure review; no new results, experiments, theorem claims or public deposition.')
(w/'Guides/WRITING_REVIEW_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
