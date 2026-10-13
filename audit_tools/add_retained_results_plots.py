                                                                           
from pathlib import Path
import hashlib, json, re, shutil
import pymupdf

w = Path('SUBMISSION_WORKSPACE')
src = w/'Source/manuscript.tex'
backup = Path('audit_tools/results_plot_addition_backup')
backup.mkdir(exist_ok=True)
assert not (backup/'manuscript.tex').exists(), 'Already applied; do not add duplicate figures.'
shutil.copy2(src, backup/'manuscript.tex')
shutil.copy2(w/'Manuscript_For_Upload.pdf', backup/'Manuscript_For_Upload.pdf')
before = src.read_text(encoding='utf-8')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
figs = w/'Source/figures'
original = Path('IRIS_Submission_Recent_Data/Expanded_Evidence_Source/figures')
inputs = [original/'faults.pdf', original/'stress.pdf', w/'Guides/Operational_Validation_Figure.pdf']
shutil.copy2(inputs[2], figs/'operational_validation.pdf')

                                                                              
out = pymupdf.open()
width = 720
docs = [pymupdf.open(p) for p in inputs[:2]]
heights = [d[0].rect.height * width/d[0].rect.width for d in docs]
height = sum(heights) + 48
page = out.new_page(width=width, height=height)
y = 0
for title, doc, h in zip(['(a) End-to-end fault recovery', '(b) Outage duration, history and missing context'], docs, heights):
    page.insert_text((8,y+15),title,fontsize=13,fontname='hebo')
    page.show_pdf_page(pymupdf.Rect(0,y+22,width,y+22+h),doc,0)
    y += h+24
target = figs/'fault_outage_robustness.pdf'
out.save(target,garbage=4,deflate=True)
for d in docs:d.close()
out.close()
                                                                             
norm = lambda s: re.sub(r'\s+',' ',s).strip()
combined = pymupdf.open(target)[0].get_text()
for p in inputs[:2]:
    assert norm(pymupdf.open(p)[0].get_text()) in norm(combined), p

robust = r'''
% BEGIN added retained robustness results
\begin{figure}[p]\centering\includegraphics[width=\linewidth]{figures/fault_outage_robustness.pdf}
\caption{Two complementary robustness checks on the original recent Indian (A) and US (B) networks. (a) End-to-end exact fault-support recovery across synthetic injection families and counts. The decoder receives no true count and reports at most two faults, making three-fault exact recovery impossible; clean-state cells are not shown as localization successes. Availability-conditioned episodes, restricted candidates and mean-filled missing context limit field interpretation. (b) Normalized causal sheaf RMSE gain over causal VAR as outage duration, pre-outage history and additional surviving-context loss change. Each cell uses matched cases within its own configuration; these sensitivity configurations are not the fixed primary comparison and are not pooled. Positive gains favour sheaf; losses and weak-fault failures are retained. Neither panel measures natural field-fault recovery or environmental intervention.}\label{fig:fault-outage-robustness}\end{figure}
% END added retained robustness results
'''
operational = r'''
% BEGIN added retained operational results
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/operational_validation.pdf}
\caption{Chronological model transfer and the direct-reading guard. Left: observed membership in the declared scenario boxes for all 18 evaluable cohort roles; the calibration rank target is not a future-coverage guarantee. Right: the fraction of stress trials receiving a direct certificate and the query saving against full-network measurement, including abstentions. No wrong issued direct certificates occurred under the simulated reading-error/fault budgets; with one corruption, most trials abstain. Roles and stress trials reuse observations and are dependent. Query savings are counts, not field expenditure, emissions reduction or superiority to the simpler training-frequency order.}\label{fig:operational-validation}\end{figure}
% END added retained operational results
'''
anchor = 'The temporal follow-up compares a four-hour causal syndrome mean'
assert before.count(anchor)==1
after = before.replace(anchor,robust+'\n'+anchor)
anchor2 = 'With one corruption, certificates are issued on only 20.53\\% of trials.'
assert after.count(anchor2)==1
after = after.replace(anchor2,operational+'\n'+anchor2)
restored = after
for name in ['robustness','operational']:
    restored = re.sub(r'% BEGIN added retained '+name+r' results.*?% END added retained '+name+r' results\n?', '', restored, flags=re.S)
assert norm(restored)==norm(before), 'Existing content changed'
src.write_text(after,encoding='utf-8')
report = dict(status='passed',date='2026-10-09',existing_content_unchanged=True,
    comparison='Removing only the two added figure blocks reproduces the full preceding manuscript after whitespace normalization.',
    source_assets={str(p):sha(p) for p in inputs},
    added_figures={p.name:sha(p) for p in [target,figs/'operational_validation.pdf']},
    original_figures_retained=9,current_figures=11,
    scope='Two figure blocks and captions added; existing prose, proofs, mathematics, tables, captions, citations and figures unchanged. Saved vector plots reused without changing their data.')
(w/'Guides/RESULTS_PLOT_ADDITION_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
