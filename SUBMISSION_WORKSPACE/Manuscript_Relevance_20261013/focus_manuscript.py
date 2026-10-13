from pathlib import Path
import hashlib,json,re,difflib

out=Path(__file__).resolve().parent
source=Path(r'C:\Users\rishi\Downloads\SheafCode_IRIS_Natural_Writing_Final (1).tex')
before=source.read_text(encoding='utf-8');text=before;changes=[]
expected='c6bff7ed1dfe01f979da12b329937c83fff76488ead25c331bd6e50549bb9250'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,'Active file changed; review new edits before applying cuts.'
(out/'Before_Relevance_Edit.tex').write_bytes(source.read_bytes())
def replace(name,old,new,reason):
    global text
    assert text.count(old)==1,(name,text.count(old))
    text=text.replace(old,new)
    changes.append(dict(change=name,reason=reason,removed=old,inserted=new))

start=text.index(r'\subsection{Implementation verification}')
end=text.index(r'\subsection{Alert detection and environmental relevance}',start)
joint=r'''\subsection{Joint threshold certification}

Of 272 hidden-target joint cases, 260 have feasible models and 12 are inconsistent. There are 155 below-reference certificates and 117 abstentions (including the 12 inconsistent cases). The checker abstains on all 40 positive truth cases and issues no measured above-reference joint certificate.

These results depend on the stated error limits; they do not establish field-valid error bounds.


'''
replace('replace_verification_with_scientific_result',text[start:end],joint,
    'Keep the completed 272-case joint-decision experiment as a scientific result. Remove rank/check totals, optimizer replay counts, synthetic software controls, the selected debug trace, development runtime timings and the separate scalar audit. Remove the adjacent historical-results pointer.')

start=text.index(r'\paragraph{Earlier supplementary studies.}')
end=text.index(r'\end{document}',start)
replace('remove_historical_result_inventory',text[start:end],'',
    'Remove the entire older-protocol family, including both favorable and unfavorable results. Its snapshot, window, later-period, Beijing, neural and weather protocols are outside this paper\'s current experiments.')

old="On 190 Indian daily-category tests, SheafCode and the city mean both reach 73.7\\% accuracy; all methods miss the one Poor-or-worse day. Four comparisons with official bulletins are descriptive, not regulatory validation \\citep{cpcbrecent,caqm2025}. Earlier joint-threshold trials certify nothing in India or the US and four below-reference decisions in the UK; they are distinct from the later 272-case test."
replace('remove_small_regulatory_and_early_trials',old,'',
    'Remove the approximate daily-category and four-bulletin side evaluation, whose protocols are not developed here, and early certification trials superseded by the completed 272-case experiment. Keep hourly alert detection with its false-alarm tradeoff.')
replace('remove_eight_value_us_aside',' The analogous US test has only eight values.','',
    'An eight-value aside does not support an interpretable external comparison. The completed Indian missingness-pattern experiment remains.')
replace('align_alert_section_scope',
    'We also check how often the estimated readings produce the correct approximate alert or daily pollution category.',
    'We also check how often the estimated readings produce the correct approximate alert.',
    'The daily-category side evaluation is removed, so the section introduction must describe only the retained alert experiment.')
replace('describe_actual_nonempty_bank_population',
    'produce 414 settings, including 90 empty-bank settings from the five groups without complete test patterns.',
    'produce 324 settings across the 18 usable banks.',
    '18 completed banks times 3 thresholds times 3 error levels times 2 corruption budgets yields 324 actual configurations. The 90 empty-input records are bookkeeping, rather than experiments. The statement that 18 of 23 attempted groups have data remains.')
replace('remove_unnecessary_search_limit_aside',
    'Our implementation queries each station once under a search limit, since the number of search states can grow exponentially',
    'Our implementation queries each station at most once; the number of search states can grow exponentially',
    'The completed reported adaptive settings have zero search-limit outcomes. Preserve the mathematical scaling limitation without making a configured safeguard sound like a truncated result.')
replace('remove_repeated_archive_inventory',
    " The supporting files remain in the authors' archive; the main methods, results, mathematical statements, and proofs are included in this paper.",'',
    'This archive-inventory sentence adds no scientific limitation and duplicates Data and code availability.')
replace('remove_repeated_proof_inventory',
    ' The main procedures, comparisons, mathematical results, and proofs are included in the paper itself.','',
    'Data availability should identify access to data and code rather than repeat an inventory of paper contents.')

                                                                 
for key in ['cpcbrecent','holm1979']:
    entries=list(re.finditer(r'\\bibitem[\s\S]*?(?=\n\\bibitem|\n\\end\{thebibliography\})',text))
    entry=next(m.group(0) for m in entries if re.search(r'\]\{'+key+r'\}',m.group(0)))
    replace('remove_unused_reference_'+key,entry,'','Remove a bibliography entry left uncited by the removed evaluation family.')

target=out/'SheafCode_Focused_Manuscript.tex';target.write_text(text,encoding='utf-8')
(out/'RELEVANCE_CHANGES.json').write_text(json.dumps(changes,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(out/'RELEVANCE_CHANGES.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile=source.name,tofile=target.name)),encoding='utf-8')
print(json.dumps(dict(changes=len(changes),original_hash=expected,focused_hash=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))
