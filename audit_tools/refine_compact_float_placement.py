                                                                       
from pathlib import Path
import difflib
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'audit_tools/final_iris_review'
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Compact_Readable_Final'
SRC = OUT / 'SheafCode_IRIS_Final.tex'
BACKUP = REVIEW / 'before_float_placement_edit'
BACKUP.mkdir(exist_ok=True)
for name in [SRC.name, 'SheafCode_IRIS_Final_Focused.pdf', 'SheafCode_Final_Overleaf.zip', 'README.md', 'final_check.json']:
    if not (BACKUP / name).exists():
        shutil.copy2(OUT / name, BACKUP / name)
old = (BACKUP / SRC.name).read_text(encoding='utf-8')
new = old.replace(r'\usepackage{tikz,pgfplots}', r'\usepackage{float}'+'\n'+r'\usepackage{tikz,pgfplots}', 1)
for env, label in [('figure', 'fig:repair-intuition'), ('table', 'tab:external'), ('table', 'tab:grin')]:
    pat = rf'\\begin\{{{env}\}}\[htbp\].*?\\end\{{{env}\}}'
    blocks = [m[0] for m in re.finditer(pat, new, re.S) if rf'\label{{{label}}}' in m[0]]
    assert len(blocks) == 1, label
    new = new.replace(blocks[0], blocks[0].replace('[htbp]', '[H]', 1), 1)
for start, end in [
    (r'\subsection{Stability under scenario changes}', "Equal-cost choices can therefore differ in robustness."),
    (r'\subsection{Fault localization and interval coverage}', "Reporting at most two faults makes three-fault exact recovery impossible.")
]:
    a = new.index(start)
    b = new.index(end, a)+len(end)
    block = new[a:b]
    new = new[:a]+r'\begin{samepage}'+'\n'+block+'\n'+r'\end{samepage}'+new[b:]
SRC.write_text(new, encoding='utf-8', newline='\n')
(OUT / 'float_placement_edit.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='before_float_placement_edit/SheafCode_IRIS_Final.tex', tofile='SheafCode_IRIS_Final.tex')), encoding='utf-8')
print('Anchored Figure 4 and Tables 4-5; kept stability and fault-testing passages together.')
