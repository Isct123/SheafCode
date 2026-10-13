                                                                                     
from pathlib import Path
import difflib
import shutil
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Shorter_Readable_Final'
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Compact_Readable_Final'
OUT.mkdir(exist_ok=True)
BACKUP = ROOT / 'audit_tools/final_iris_review/before_layout_tighten'
BACKUP.mkdir(exist_ok=True)
for name in ['SheafCode_IRIS_Final.tex', 'SheafCode_IRIS_Final.pdf',
             'SheafCode_Final_Overleaf.zip', 'README.md', 'final_check.json', 'display_changes.diff']:
    if not (BACKUP / name).exists():
        shutil.copy2(BASE / name, BACKUP / name)
original = (BACKUP / 'SheafCode_IRIS_Final.tex').read_text(encoding='utf-8')
FLOAT_SPACING = r'''
% Compact float spacing while retaining room between text and displays.
\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{10pt plus 2pt minus 2pt}
\setlength{\abovecaptionskip}{6pt}
\setlength{\belowcaptionskip}{0pt}
'''
PROOF_SPACING = r'''
% Compact appendix headings; proof text retains the 11-point body size.
\makeatletter
\renewcommand{\section}{\@startsection{section}{1}{\z@}{-10pt plus -2pt minus -1pt}{5pt plus 1pt}{\normalfont\large\bfseries}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}{-5pt plus -1pt minus -1pt}{-0.7em}{\normalfont\normalsize\bfseries}}
\makeatother
'''
source = original.replace(r'\setcounter{topnumber}{3}\setcounter{totalnumber}{4}',
    r'\setcounter{topnumber}{3}\setcounter{totalnumber}{4}' + '\n' + FLOAT_SPACING)
source = source.replace(r'\resizebox{.52\linewidth}{!}', r'\resizebox{.46\linewidth}{!}')
source = source.replace('\\appendix\n', '\\appendix\n' + PROOF_SPACING)
head = source.index(r'\appendix')
source = source[:head] + re.sub(r'\\subsection\{Proof of ((?:Proposition|Theorem)~\\ref\{[^}]+\})\}',
    r'\\subsection{\1.}', source[head:])
restored = source.replace('\n' + FLOAT_SPACING, '').replace(PROOF_SPACING, '')
restored = restored.replace(r'\resizebox{.46\linewidth}{!}', r'\resizebox{.52\linewidth}{!}')
restored = re.sub(r'\\subsection\{((?:Proposition|Theorem)~\\ref\{[^}]+\})\.\}',
    r'\\subsection{Proof of \1}', restored)
assert restored == original
(OUT / 'SheafCode_IRIS_Final.tex').write_text(source, encoding='utf-8')
(OUT / 'layout_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
    fromfile='22-page layout', tofile='Tighter layout')), encoding='utf-8')
print('Applied only float spacing, appendix heading spacing and Figure 3 scaling.')
