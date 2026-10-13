                                                                                    
from pathlib import Path
import re, json, shutil, hashlib

root = Path('SUBMISSION_WORKSPACE')
source = root / 'Source/manuscript.tex'
backup = Path('audit_tools/readability_backup')
backup.mkdir(exist_ok=True)
before = source.read_text(encoding='utf-8')
if not (backup / 'manuscript.tex').exists():
    shutil.copy2(source, backup / 'manuscript.tex')
    shutil.copy2(root / 'Manuscript_For_Upload.pdf', backup / 'Manuscript_For_Upload.pdf')
else:
    raise RuntimeError('Backup exists: do not apply an editorial revision twice.')

replacements = {
    'We develop a decision-first monitoring framework: reconstruct missing readings, determine whether every admissible explanation gives the same pollution decision, and select additional measurements when ambiguity remains.':
    'The approach follows three questions. What missing readings fit the available information? Do all of those explanations give the same pollution decision? If they disagree, which additional stations should we measure?',
    'Its mathematical contribution links joint threshold certificates, corruption-aware repair, an exact continuous-box stability radius and adaptive query policies.':
    'The mathematical results connect these questions: first certify a joint threshold decision, then choose measurements that can withstand corrupted readings, then find how far the declared scenarios may change and how each response can guide the next query.',
    'The computational model fits noisy data. The following exact statements explain its information limits; they do not promise accurate real-world recovery. We begin with graph information, ask what decision survives bounded fault ambiguity, then choose the measurements needed when that decision remains unresolved.':
    'The fitted model works with noisy observations. The proofs below answer a different question: what can the available information establish under explicit assumptions? They do not promise accurate real-world recovery. The progression follows one monitoring problem. First ask what a surviving station determines. Next ask whether a pollution decision is still certain when readings are ambiguous. Finally ask which measurements would resolve the remaining uncertainty.',
    'Training alone determines normalization, graph construction and learned parameters. Selection chooses hyperparameters; a separate calibration block determines interval or alarm cutoffs.':
    'Each block has a separate job. Training sets the normalization, builds the graph and fits the model. Selection chooses its hyperparameters. Calibration sets the interval or alarm cutoffs.',
    'Here $x(z)$ is the full vector with $z$ inserted at missing coordinates. The first term penalizes inconsistent comparisons; the second regularizes implausible states.':
    'Here $x(z)$ means the complete collection of readings, with $z$ filling the missing coordinates. The first term charges for disagreement across comparisons. The second discourages states that are implausible under the training-fitted prior.',
    'This is the reduction in root mean station-day MSE, not average daily percentage improvement or pooled-cell RMSE.':
    'To read this percentage, first average the squared errors within each station-day, then average across station-days and take the square root. The reported gain compares those roots. It is not an average of daily percentage improvements or a pooled-cell RMSE.',
    'The joint-decision argument uses bounds that must hold for \\emph{every} feasible explanation. A computer may find useful inequality weights whose coefficients match the desired score only approximately. A small leftover coefficient can matter when a variable has a large permitted range. We can repair the bound using multiplication, addition and interval endpoints, without assuming that a solver\'s tolerance is a proof.':
    'A decision certificate must hold for \\emph{every} feasible explanation. The computer may find almost the right weights for combining inequalities, but an almost-correct coefficient is not an exact bound. Even a small mismatch can matter if a variable is allowed to be large. The next proposition accounts for that mismatch using multiplication, addition and interval endpoints. A solver\'s tolerance alone is not a proof.',
    'The contribution is witness-driven, minimum-cost decision repair with explicit impossibility outcomes. Credited multicover and error-correction theory supplies its foundations; the evaluated monitoring formulation connects these to pollution decisions.':
    'The repair procedure uses a pair of conflicting explanations to show which measurements are still needed. It finds a minimum-cost plan or explicitly reports that no plan can work. Its foundations come from the credited multicover and error-correction theory; the evaluated contribution applies them to pollution decisions.',
}
after = before
for old, new in replacements.items():
    assert after.count(old) == 1, old
    after = after.replace(old, new)

                                                                             
                                                                         
pieces = re.split(r'(\\begin\{(?:abstract|proposition|theorem|equation|table|figure)\}.*?\\end\{(?:abstract|proposition|theorem|equation|table|figure)\}|\\\[.*?\\\])', after, flags=re.S)
splits = 0
for n in range(0, len(pieces), 2):
    paras = pieces[n].split('\n\n')
    for i, para in enumerate(paras):
        if len(para.split()) < 95 or '\\textbf{Proof.}' in para or '\\begin' in para or '\\end' in para or para.lstrip().startswith('%'):
            continue
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|\\(?:textbf|emph))', para)
        if len(sentences) < 3:
            continue
        groups, group, count = [], [], 0
        for sentence in sentences:
            group.append(sentence)
            count += len(sentence.split())
            if count >= 65:
                groups.append(' '.join(group)); group = []; count = 0
        if group:
            if len(group) == 1 and groups and count < 22:
                groups[-1] += ' ' + group[0]
            else:
                groups.append(' '.join(group))
        if len(groups) > 1:
            paras[i] = '\n\n'.join(groups); splits += len(groups)-1
    pieces[n] = '\n\n'.join(paras)
after = ''.join(pieces)

norm = lambda s: re.sub(r'\s+', ' ', s).strip()
def extract(s, pattern):
    return [norm(x) for x in re.findall(pattern, s, re.S)]
patterns = {
    'formal_statements': r'\\begin\{(?:proposition|theorem)\}.*?\\end\{(?:proposition|theorem)\}',
    'proofs': r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}',
    'displayed_math': r'\\\[.*?\\\]|\\begin\{equation\}.*?\\end\{equation\}',
    'inline_math': r'(?<!\\)\$[^$]*?(?<!\\)\$',
    'tables': r'\\begin\{table\}.*?\\end\{table\}',
    'figures_and_captions': r'\\begin\{figure\}.*?\\end\{figure\}',
    'abstract': r'\\begin\{abstract\}.*?\\end\{abstract\}',
    'citations': r'\\cite\w*\{[^}]*\}',
    'references': r'\\(?:ref|label)\{[^}]*\}',
}
checks = {}
for name, pattern in patterns.items():
    original, revised = extract(before, pattern), extract(after, pattern)
    assert original == revised, name
    checks[name] = {'unchanged': True, 'count': len(original)}
numbers = lambda s: re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?', s)
assert numbers(before) == numbers(after), 'numeric tokens'
checks['numeric_tokens'] = {'unchanged': True, 'count': len(numbers(before))}
source.write_text(after, encoding='utf-8')
report = {
    'status': 'passed', 'date': '2026-10-08',
    'scope': 'Editorial explanations and paragraph structure only; no scientific content, mathematics, evidence, figures or citations removed.',
    'prose_replacements': len(replacements), 'additional_paragraph_breaks': splits,
    'checks': checks,
    'before_source_sha256': hashlib.sha256(before.encode()).hexdigest(),
    'after_source_sha256': hashlib.sha256(after.encode()).hexdigest(),
    'note': 'Existing network, causal-input and repair diagrams retained. Formal proofs and all mathematical expressions unchanged. No research rerun or new scientific claim.',
}
(root/'Guides/READABILITY_CHECK.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
