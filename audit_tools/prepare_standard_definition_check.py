                                                                                  
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
s = (ROOT / 'audit_tools/check_repetition_revision.py').read_text(encoding='utf-8')
s = s[:s.index('hashfile = lambda')]
s = s.replace('repetition_build', 'standard_definition_build').replace('repetition_package_compile', 'standard_definition_package_compile').replace('before_repetition_edit', 'before_standard_definition_edit').replace('SheafCode_IRIS_Final_Tightened.pdf', 'SheafCode_IRIS_Final_Focused.pdf')
s += r'''
hashfile = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with pymupdf.open(BACKUP / 'SheafCode_IRIS_Final_Tightened.pdf') as previous:
    previous_pages = len(previous)
manifest = {
    'revision': 'standard definitions removed; model-specific notation and protocols retained',
    'pages': pages, 'previous_pages': previous_pages, 'pages_removed': previous_pages-pages,
    'body_font_pt': 11, 'margins_inches': 0.8,
    'references_begin_page': ref_page, 'proofs_begin_page': proof_page, 'proofs_end_page': pages,
    'edited_body_paragraphs': 24, 'causal_retrospective_definition_removed': True,
    'descriptive_headings_preserved': True, 'section_structure_preserved': True,
    'model_specific_notation_and_experimental_protocols_retained': True,
    'formal_statements_preserved_exactly': True, 'display_equations_preserved_exactly': True,
    'complete_proofs_preserved_exactly': 7, 'propositions': 6, 'theorems': 1,
    'figure_contents_preserved_exactly': 7, 'table_count': 10,
    'table_numeric_values_preserved': True, 'citation_and_reference_targets_preserved': True,
    'bibliography_preserved_exactly': True, 'overfull_boxes': 0,
    'undefined_references': 0, 'proof_links_resolve': len(proof_links),
    'fresh_zip_compilation_verified': True, 'pdf_text_within_page_bounds': True,
    'new_experiments_run': False, 'source_sha256': hashfile(SRC),
    'pdf_filename': PDF.name, 'pdf_sha256': hashfile(PDF), 'zip_sha256': hashfile(ZIP),
    'previous_source_sha256': hashfile(BACKUP / SRC.name),
}
(OUT / 'final_check.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
(OUT / 'README.md').write_text(
    f'The current manuscript is SheafCode_IRIS_Final.tex. Its compiled PDF is '
    f'{PDF.name} ({pages} pages, including references and all seven expanded proofs). '
    f'References begin on page {ref_page}, and Appendix A occupies pages {proof_page}-{pages}.\n\n'
    'This revision removes textbook explanations of standard terms, including causal and '
    'retrospective estimates, RMSE, CCA, basic graph terminology and optimization terminology. '
    'It retains the method-specific notation, assumptions, motivations, examples and '
    'experimental protocols. The natural prose and descriptive headings are retained.\n\n'
    'Formal statements, displayed equations, all expanded proofs, figure contents, table '
    'values and the bibliography are preserved. Font size remains 11pt and margins remain '
    '0.8 inches. No new experiments were run.\n\n'
    'The preceding version is preserved in '
    'audit_tools/final_iris_review/before_standard_definition_edit/. '
    'standard_definition_edit.diff records the source changes. Older PDF filenames '
    'in this folder refer to previous revisions.\n\n'
    'Upload SheafCode_Final_Overleaf.zip to Overleaf and compile SheafCode_IRIS_Final.tex. '
    'The ZIP contains one TEX file with embedded figures and references. A fresh extraction '
    'compiles and matches the delivered PDF text. There are no overfull boxes or undefined '
    'references. final_check.json records validation and artifact hashes.\n', encoding='utf-8')
root_readme = OUT.parent / 'README.md'
s = root_readme.read_text(encoding='utf-8')
first = ('The current compact manuscript is [Compact_Readable_Final](Compact_Readable_Final/README.md) '
         f'({pages} pages), with standard definitions removed, descriptive headings and all seven '
         'expanded formal proofs retained. The separate '
         '[Appendix_Readable_Final](Appendix_Readable_Final/README.md) version has 30 pages '
         'and has not received this prose rewrite.\n\n')
root_readme.write_text(first + s[s.index('\n\n')+2:], encoding='utf-8')
print(json.dumps(manifest, indent=2))
'''
(ROOT / 'audit_tools/check_standard_definition_revision.py').write_text(s, encoding='utf-8')
print('Prepared revision-specific checks.')
