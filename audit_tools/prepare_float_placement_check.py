                                                                     
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
s = (ROOT / 'audit_tools/check_standard_definition_revision.py').read_text(encoding='utf-8')
s = s.replace('standard_definition_build', 'float_placement_build').replace('standard_definition_package_compile', 'float_placement_package_compile').replace('before_standard_definition_edit', 'before_float_placement_edit')
s = s.replace("PDF = OUT / 'SheafCode_IRIS_Final_Focused.pdf'", "PDF = OUT / 'SheafCode_IRIS_Final_Layout.pdf'")
s = s.replace("BACKUP / 'SheafCode_IRIS_Final_Tightened.pdf'", "BACKUP / 'SheafCode_IRIS_Final_Focused.pdf'")
insert = r'''
def strip_layout(s):
    return s.replace(r'\usepackage{float}', '').replace('[H]', '[htbp]').replace(r'\begin{samepage}', '').replace(r'\end{samepage}', '')
assert re.sub(r'\s+', ' ', strip_layout(new)).strip() == re.sub(r'\s+', ' ', old).strip()
'''
s = s.replace("new = SRC.read_text(encoding='utf-8')", "new = SRC.read_text(encoding='utf-8')\n"+insert)
s = s.replace("'revision': 'standard definitions removed; model-specific notation and protocols retained'", "'revision': 'layout-only float placement and passage grouping'")
s = s.replace("'edited_body_paragraphs': 24, 'causal_retrospective_definition_removed': True,", "'prose_preserved_exactly': True, 'figure_4_page': 7, 'complete_stability_example_page': 8,\n    'table_4_page': 12, 'table_5_page': 13, 'tables_precede_fault_testing': True,")
pagechecks = r'''
    assert 'Figure 4:' in texts[6]
    assert 'Consider archived states' in texts[7] and 'The first tolerates' in texts[7]
    assert 'Table 4:' in texts[11] and 'Table 5:' in texts[12]
    assert texts[12].index('Neural comparisons and component ablations') < texts[12].index('Table 5:') < texts[12].index('Fault localization and interval coverage') < texts[12].index('Equal opposite two-fault')
'''
s = s.replace('    pages = len(doc)', pagechecks+'\n    pages = len(doc)')
start = s.index("    'This revision removes textbook explanations")
end = s.index("    'Formal statements, displayed equations", start)
s = s[:start]+("    'This layout-only revision anchors Figure 4 and Tables 4-5 at their source positions. '\n"
    "    'The stability introduction and complete tolerance example stay together on page 8, '\n"
    "    'following Figure 4 on page 7. Table 4 accompanies external reconstruction comparisons '\n"
    "    'on page 12; Table 5 accompanies the neural comparison on page 13. Neither table '\n"
    "    'interrupts fault testing. The fault-testing introduction and results are grouped.\\n\\n'\n")+s[end:]
s = s.replace('standard_definition_edit.diff', 'float_placement_edit.diff')
s = s.replace('with standard definitions removed, descriptive headings and all seven ', 'with float placement refined, descriptive headings and all seven ')
(ROOT / 'audit_tools/check_float_placement_revision.py').write_text(s, encoding='utf-8')
print('Prepared layout preservation and placement checks.')
