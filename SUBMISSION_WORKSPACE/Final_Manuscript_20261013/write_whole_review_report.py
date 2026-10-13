from pathlib import Path
import json,re,hashlib,difflib

OUT=Path(__file__).resolve().parent
texpath=OUT/'SheafCode_IRIS_Final_Manuscript.tex';tex=texpath.read_text(encoding='utf-8')
original=(OUT/'Original_Selected_Manuscript.tex').read_text(encoding='utf-8-sig')
changes=json.loads((OUT/'FINAL_CHANGES.json').read_text(encoding='utf-8'))
evidence=json.loads((OUT/'EVIDENCE_AUDIT.json').read_text());details=json.loads((OUT/'FINAL_DETAIL_CHECKS.json').read_text())
inference=json.loads((OUT/'WHOLE_REVIEW_CHECKS.json').read_text());followup=json.loads((OUT/'WHOLE_FOLLOWUP_REPLAY.json').read_text())
joint=json.loads((OUT/'WHOLE_JOINT_REPLAY.json').read_text());archive=json.loads((OUT/'WHOLE_ARCHIVE_REPLAY.json').read_text());fault=json.loads((OUT/'WHOLE_FAULT_REPLAY.json').read_text())
pdf=json.loads((OUT/'PDF_VALIDATION.json').read_text());downloads=json.loads((OUT/'FINAL_DOWNLOADS.json').read_text())
parse_log=lambda name:next(json.loads(line) for line in (OUT/name).read_text().splitlines() if line.startswith('{'))
repair=parse_log('whole_repair_optima.log');rank=parse_log('whole_exact_rank.log')
assert all(v['status']=='passed' for v in [evidence,details,inference,followup,joint,archive,fault,pdf,repair,rank])
original_words=original.split();final_words=tex.split();matcher=difflib.SequenceMatcher(None,original_words,final_words,autojunk=False)
retained=sum(block.size for block in matcher.get_matching_blocks());preservation=100*retained/len(original_words)
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',tex,re.S)[1]
record={'status':'completed_with_corrections','source_sha256':hashlib.sha256(texpath.read_bytes()).hexdigest(),'documented_changes':len(changes),'original_tex_tokens':len(original_words),'final_tex_tokens':len(final_words),'aligned_original_token_retention_percent':preservation,'abstract_words':len(abstract.split()),'references':30,'formal_mathematics_unchanged':True,'main_result_values_unchanged':True,'checks':{'final_evidence':evidence['check_count'],'final_details':details['check_count'],'fresh_inference':inference['checks'],'fresh_followup':followup['check_count'],'joint':joint['checks'],'archive':archive['checks']},'replayed_station_days':inference['all_replayed_station_days'],'repair':{k:repair[k] for k in ['fixed_configurations','radius_frontiers','radius_budget_points','adaptive_policies']},'teacher_acknowledgement':'Removed at the author\'s explicit factual correction.','download_files':downloads['final_files']}
(OUT/'WHOLE_REVIEW_SUMMARY.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')

report=f'''# Whole-manuscript review

Completed 13 October 2026 on the author's selected manuscript and its saved research evidence. This review found substantive description and attribution errors missed by the earlier confirmation. The corrected manuscript now passes the checks below. The conclusion is scoped to the reviewed manuscript, recorded computations and stated mathematical assumptions; it is not a guarantee against every possible undiscovered error.

## Corrections and necessary clarifications

| Topic | What was corrected | Basis |
|---|---|---|
| Practical alerts | The 60-unit threshold defines recorded true exceedance; predictions use separately calibrated operating cutoffs, 56.08 for SheafCode and 66.82 for VAR. | `environment.py`, `alert_cutoffs.json`, and every saved alert flag. The reported rates are unchanged. |
| GRIN inputs | Future surviving readings come from within the outage window, not after it. | Both neural and retrospective sheaf inputs span 24 history hours and 24 outage hours. |
| CCA ablation | The identity replacement removes spatial weights as well as projections. The 5.71% difference describes the two complete settings. | `extensions.py` and saved ablation cases; it cannot isolate CCA alone. |
| CCA priority | Replaced the ambiguous prior-use claim and cited existing CCA sheaf-map work. | [Grant and Wang's primary paper](https://bryceag11.github.io/files/sheaf_paper.pdf) and [project bibliography](https://cwru-aism.github.io/gluing-lc-page/). |
| Reconstruction details | Specified CCA regularization/weights/fallbacks and the repeated covariance blocks forming P. Specified VAR observation-noise variance. | Training-only map reconstruction and independent spatiotemporal incidence-row replay. |
| Fault method | Added the whitened residual projections, candidate supports, classifier fitting, calibration gate and mean-filled context. | The detector was rebuilt and all 8,712 Indian and 9,072 US episode rows reproduced. |
| Prediction intervals | Identified the spatial snapshot estimator, calibration rank, infinite-radius case, station-specific extrema and actual risk grouping. | Fresh error-quantile and coverage checks. These are separate from the temporal outage decoder. |
| Matched observations | Replaced the statement that methods predict identical values with evaluation on the same hidden readings. | All seven methods have matched target coordinates and distinct predictions. |
| Context loss | The percentages are probabilities of dropping each surviving station/pollutant channel for the full window. | The actual Bernoulli mask construction. They are not fixed exact fractions. |
| Joint score | Explicitly identified the hidden target station as the scored station in the 272-case experiment. | Reconstructed H and offsets, station/channel units, and stored exact certificates. |
| Causality | Defined it by observation timestamp, without claiming a replay of provider publication delays or revisions. | The archive-based input construction. Only one abstract sentence changes, retaining all its results. |
| EPA channels | Added parameter 88502 to the source inventory. | Expanded manifests contain ten selected station memberships using it. |
| References | Narrowed the Donoho citation, matched documentation titles to their webpages and made the EPA format URL clickable. | Primary records and the source-by-source reference review. |
| Teacher acknowledgement | Removed. | The author confirmed that the teacher did not provide the stated guidance. |

All {len(changes)} changes relative to the original selected source, including the earlier edits and table of contents, are recorded in `FINAL_CHANGES.json` and `FINAL_CHANGES.diff`. No audit/result inventory was added to the paper. The removed hybrid, direct-reading-guard and historical experiment families stay removed.

## Verification completed

- Independently reconstructed all seven inference methods on **{inference['all_replayed_station_days']:,} station-days**, including the original networks, every evaluable external group and the longer Indian split. Normalized and physical losses match the saved values. Rebuilt the fitted CCA maps/prior from training data and recomputed model-selection minima, chronology, normalized tensors and paired intervals: **{inference['checks']:,} checks**.
- Replayed forest checkpoints, independent feature construction, per-hour hidden/future-value invariance, interval arithmetic/calibration and selection choices: **{followup['check_count']:,} checks**.
- Rebuilt the separate original-network fault detector and reproduced every retained injected-fault episode outcome; exact support, count, alarm and score fields agree.
- Replayed **272 joint cases and two constructed controls**, with every support and endpoint certificate, using exact rational bounds: **{joint['checks']:,} checks**. This verifies stored coefficients and mathematical certificates, not physical error assumptions.
- Independently optimized **414 fixed configurations, 122 radius frontiers / 594 budget points, and 76 adaptive policies**. The 25% finite-bank query result and strict-boundary logic remain valid on the declared banks. Exact modular rank checks cover 23 operators and 246 endpoint blocks.
- Independently checked archive bank/calibration inputs, membership and the saved stress trials. The verifier examines every saved direct guard outcome and samples archive-plan inference every 313th trial, alongside exact separation checks for every feasible plan. Its aggregate outside-model errors agree with the manuscript. The omitted direct-guard results remain omitted.
- Final manuscript checks pass: **191 evidence/structure checks and 135 detailed checks**, including all 100 heatmap cells, all 76 radius points, table rows, plotted headline values and all cited keys.
- Reviewed all six propositions, the stability theorem, seven proofs, decoder equations and worked examples under their stated hypotheses. The formal blocks and numbered equations remain unchanged from the selected source. No additional error was found in those reviewed arguments.
- Checked all **30 real references**, their supplied identifiers/destinations and nearby citation roles. See `REFERENCE_REVIEW.md` for publisher-access qualifications and the additional prior-work source.
- Compiled successfully to **{pdf['pages']} pages** with no undefined citations/references, missing glyphs or overfull boxes. All-page previews were inspected; document design and figures retain the selected presentation.

The core results remain 16.48% lower normalized RMSE and 10.45% lower physical PM2.5 RMSE on the fixed Indian benchmark, and 25% fewer finite-bank queries over the ten jointly feasible settings. Negative external comparisons, poor extreme coverage, fault failures, all 40 positive-truth joint abstentions and archive-transfer failures remain disclosed.

## Preservation and remaining limits

The word-token alignment retains {preservation:.2f}% of the original TeX tokens. This is a mechanical preservation measure, not a claim about stylistic authorship. The added text supplies needed method details and corrected claim scope. The abstract is {len(abstract.split())} words and differs only in its causality sentence. Introduction, research questions, conclusion, numerical tables, all formal proofs and figure data retain their selected wording/content apart from the specifically documented edits.

The review does not independently retrain every VAR, random forest or neural model, formally verify proofs in a proof assistant, establish worldwide first-ever priority, validate instrument-error limits, recover original provider release histories, reproduce the study in an independent laboratory, or supply a field deployment. Calibration-derived ranges do not guarantee future model membership. These limits are consistent with the manuscript's conditional claims and discussion.

## Current files

- Editable final source: `SheafCode_IRIS_Final_Manuscript.tex` in this folder and Downloads. The `(4).tex` working copy is synchronized.
- Latest checked Downloads PDF: `{Path(next(p for p in downloads['final_files'] if p.endswith('.pdf'))).name}`. The standard-named old PDF was locked; it must not be mistaken for this checked copy.
- Source SHA-256: `{record['source_sha256']}`.
- PDF SHA-256: `{pdf['pdf_sha256']}`.

Verification scripts, full edit records and check logs are kept outside the manuscript. Saved research inputs and outcome tables were not altered.
'''
(OUT/'WHOLE_MANUSCRIPT_REVIEW.md').write_text(report,encoding='utf-8')
confirmation=f'''# Final manuscript confirmation

Completed the renewed whole-manuscript review on 13 October 2026. It found and corrected additional method-description and citation-scope errors that the earlier confirmation missed.

**The final manuscript passes the completed mathematical, implementation, numerical, citation and document checks within the stated review scope.** See [the complete review](WHOLE_MANUSCRIPT_REVIEW.md) for findings, evidence and limits. This is not an unconditional guarantee that every possible error has been eliminated.

The main corrections distinguish the actual alert threshold from calibrated operating cutoffs; correctly describe GRIN's within-outage future inputs; identify the weighted CCA versus unweighted identity comparison; credit earlier CCA sheaf-map work; specify the spatial snapshot interval and fault-detection methods; make P's dimensions and VAR noise variance explicit; distinguish dropout probabilities and observation-time causality; identify the hidden-target score; and complete the EPA parameter inventory. The teacher acknowledgement was removed at the author's explicit factual correction. Reference titles and citation scope were also checked.

All six propositions, the stability theorem, seven proofs and both numbered equations remain unchanged. No reported outcome value, table row or plotted datum was changed. The abstract changes only its causality sentence and is now {len(abstract.split())} words. The selected prose and presentation were preserved through targeted edits; removed hybrid, direct-guard and historical results were not restored, and no audit tables were added to the manuscript.

Fresh checks include {inference['all_replayed_station_days']:,} station-days / seven independently assembled inference methods, {inference['checks']:,} inference/data/calibration checks, {followup['check_count']:,} follow-up checks, every retained original-network fault episode, {joint['checks']:,} exact joint checks, 414 fixed configurations, 122 radius frontiers, 594 budget points, 76 adaptive optima, 30 references, and the final 191 evidence plus 135 detail checks. Successful compilation produces {pdf['pages']} pages without unresolved citations or overflow warnings.

The review does not establish field-valid instrument-error assumptions, future archive membership, universal superiority, unrestricted mathematical priority, independent laboratory reproduction or provider-release timing. The paper states its conditional scope and negative findings.

Exact edits: `FINAL_CHANGES.json` and `.diff` ({len(changes)} records relative to the original selected attachment). The table of contents remains present for the listed IRIS research-paper structure.

Final editable source: `SheafCode_IRIS_Final_Manuscript.tex`, synchronized with Downloads and the `(4).tex` working copy.

Latest checked Downloads PDF: `{next(p for p in downloads['final_files'] if p.endswith('.pdf'))}`. The standard-named previous Downloads PDF was locked; use this whole-checked copy or the current workspace PDF.

Original SHA-256: `8b78d30a7886342cdb1169074473f2365b00a26f31c5f01397e9d5d5548ef68a`.

Final SHA-256: `{record['source_sha256']}`.
'''
(OUT/'FINAL_CONFIRMATION.md').write_text(confirmation,encoding='utf-8')
ready=OUT/'SUBMISSION_READINESS.md';s=ready.read_text(encoding='utf-8');s=s.replace('The abstract is 235 words','The abstract is 234 words').replace('all 29 sources','all 30 sources').replace('all 29 references','all 30 references')
s=re.sub(r'Use `SheafCode_IRIS_Final_Manuscript[^`]*\.pdf` from Downloads;[^\n]*',f'Use `{Path(next(p for p in downloads["final_files"] if p.endswith(".pdf"))).name}` from Downloads; the standard-named PDF was locked during the update. The `(4).tex` working copy is synchronized with the final editable manuscript.',s)
s=re.sub(r'The final reference review verified[^\n]*','The whole-manuscript review corrected additional substantive descriptions, removed the teacher acknowledgement at the author\'s request and verified all 30 references. See `WHOLE_MANUSCRIPT_REVIEW.md` and `REFERENCE_REVIEW.md`.',s)
ready.write_text(s,encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ['repair','checks','download_files']},indent=2))
