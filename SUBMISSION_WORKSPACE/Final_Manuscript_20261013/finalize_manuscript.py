from pathlib import Path
import hashlib,json,re,difflib

out=Path(__file__).resolve().parent
source=Path(r'C:\Users\rishi\.codex\attachments\d4bc0e0f-42be-49ea-8ac8-e9f1e605d8b7\Pasted text.txt')
raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()=='8b78d30a7886342cdb1169074473f2365b00a26f31c5f01397e9d5d5548ef68a'
(out/'Original_Selected_Manuscript.tex').write_bytes(raw)
original=raw.decode('utf-8-sig');text=original;changes=[]
def fix(name,before,after,reason):
    global text
    assert text.count(before)==1,(name,text.count(before))
    changes.append(dict(change=name,original_line=original[:original.index(before)].count('\n')+1,before=before,after=after,reason=reason))
    text=text.replace(before,after)

fix('adaptive_query_frequency',
    'Our implementation queries each station once under a search limit,',
    'Our implementation queries each station at most once under a search limit,',
    'Adaptive querying may stop before querying all stations; it forbids repeat queries. All published feasible adaptive settings completed without search-limit outcomes.')
fix('certification_simulation_scope',
    'the real-data tests do not demonstrate positive-alert certification.',
    'these test cases do not demonstrate positive-alert certification.',
    'Archived target truth is real, but outages and some surviving readings are deliberately modified. Describe the 272 cases as test cases rather than unmodified field trials.')
fix('discussion_simulation_scope',
    'it has not demonstrated positive-alert certification on those real-data cases.',
    'it has not demonstrated positive-alert certification on those test cases.',
    'Use the same accurate simulation scope in the discussion.')
fix('interval_pollutant_scope',
    'Across 49,996 external pollutant readings,',
    'Across 49,996 external PM2.5 readings,',
    'This coverage analysis evaluates PM2.5 only, rather than the normalized multichannel reconstruction population.')
fix('false_alarm_threshold_equality',
    '9.6\\% of 4,012 below-threshold hours,',
    '9.6\\% of 4,012 hours at or below the threshold,',
    'The negative class is truth <= 60, including eight observations exactly at 60. The denominator and false-alarm rate stay unchanged.')
fix('heatmap_detector_capacity',
    'The heatmap includes injected-fault localization and separate reconstruction sensitivity tests in India and the US.',
    'The heatmap includes injected-fault localization and separate reconstruction sensitivity tests in India and the US. The detector reports at most two faulty stations, so exact localization of three injected faults is impossible.',
    'All zero-valued three-fault cells follow from the configured maximum of two reported faults. This capacity limit must accompany the restored heatmap to prevent misleading interpretation.')

fix('iris_table_of_contents', r'\end{abstract}', r'''\end{abstract}
\clearpage
\tableofcontents
\clearpage''', 'The official IRIS research-paper structure lists a table of contents. Add it after the unchanged abstract; the scientific wording is unchanged.')

fix('donoho_citation_scope',
    r'This is related to distinguishing explanations despite errors \citep{donoho2003}',
    r'This is related to uniqueness in sparse recovery \citep{donoho2003}',
    'The cited Donoho-Elad paper proves uniqueness and recovery results for sparse representations. It is related background, not a direct source for the manuscript\'s noisy-query separation guarantee.')

fix('cca_prior_work_scope',
    'To our knowledge, this particular application of CCA has not been studied before.',
    r'CCA has also been used to learn sheaf maps for transformer representations \citep{grant2026gluing}. Our application concerns air quality station outages.',
    'A primary 2026 workshop paper by Grant and Wang explicitly evaluates CCA restriction maps. Avoid an ambiguous priority claim about CCA-based sheaf maps; identify this manuscript\'s application instead.')

fix('cca_weights_and_temporal_fitting',
    r'''\end{enumerate}

Stacking all station readings gives a vector $x$.''',
    r'''\end{enumerate}

For both spatial and temporal comparisons, CCA uses complete paired training vectors and covariances regularized by adding $0.001I$ before whitening. All $p$ canonical directions are retained. If $\gamma_j$ is a canonical correlation, clipped to $[0.001,0.999]$, its spatial comparison row is multiplied by $\sqrt{\gamma_j/(1-\gamma_j)}$. Temporal rows use $\sqrt{\theta\gamma_j/(1-\gamma_j)}$, where $\theta$ is the selected temporal scale. Temporal maps are fitted to training pairs at the corresponding lag. When fewer than 200 complete pairs are available, the projection maps are identity matrices with fixed correlations of 0.3 for spatial comparisons and 0.5 for temporal comparisons.

Stacking all station readings gives a vector $x$.''',
    'Document the actual regularized CCA construction, retained directions, correlation-dependent weights and temporal fallback. These weights matter to the identity ablation.')

fix('prior_dimensions_across_hours',
    'We regularize each covariance so it can be inverted and combine the inverses into a positive-definite matrix $P$ used at every hour.',
    r'We add $0.001I$ to each covariance before inverting it, using an identity matrix when fewer than $p+3$ complete training vectors are available. We repeat these blocks at every hour to form a positive-definite matrix $P$ over the prediction window.',
    'Clarify the dimensions of P in the spatiotemporal objective and the implemented covariance fallback. The decoder equations and proofs remain unchanged.')

fix('matched_hidden_coordinates',
    'All models predict identical hidden values for matched comparison.',
    'All models are evaluated on the same hidden readings for matched comparison.',
    'The hidden truth coordinates are identical across methods; their predictions are different.')

fix('grin_information_window',
    'GRIN can access readings after the outage, so we report it only as a retrospective comparison.',
    'GRIN can access later surviving-station readings within the outage window, so we report it only as a retrospective comparison.',
    'The implemented GRIN and retrospective sheaf windows end at the last outage hour. They use future surviving-station readings within that window, not target readings after the outage.')

fix('identity_ablation_weight_scope',
    r'Dropping the 24-hour temporal comparison hurts Indian accuracy, but replacing CCA maps with identity maps improves it: CCA raises RMSE by 5.71\%.',
    r'Dropping the 24-hour temporal comparison hurts Indian accuracy, but replacing the weighted CCA spatial maps with unweighted identity maps improves it: the weighted CCA setting has 5.71\% higher RMSE. This replacement changes both the maps and their spatial weights.',
    'The ablation removes correlation-dependent spatial weights as well as CCA projections, so the difference cannot be attributed to CCA alone.')

fix('discussion_identity_setting_scope',
    'The identity-map experiment also shows that the more elaborate CCA maps do not always help.',
    'The identity-map experiment also shows that the weighted CCA setting does not always help.',
    'Keep the discussion consistent with the ablation that changes both projections and spatial weights.')

fix('fault_detector_method_account',
    'These tests are distinct from outage reconstruction.',
    r'''These tests are distinct from outage reconstruction. We whiten the spatial comparison residuals using their training mean and covariance, adding $0.01I$ to that covariance. For each fully observed candidate support of zero, one or two stations, we project the residual onto the complement of that support's comparison columns and retain the least squared residual at each support size. A multinomial classifier uses the log residuals and their normalized decreases to choose the fault count. It is fitted on injected examples in the selection period, with ridge weight chosen from 0.1, 1 and 10. The clean-calibration cutoff gates the alarm. Missing context is filled with its training mean for this separate detector.''',
    'Provide the actual residual projection, support search, classifier fitting and missing-context treatment behind the retained fault-localization results.')

fix('snapshot_interval_method_account',
    r'''In separate single-hour tests, earlier absolute PM2.5 errors set pooled or station-specific intervals targeting 90\% coverage. We call readings above the training 95th percentile \emph{extreme} and group predicted risk relative to the training 75th percentile.''',
    r'''In separate single-hour tests, we use the spatial decoder without temporal comparisons. Absolute PM2.5 errors from the calibration period set symmetric pooled or station-specific intervals targeting 90\% coverage. For $m$ calibration errors, the radius is the $\lceil0.9(m+1)\rceil$-th smallest error, or infinity when that rank exceeds $m$. We call readings above their station's training 95th percentile \emph{extreme}. Risk groups compare the larger of the sheaf and city-mean predictions with that station's training 75th percentile; each station/risk group has its own calibrated radius.''',
    'Identify the spatial snapshot estimator, exact calibration order statistic, infinite-radius case and actual prediction-time risk grouping. These are not intervals for the main temporal outage decoder.')

fix('practical_alert_calibration_method',
    'Hidden truths never update the intervals. Measured coverage does not validate the separate error assumptions defining $Z$.',
    r'''Hidden truths never update the intervals. Measured coverage does not validate the separate error assumptions defining $Z$.

For the separate approximate alert test, actual hourly PM2.5 above 60 $\mu$g/m$^3$ defines an exceedance. Each model's alert cutoff is the 95\% finite-sample quantile of its predictions on calibration hours whose true PM2.5 is at or below 60. The Indian cutoffs are 56.08 $\mu$g/m$^3$ for SheafCode and 66.82 $\mu$g/m$^3$ for VAR. These cutoffs are fixed before evaluating test alerts.''',
    'Saved alert flags compare predictions to model-specific calibration cutoffs, while the true class uses 60. Document the actual operating rule and its calibration-only selection.')

fix('practical_alert_truth_vs_operating_threshold',
    r'''These tests ask how often estimated readings cross an illustrative hourly PM2.5 threshold of 60 $\mu$g/m$^3$; they are separate from mathematical certification.''',
    r'''These tests measure detection of actual hourly PM2.5 exceedances above 60 $\mu$g/m$^3$ using the model-specific cutoffs fitted on calibration data; they are separate from mathematical certification.''',
    'Correct the attribution of the reported sensitivities and false-alarm rates. Their values are unchanged.')

fix('context_dropout_probability',
    r'''and the fraction of missing context (0\%, 20\% or 50\%).''',
    r'''and the probability of withholding each surviving station--pollutant channel (0\%, 20\% or 50\%) throughout the window.''',
    'The sensitivity code samples one Bernoulli mask per surviving station/pollutant coordinate and applies it to the whole window. The percentages are dropout probabilities, not exact fractions of withheld readings.')

fix('joint_hidden_target_scope',
    'for 272 cases total, using only spatial comparisons in $D$.',
    'for 272 cases total, using only spatial comparisons in $D$. The joint score uses all pollutant channels at the hidden target station.',
    'The 272-case checker scores the hidden station\'s pollutants, not the maximum pollution score over every station in the network.')

fix('var_observation_noise_variance',
    'Ridge is 1 or 100, noise scale is 0.05 or 0.5, and covariance shrinkage is 0, 0.2 or 1.',
    'Ridge is 1 or 100, observation-noise variance is 0.05 or 0.5, and process-covariance shrinkage is 0, 0.2 or 1.',
    'The VAR filter adds the noise parameter directly to the observation covariance, so it is a variance rather than a standard-deviation scale. Shrinkage applies to the process covariance.')

fix('abstract_causality_scope',
    'Predictions use only information available at the time.',
    'Predictions use no observations from later hours.',
    'The offline archive experiments establish causality by observation timestamp, not an as-of replay of provider release times or historical data revisions.')

fix('causal_observation_timestamp_scope',
    "For each outage hour, the decoder uses only readings available by then: the affected station's history before the outage and readings from stations still operating (Figure~\\ref{fig:causal-workflow}). It never uses hidden readings or future observations.",
    "For each outage hour, the decoder uses only readings indexed at or before that hour: the affected station's history before the outage and readings from stations still operating (Figure~\\ref{fig:causal-workflow}). It never uses hidden readings or future observations. Causality here is by observation timestamp; provider release delays and later data revisions are not replayed.",
    'The study uses downloaded historical observations. Clarify the exact tested causality definition without claiming that real-time release/version histories were reconstructed.')

fix('remove_unconfirmed_teacher_acknowledgement',
    'We are grateful to our school computer science teacher for guidance. ',
    '',
    'The author explicitly confirmed that this acknowledgement is not factually accurate and requested removal.')

fix('grant_cca_primary_reference',
    r'''\bibitem[Hansen and Ghrist(2019)]{hansen2019spectral}''',
    r'''\bibitem[Grant and Wang(2026)]{grant2026gluing}
Bryce Grant and Peng Wang.
\newblock Gluing local contexts into global meaning: A sheaf-theoretic decomposition of transformer representations.
\newblock In \emph{ICLR 2026 Workshop on Unifying Concept Representation Learning}, 2026.
\newblock URL \url{https://bryceag11.github.io/files/sheaf_paper.pdf}.

\bibitem[Hansen and Ghrist(2019)]{hansen2019spectral}''',
    'Add the checked primary paper for the newly clarified CCA-related-work statement. The author-hosted paper and project bibliography identify the authors, title and workshop.')

for name,before,after in [
    ('caqm_page_title', 'Graded response action plan: Regional and forecast-based invocation,', 'CAQM announces revised GRAP schedule for Delhi-NCR in wake of the upcoming winter season,'),
    ('aurn_page_title', 'Automatic urban and rural network: hourly observations and quality\n  verification, 2026.', 'Air pollution monitoring: Automatic Urban and Rural Network (AURN), 2026.'),
    ('nsw_api_page_title', 'Air quality api: official data services and hourly observation\n  schema,', 'Air quality API,'),
    ('nsw_explorer_page_title', 'Air quality data explorer: Australian eastern standard time\n  convention,', 'Air quality data explorer,'),
    ('openair_page_title', 'Uk air quality data import: timestamps, units and ratification\n  metadata,', 'Import data from individual UK Air Pollution Networks,'),
    ('openaq_page_title', 'Original measurement archive and public aws access.', 'About Open Data on AWS.'),
    ('epa_page_title', 'Airdata hourly concentration files and data formats.', 'Pre-Generated Data Files.'),
]:
    fix(name,before,after,'Use the actual title of the already verified webpage, rather than a descriptive label; the cited URL is unchanged.')

fix('epa_format_link_clickable',
    'documentation at https://aqs.epa.gov/aqsweb/airdata/FileFormats.html;',
    r'documentation at \url{https://aqs.epa.gov/aqsweb/airdata/FileFormats.html};',
    'Make the separately checked EPA format-documentation URL clickable and safely typeset.')

fix('epa_parameter_inventory',
    '2025--2026 observations; parameters 88101, 44201 and 42602;',
    '2025--2026 observations; parameters 88101, 88502, 44201 and 42602;',
    'The expanded EPA data use parameter 88502 at ten selected station memberships, as recorded in the dataset manifests. Include it in the source inventory alongside the original-network parameters.')

target=out/'SheafCode_IRIS_Final_Manuscript.tex';target.write_text(text,encoding='utf-8')
(out/'FINAL_CHANGES.json').write_text(json.dumps(changes,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(out/'FINAL_CHANGES.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='Original_Selected_Manuscript.tex',tofile=target.name)),encoding='utf-8')
print(json.dumps(dict(changes=len(changes),original_sha256=hashlib.sha256(raw).hexdigest(),final_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),abstract_words=len(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',text,re.S)[1].split())),indent=2))
