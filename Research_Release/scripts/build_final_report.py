                                                                                
import ast,hashlib,json,os,sys,zipfile,re
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sheafcode.reliable import paired_bootstrap
D=ROOT/'docs';F=D/'figures';F.mkdir(exist_ok=True)
C=pd.read_csv(D/'dependence_sensitivity.csv')
W=json.loads((ROOT/'results/weather/weather_summary.json').read_text())
audit=json.loads((D/'final_audit.json').read_text())
checks=audit['checks'];assert all(x['passed'] for x in checks)
assert hashlib.sha256((ROOT/'scripts/run_weather.py').read_bytes()).hexdigest()==W['source_sha256']
lines=['# Scientific results and final numerical audit','',
 'Research reference. These are reproducible analyses of previously examined archives, not untouched external validation. No competition-selection guarantee follows from them.','',
 '## Strongest defensible finding','',
 'On the Delhi raw-concentration cohort, causal reconstruction of 24-hour station outages reduced normalized RMSE by 28.0% against a validation-tuned ridge VAR filter, 24.3% against Gaussian conditioning, and 23.0% against seasonal persistence. These are 69 paired station-outage cases across 12 sampled days, with positive day-cluster, station-cluster and ordered sampled-day block sensitivity intervals. The exact day-cluster sign analysis, with Holm adjustment across seven networks for each comparator, remains below 0.05; it assumes sign symmetry across day clusters. The test archive had already been examined, and this is internal evidence.','',
 'Bengaluru raw reconstruction improved 31.1% over VAR and 43.7% over Gaussian conditioning, but was 20.6% worse than seasonal persistence. It therefore does not establish superiority to the strongest baseline. Single-station snapshot reconstruction does not establish a general sheaf advantage over Gaussian conditioning.','',
 '## Complete causal comparisons','',
 '| Network | Comparator | Cases | RMSE gain | MSE-difference 95% day CI | Holm p |',
 '|---|---|---:|---:|---|---:|']
for _,r in C[C['mode']=='causal'].iterrows():
    lo,hi=ast.literal_eval(r.difference_ci95)
    lines.append(f'| {r.region} | {r.reference} | {r.n} | {100*r.rmse_relative_gain:.1f}% | [{lo:.3f}, {hi:.3f}] | {r.holm_across_networks_p:.4f} |')
lines += ['', 'Positive differences favor sheaf. Complete retrospective results and alternative clustering intervals are in `dependence_sensitivity.csv`. Ordered blocks contain three sampled days, not three consecutive calendar days. Overlapping regional archives are not independent replications. Seven networks completed; Mumbai and Kolkata did not pass the fixed training-only coverage rule.','',
 '## Meteorology and actual temperature forecasting','',
 'Weather-context sheaf reconstruction uses only surviving stations’ wind, fits low/high wind operators on training data, and selects gate sharpness on validation data. Its test sample contains 179 paired station-days across 60 days. Target-station pollution and weather are hidden for reconstruction. This extension did not improve over the ordinary sheaf or Gaussian model. It is an evaluated negative result, not an environmental improvement claim.','',
 '| Task | Weather model comparator | Cases / days | RMSE gain | MSE-difference 95% day CI |',
 '|---|---|---:|---:|---|']
items=[('Pollution reconstruction','ordinary sheaf',W['weather_reconstruction']),('Pollution reconstruction','Gaussian',W['weather_reconstruction_vs_gaussian'])]
for target,table in W['one_hour_forecasting'].items():
    for ref,r in table.items():items.append((f'{target} one-hour forecast',ref,r))
for task,ref,r in items:
    lo,hi=r['difference_ci95'];lines.append(f"| {task} | {ref} | {r['n']} / {r['clusters']} | {100*r['rmse_relative_gain']:.2f}% | [{lo:.4f}, {hi:.4f}] |")
lines += ['', 'Temperature forecasting used actual next-hour temperature labels, not pollution as a surrogate. All forecast methods use identical hours with the current and next-hour target observed. Weather features include temperature, humidity, wind components, rainfall and solar radiation where training coverage permits. Ridge parameters are validation-selected. The weather model improves over temperature persistence, but performs worse than its matched model without the extra weather features; the gain over persistence cannot be attributed to those features. This is not a comparison with operational numerical weather prediction.','',
 '## Corrected mathematics and scope of novelty','',
 'The full-state quadratic includes the observed–unknown prior cross term. Generic matching characterizes independent supported matrix entries; fitted CCA and tied identity maps require numerical verification. Uniform block-sparse syndrome uniqueness requires block distance greater than twice the maximum fault count. No generic graph condition guarantees small real-world reconstruction error.','',
 'For exactly t faults, noisy least-squares support recovery is sufficient when every competing support S obeys sigma_min(D_(F union S)) times ||e_(F\\S)|| > 2||r||. A uniform sufficient bound uses the smallest individual nonzero fault-block norm, not the whole fault norm. Missing equations change the operator and the bound; the experiment checks the actual observed operator. Unknown fault count and clean rejection are evaluated separately.','',
 'The weather-context extension has a conditioning-sensitivity result: for Q(a)=(1-a)L_low+a L_high+lambda P, fixed mask O/U and fixed observed vector, the change in reconstruction satisfies an exact resolvent identity and a norm bound using lambda times the smallest eigenvalue of P_UU. The derivation is included in the accompanying PDF, and 100 actual-mask identity/bound checks passed. This is a project-specific application of classical matrix perturbation, not a claim to invent resolvent identities. Structural rank, block-sparse uniqueness and adaptive feedback calibration also have established antecedents.','',
 'Wind-driven air-quality graphs already exist: [Zhou et al., dynamic wind-field forecasting](https://www.sciencedirect.com/science/article/pii/S0269749121000518), [context-conditioned air-quality graph forecasting](https://researchportal.vub.be/files/85598881/EUSIPCO22_Forecasting_model_AirPoll.pdf). The defensible contribution is the monitoring application, explicit conditioning and failure certificates, corrected causal protocol and empirical evaluation.','',
 '## Environmental interpretation','',
 'The established benefit is monitoring continuity during outages. PM2.5 threshold reconstruction is scored against observed concentrations and includes false alarms as well as misses. Delhi’s raw threshold study has 28 positive station-days: the sheaf misses 39.3% versus 64.3% for city mean, with false alarms 10.2% versus 6.8%. This is a sensitivity/specificity trade-off on a small sample, not established policy-action recovery. Bengaluru raw misses all eight positive days in the sampled threshold set, so accuracy alone must not be advertised as environmental success.','',
 'Estimated pollution on naturally missing days is not verified exposure. Repair prioritization is a retrospective proxy, not proof of reduced emissions or improved health. No experiment here establishes physical pollution reduction. The approximate threshold endpoint is not official AQI or GRAP; forecast-driven Delhi-NCR invocation cannot be inferred from individual station thresholds.','',
 '## Uncertainty and fault detection','',
 'Full and empty adaptive prediction sets implement the theorem’s boundaries. The feedback guarantee requires revealed labels; it does not supply operational adaptation during a completely unobserved outage. Fixed intervals without test labels are reported separately. Constant station scale produces identical intervals to unit scale. Actual-mask-varying scales are tested rather than assumed beneficial. Coverage, lowest station quarter and full/empty frequencies are retained, not only favorable averages.','',
 'Fault trials include clean states, unknown counts, all eligible candidates, positive/negative/random-direction faults and the actual equation mask. Known-count support accuracy is diagnostic, not end-to-end success. Read `fault_summary.csv` for alarms, clean false alarms, exact end-to-end support and conditional sufficient-bound trials. Counts, sizes and transformed-coordinate fault amplitudes must accompany every performance claim.','',
 '## Final audit and reproducibility','',
 f'{len(checks):,} numerical/source-consistency checks passed. The regression and supplied legacy suites passed 26 tests, including counterexamples to the rejected robust norm bound and finite adaptive boundary rule, the effect of missing equations on a decoder certificate, and 100 randomized partial-mask context identity/bound cases. Weather calculations add 100 actual-mask perturbation checks. This verifies the listed checks; it does not imply every scientific hypothesis is correct.','',
 'The completed core runs, independent Gaussian comparator reconstruction, alternative dependence analysis, weather results and exact source/data hashes are retained. Supplementary studies have their own hashes; the frozen core sources are preserved. Figures are generated from the saved comparison CSVs, including negative results. Raw data stay in the local release.','',
 'The compact draft and earlier claims have not received blanket scientific approval. This report supersedes their unsupported numerical/theoretical assertions; complete manuscript integration and independent external validation remain necessary before calling a submission finished.','']
(D/'SCIENTIFIC_RESULTS_AND_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(11,5))
for ax,region,title in zip(axes,['delhi2025','bengaluru_raw'],['Delhi raw concentrations','Bengaluru raw concentrations']):
    g=C[(C.region==region)&(C['mode']=='causal')]
    ax.barh(g.reference.str.replace('_',' '),100*g.rmse_relative_gain,color=['#237c88' if x>0 else '#b34e42' for x in g.rmse_relative_gain])
    ax.axvline(0,color='black',lw=.8);ax.set_title(title);ax.set_xlabel('Relative RMSE gain (%)')
fig.suptitle('Causal 24-hour outage reconstruction: all comparators')
fig.tight_layout();fig.savefig(F/'causal_all_comparators.pdf');fig.savefig(F/'causal_all_comparators.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
ax.barh([a+' vs '+b for a,b,r in items],[100*r['rmse_relative_gain'] for a,b,r in items],color=['#237c88' if r['rmse_relative_gain']>0 else '#b34e42' for a,b,r in items])
ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Relative RMSE gain (%)');ax.set_title('Meteorology ablations include unfavorable results')
fig.tight_layout();fig.savefig(F/'weather_ablations.pdf');fig.savefig(F/'weather_ablations.png',dpi=160);plt.close(fig)
                                                                               
def esc(s):
    for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
    return s.replace('–','--').replace('’',"'").replace('×',r'$\times$')
tex=[r'\documentclass[11pt]{article}',r'\usepackage[margin=1in]{geometry}',r'\usepackage{amsmath,amssymb,graphicx,longtable}',r'\usepackage[hidelinks]{hyperref}',r'\emergencystretch=3em',r'\begin{document}',r'\title{SheafCode: scientific results and verified conditioning statements}',r'\author{Research reference}',r'\date{}\maketitle']
in_table=False
for line in lines:
    if in_table and not line.startswith('|'):
        tex.append(r'\end{longtable}\endgroup');in_table=False
    if line.startswith('# '):continue
    if line.startswith('## '):tex.append(r'\section{'+esc(line[3:])+'}');continue
    if line.startswith('|---'):continue
    if line.startswith('|'):
        cells=[x.strip() for x in line.strip('|').split('|')]
        if not in_table:
            widths=[.13,.19,.07,.10,.22,.09] if len(cells)==6 else [.20,.19,.12,.12,.19]
            tex.append(r'\begingroup\setlength{\tabcolsep}{3pt}\begin{longtable}{'+''.join('p{'+str(w)+r'\linewidth}' for w in widths)+'}')
            in_table=True
        tex.append(' & '.join(esc(x).replace(r'\_',r'\_\allowbreak ') for x in cells)+r' \\')
        continue
    if line:
        parts=[];last=0
        for match in re.finditer(r'\[([^\]]+)\]\((https?://[^)]+)\)',line):
            parts.extend([esc(line[last:match.start()]),r'\href{'+match[2]+'}{'+esc(match[1])+'}']);last=match.end()
        parts.append(esc(line[last:]));tex.append(''.join(parts).replace('`','')+'\n')
tex += [r'\clearpage\section{Context conditioning proposition and proof}',
 r'Let $L_0,L_1\succeq0$, $P\succ0$, $\lambda>0$ and $Q(a)=(1-a)L_0+aL_1+\lambda P$, with $a\in[0,1]$. Fix an observation mask $O/U$ and $x_O$. Let $x(a)$ be its full reconstruction, satisfying $x(a)_O=x_O$ and $Q(a)_{UU}x(a)_U=-Q(a)_{UO}x_O$. Then',
 r'\[x(a)_U-x(b)_U=(a-b)Q(a)_{UU}^{-1}[(L_0-L_1)x(b)]_U.\]',
 r'Consequently,',
 r'\[\|x(a)_U-x(b)_U\|_2\leq\frac{|a-b|\,\|[(L_0-L_1)x(b)]_U\|_2}{\lambda\lambda_{\min}(P_{UU})}.\]',
 r'For the proof, subtract the stationarity equations, writing the second with $Q(a)$. The observed-coordinate difference is zero, so',
 r'\[Q(a)_{UU}(x(a)_U-x(b)_U)=-[(Q(a)-Q(b))x(b)]_U.\]',
 r'Insert $Q(a)-Q(b)=(a-b)(L_1-L_0)$. Since $Q(a)_{UU}\succeq\lambda P_{UU}\succ0$, its inverse norm is at most $1/(\lambda\lambda_{\min}(P_{UU}))$, proving the inequality. The bound requires a fixed mask and observed vector; it does not bound forecast error or claim that weather changes cause pollution changes.',
 r'\section{Corrected reconstruction and robust support statements}',
 r'\[\widehat x_U=-(D_U^TD_U+\lambda P_{UU})^{-1}(D_U^TD_O+\lambda P_{UO})x_O.\]',
 r'If $s=D_Fe_F+r$, $|F|=t$ with exactly $t$ nonzero blocks, and $|S|=t$, every competing least-squares fit has residual at least $\sigma_{\min}(D_{F\cup S})\|e_{F\setminus S}\|_2-\|r\|_2$. The coefficient difference necessarily retains $e_{F\setminus S}$; the true support residual is at most $\|r\|_2$. Strict separation for every wrong support therefore suffices. A uniform sufficient condition is',
 r'\[\sigma_{2t}\min_{v\in F}\|e_v\|_2>2\|r\|_2,\quad\sigma_{2t}=\min_{1\leq |T|\leq\min(2t,n)}\sigma_{\min}(D_T).\]',
 r'Use the actual retained-equation operator and actual competing candidate universe. Supersets prevent exact support recovery when fewer than $t$ faults are fitted with exactly $t$ blocks. This is a sufficient condition, not a necessary predictor of all empirical failures.',
 r'\section{Figures}',r'\includegraphics[width=\linewidth]{figures/causal_all_comparators.pdf}',r'\includegraphics[width=\linewidth]{figures/weather_ablations.pdf}',r'\includegraphics[width=\linewidth]{figures/intervals.pdf}',r'\end{document}']
(D/'SCIENTIFIC_RESULTS_AND_AUDIT.tex').write_text('\n'.join(tex),encoding='utf-8')
paths=list((ROOT/'scripts').glob('*.py'))+[ROOT/'docs/CORRECTED_THEORY.md',ROOT/'docs/SCIENTIFIC_RESULTS_AND_AUDIT.md']
(D/'supplement_manifest.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'core_check_count':len(checks)},indent=2))
with zipfile.ZipFile(ROOT/'Scientific_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for folder in [ROOT/'results',D,ROOT/'scripts',ROOT/'tests',ROOT/'sheafcode']:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix not in ['.npz','.pyc'] and '__pycache__' not in str(p):z.write(p,p.relative_to(ROOT))
print(D/'SCIENTIFIC_RESULTS_AND_AUDIT.md',flush=True)
