                                                                               
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=ROOT/'docs';F=D/'figures'
R=json.loads((D/'sensor_repair_summary.json').read_text());I=json.loads((D/'interval_replay_audit.json').read_text())
assert all(c['passed'] for c in R['checks']+I['checks'])
assert R['source_sha256']==hashlib.sha256((ROOT/'scripts/run_sensor_repair.py').read_bytes()).hexdigest()
assert I['source_sha256']==hashlib.sha256((ROOT/'scripts/replay_interval_details.py').read_bytes()).hexdigest()
lines=['# Technical completion and additional mathematical experiment','',
 'Completed 6 October 2026. Code and experiments only; manuscript unchanged. The core run is complete and does not need restarting overnight. The extended launcher can reproduce the supplements, but no overnight job is required to recover these completed results.','',
 '## Completed additions','',
 '- Exact block sensor-repair covariance updates and weighted one-repair optimization implemented in `sheafcode/acquisition.py`. Invalid weights, duplicate coordinates and singular submatrices are rejected.',
 '- Three added tests validate direct conditioning, positive-semidefinite covariance reduction, weighted trace identity and candidate optimality over randomized matrices.',
 '- Real-data theorem test no longer silently returns on data-loading failure and cannot pass with zero observed vectors.',
 '- All 29 regression/legacy tests pass.',
 f"- {sum(c['one_repair_model_optimality_checks'] for c in R['checks']):,} artificial two-station repair cases check exact one-repair model optimality.",
 f"- {len(I['checks'])} independent station/setting interval replay checks reproduce original coverage, cell counts and full/empty-set counts.",
 '- Pollutant-specific finite interval widths, actual calendar-quarter coverage and quarter sample sizes are now saved in each network’s `intervals_by_pollutant.csv`. The original `worst_calendar_quarter` field actually represented four equal-duration test-timeline segments. That field is preserved for reproducibility, but its meaning is corrected here; the new actual-quarter column is separate.',
 '- Frozen protocol wording promised seven-day block sensitivity. The implemented supplement uses three ordered sampled-day blocks, not seven calendar days. Existing inference is described accordingly; no seven-day analysis is claimed.','',
 '## Mathematical statement: exact value of a repaired station','',
 'For a positive-definite frozen precision Q and currently unknown coordinates U, define C=(Q_UU)^(-1). In the Gaussian model this is the conditional covariance given surviving readings. Reveal station-block coordinates R exactly; keep the original coordinate set, setting revealed-coordinate uncertainty to zero. Then C_after=C-C_:R(C_RR)^(-1)C_R:. The subtraction is positive semidefinite. For nonnegative coordinate weights w, the exact reduction in model-based expected weighted squared reconstruction error is trace(diag(w) C_:R(C_RR)^(-1)C_R:). Selecting the candidate with the largest reduction is optimal among the supplied single-station repairs.','',
 'Proof: Gaussian block conditioning gives the stated Schur complement on unrevealed coordinates. The same formula makes revealed rows and columns zero, giving the fixed-coordinate representation. Since C_RR is positive definite, the removed matrix factors as B B^T and is positive semidefinite. Conditional means minimize Gaussian expected squared error, whose weighted expectation is trace(diag(w) C). Subtracting the two traces gives the gain. Exhaustive evaluation of single-candidate gains establishes the one-repair optimum.','',
 'This is a project-specific specialization of classical Gaussian conditioning and experimental design, not an invention of those identities. It assumes the fitted Gaussian model describes error; fitted precision alone does not establish calibrated real-world covariance. Equal weights in transformed model coordinates differ from the concentration-space metric used for empirical scoring. The theorem therefore does not imply positive empirical gains, a submodular objective, an optimal multi-repair policy, or a pollution-reduction guarantee. See [Krause, Singh and Guestrin, sensor placement](https://jmlr.org/papers/v9/krause08a.html).','',
 '## Empirical repair results','',
 'At up to 240 deterministic held-out hours per network, sample at most four eligible station pairs using a fixed seed. Both station vectors are hidden before policy selection. Selection uses only frozen precision and masks. Reveal one chosen station and reconstruct the same original two-station target set. The random comparator is the exact average over all candidate repairs, not a lucky random draw. The outcome oracle is stored as a diagnostic and excluded from valid policy claims.','',
 '| Network | Cases / sampled days | RMSE gain vs random expected | Paired MSE-difference 95% day CI | RMSE gain vs variance-only |',
 '|---|---:|---:|---|---:|']
for region,g in R['comparisons'].items():
    a=g['random_expected'];b=g['variance_only'];lo,hi=a['difference_ci95']
    lines.append(f"| {region} | {a['n']} / {a['clusters']} | {100*a['rmse_relative_gain']:.2f}% | [{lo:.4f}, {hi:.4f}] | {100*b['rmse_relative_gain']:.2f}% |")
lines += ['',
 'The policy improves over random expected repair in six of seven networks, but is worse in Bengaluru’s archive cohort. Crucially, it provides essentially no improvement over the simpler variance-only policy in this experiment. Accordingly these results support model-guided repair over random repair in some settings, but do not establish the advanced cross-station criterion as a superior empirical innovation. This exploratory study was added after examining the archive and uses day-cluster intervals without a new multiplicity adjustment. Shared stations/periods are not independent replications.','',
 '## Environmental benefit and remaining limits','',
 'The tested repair endpoint is reconstruction error after a simulated repair. It connects the mathematical method to maintaining pollution-monitoring coverage under limited maintenance capacity. No physical repair, emissions reduction, health improvement, official AQI/GRAP recovery or calibrated unobserved-exposure interval has been demonstrated.','',
 'The code-level defects identified in the supplied review are addressed by corrected implementations, reruns or removal of unsupported claims. Established-theory attribution, actual-mask bounds, equal candidate universes, clean/unknown-count fault evaluation, train-only cohorts, threshold provenance, calendar-correct yesterday comparisons and source-aware overlap summaries are documented in the existing scientific audit.','',
 'Stronger neural baselines, a genuinely untouched external period, operational delayed/missing-label interval validation and actual policy/field validation remain additional evidence needed for stronger claims. These are not hidden inside a claim that every possible experiment is finished. A rigorous project can report its demonstrated scope without them; global state-of-the-art, mathematical-first and realized environmental-impact claims cannot.','',
 '## Reproduction','',
 'The extended `Run_Research_Overnight.cmd` runs tests, resumes valid completed core runs, and executes the supplementary comparison, weather, interval and repair analyses. Normal execution does not require rerunning valid frozen cores. Its report and zip generation follows the supplements. Source hashes for this extension are in `technical_extension_manifest.json`.','']
(D/'TECHNICAL_EXTENSION_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
fig,ax=plt.subplots(figsize=(9,5))
regions=list(R['comparisons']);values=[100*R['comparisons'][r]['random_expected']['rmse_relative_gain'] for r in regions]
ax.barh(regions,values,color=['#237c88' if x>0 else '#b34e42' for x in values]);ax.axvline(0,color='black',lw=.8)
ax.set_xlabel('Relative RMSE gain vs expected random repair (%)');ax.set_title('Simulated single repair during two-station outages')
fig.tight_layout();fig.savefig(F/'repair_selection.pdf');fig.savefig(F/'repair_selection.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,5))
for region in ['delhi2025','bengaluru_raw']:
    frame=pd.read_csv(ROOT/'results'/region/'intervals_by_pollutant.csv')
    g=frame[frame.pollutant=='PM2.5'].copy();g['finite_n']=g.n-g.full_sets-g.empty_sets
    methods=['fixed_no_labels','adaptive_revealed','adaptive_mask_scale']
    coverage=[];width=[]
    for name in methods:
        h=g[g.method==name];coverage.append(100*np.average(h.coverage,weights=h.n))
        width.append(np.average(h.mean_finite_halfwidth_concentration_units,weights=h.finite_n))
    axes[0].plot(methods,coverage,marker='o',label=region);axes[1].plot(methods,width,marker='o',label=region)
axes[0].axhline(90,color='black',ls='--');axes[0].set_ylabel('PM2.5 observed-cell coverage (%)')
axes[1].set_ylabel('Mean finite PM2.5 half-width (micrograms/m³)')
for ax in axes:ax.tick_params(axis='x',rotation=20);ax.legend()
fig.suptitle('Finite width excludes full/empty sets; feedback settings require labels')
fig.tight_layout();fig.savefig(F/'pollutant_interval_width.pdf');fig.savefig(F/'pollutant_interval_width.png',dpi=160);plt.close(fig)
paths=[ROOT/'sheafcode/acquisition.py',ROOT/'tests/test_acquisition.py',ROOT/'tests/test_theory.py',Path(__file__),ROOT/'scripts/run_sensor_repair.py',ROOT/'scripts/replay_interval_details.py',D/'TECHNICAL_EXTENSION_RESULTS.md',D/'sensor_repair_summary.json',D/'interval_replay_audit.json']
(D/'technical_extension_manifest.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2))
print(D/'TECHNICAL_EXTENSION_RESULTS.md',flush=True)
