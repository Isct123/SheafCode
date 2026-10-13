                                                                                 
import hashlib,json,sys,zipfile
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def main():
    out=ROOT/'followup_results';out.mkdir(exist_ok=True)
    required=['interval_summary.csv','alert_summary.csv','manifest.json','robust_quantity/summary.csv',
        'robust_quantity/manifest.json','subspace_compression/benchmark.csv','neural_control/summary.json','external/status.json']
    absent=[name for name in required if not (out/name).exists()]
    if absent:raise RuntimeError('Missing follow-up outputs: '+str(absent))
    external=json.loads((out/'external/status.json').read_text())
    lines=['# Consolidated follow-up results','','This records completion of the declared local experiments. It is not comprehensive scientific sign-off or proof of first-ever novelty.','',
        '## No-feedback uncertainty','',pd.read_csv(out/'interval_summary.csv').to_string(index=False),'',
        '## Alert operating points','',pd.read_csv(out/'alert_summary.csv').to_string(index=False),'',
        'Compare actual test false-positive rates before claiming matched operating-point superiority. Validation cutoffs target the same rate but do not guarantee identical test rates.','',
        '## Unknown-fault quantity bounds','',pd.read_csv(out/'robust_quantity/summary.csv').to_string(index=False),'',
        'Report abstention and unboundedness along with accuracy on decided cases. These are illustrative hourly pollution thresholds.','',
        '## Subspace-compression computation','',pd.read_csv(out/'subspace_compression/benchmark.csv').to_string(index=False),'',
        'Compression is not a demonstrated acceleration for the fitted Delhi model. Exact rational reference tests verify the quotient structure on small examples; real-valued bounds use numerical tolerances.','',
        '## Nonlinear snapshot control','',json.dumps(json.loads((out/'neural_control/summary.json').read_text()),indent=2),'',
        'This custom graph-conditioned neural control is not GRIN or a published temporal graph benchmark.','',
        '## External evidence','',json.dumps(external,indent=2),'',
        'External source independence still needs verification even if supplied data pass the period-overlap checks. No field intervention, pollution reduction or health benefit is established.','',
        '## Final review requirements','',
        'Retain all earlier valid results and negative findings. Reconcile the supplied manuscripts, archive reports and corrected experiments under docs/FINAL_REVIEW_REQUIREMENTS.md. This consolidation does not claim that every historical result has been independently replayed. Use expanded_docs/REVIEWED_RESULTS.md for corrected seven-day intervals.','',
        'Proof files: ROBUST_QUANTITY_THEOREM.md and SUBSPACE_COMPRESSION_THEOREM.md under extensions/followup. First-ever mathematical priority remains unestablished.']
    (out/'FINAL_RESEARCH_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    files=[p for base in [HERE,out,ROOT/'expanded_docs',ROOT/'docs',ROOT/'constrained_results',ROOT/'Results_Figure_Pack',ROOT/'extension_results'] for p in base.rglob('*') if p.is_file() and p.suffix in ['.py','.md','.json','.csv','.pdf','.png','.ps1'] and '__pycache__' not in p.parts and p!=out/'release_hashes.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'release_hashes.json').write_text(json.dumps(hashes,indent=2))
    with zipfile.ZipFile(ROOT/'Final_Research_Results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
        z.write(out/'release_hashes.json','followup_results/release_hashes.json')
    print('Consolidated report and result archive created. External/field evidence may remain pending.',flush=True)

if __name__=='__main__':main()
