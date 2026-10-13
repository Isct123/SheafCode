                                                                        
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'IRIS_Submission_Recent_Data'

def main():
    fields = {
        'Abstract': (PACKAGE/'Upload_Files/Project_Abstract_Expanded.txt').read_text().strip(),
        'Introduction and objective': """Air-quality monitoring stations sometimes fail while neighbouring sensors continue recording. Replacing a missing station with a city average can hide local differences, while complex reconstruction needs comparison with strong causal controls. This project asks whether training-fitted spatial and temporal consistency relations improve reconstruction during station outages, and when surviving observations can establish a pollution quantity or threshold decision despite uncertain faults. The mathematical progression connects graph recoverability, indistinguishable fault explanations, search compression and conditional decision certificates. Recent public observations from four countries support the computational evaluation, with chronological separation of fitting, selection, calibration and testing. The intended environmental application is continuity of local pollution monitoring. Simulated alerts and repair studies evaluate possible uses without claiming measured emissions reduction or verified regulatory decisions.""",
        'Innovation': """Six linked propositions address actual-map graph recoverability, quantity identification under unknown faults, equivalent-support BFS compression, joint pollutant-threshold certification, interrupted support search and nominal numerical bounds. Supporting classical mathematics is credited rather than presented as additional novelty. Unrestricted-fault compression is explicitly separated from bounded-fault decision search. The contribution is the specialized monitoring formulation, derivations, implementation and transparent evaluation. First-ever mathematical priority is unestablished; a certificate remains conditional on its declared model and budgets.""",
        'Methodology': """Original recent Indian, US and UK benchmarks are preserved. A fixed exploratory expansion selected twenty-two additional regional cohorts from official EPA, AURN and NSW archives; twenty have evaluable test observations and two remain unavailable. Additional observations begin on 14 October 2025. Cohorts are selected using training availability. Separate chronological blocks fit models, select parameters, calibrate intervals and evaluate held-out outcomes. Training alone determines normalization, station membership and fitted consistency maps. A causal quadratic decoder uses surviving observations and available history. Tuned full-covariance VAR, city mean, seasonal, Gaussian and spatial controls share eligible target cells. Evaluation includes paired twenty-four-hour station outages, physical PM2.5 errors, partial-channel tests, eighteen stress configurations, synthetic unknown-count faults, uncertainty, ablations, repair and threshold proxies where data support them. Seven-day and fourteen-day dependence sensitivities are reported separately. A longer Indian resplit reuses observations and is labelled non-independent. Independent implementations replay seven-method losses; raw records, code hashes, mathematical checks and archived case tables support reproducibility. All unfavorable results and unavailable experiments remain visible.""",
        'Results and conclusions': """The original Indian primary test reports 16.5% lower normalized RMSE than tuned VAR on 77 paired station-days, with an exploratory weekly interval of 12.6 to 20.2%. Twenty additional external cohorts provide 2,114 paired station-days and 549,302 observed pollutant-hour cells. Sheaf reconstruction has positive normalized all-day gains in thirteen of twenty cohorts; seven losses are retained. The longer reused-observation Indian sensitivity gives 4.6% improvement. Physical concentration errors and dependence intervals are separate. Extreme-cell interval undercoverage, weak-fault failures and conditional-certificate abstentions limit operational claims. Joint threshold examples demonstrate a mathematical possibility, not measured regulatory success. The evidence supports monitoring continuity in some tested networks, rather than universal superiority, measured emissions reduction or independently blinded validation.""",
        'Acknowledgements and references': """Public observations are credited to OpenAQ, original providers, EPA, UK AURN and NSW monitoring services. CPCB materials and published GRIN code are acknowledged. References include Hansen and Ghrist, Donoho and Elad, Boyd and Vandenberghe, Gibbs and Candes, and Land and Doig; complete bibliographic details appear in the manuscript. AI assistance in code, mathematical derivation, checking, audits and writing is disclosed. The submitter must accurately record student work, external assistance and the supervising guide.""",
    }
    fields['Methodology']+=' Additional exploratory controls include a causal forest and official GRIN refits on three fixed provider representatives, with all seeds retained and a retrospective matched comparator for bidirectional GRIN.'
    followup=json.loads((ROOT/'Research_Release/weakness_resolution_results/REPORT_STATUS.json').read_text())
    if followup['status']!='generated' or followup['neural_seed_comparisons']!=9:raise RuntimeError('Follow-up publication is incomplete')
    fields['Results and conclusions']+=f" Sheaf improves over the causal forest in {followup['external_forest_wins']}/{followup['external_cohorts']} added cohorts. Three external retrospective GRIN comparisons retain every seed; extreme-risk intervals remain imperfect."
    limits={'Abstract':(1,250),'Introduction and objective':(100,150),
            'Innovation':(50,100),'Methodology':(150,250),
            'Results and conclusions':(100,150),'Acknowledgements and references':(50,100)}
    counts={k:len(v.split()) for k,v in fields.items()}
    for k,(lo,hi) in limits.items():
        if not lo<=counts[k]<=hi: raise AssertionError((k,counts[k],lo,hi))
    title='SheafCode: Causal Air-Quality Reconstruction and Conditional Pollution Certificates'
    text='Project title: '+title+'\n\n'+'\n\n'.join(k+'\n'+v for k,v in fields.items())+'\n'
    (PACKAGE/'Upload_Files/Portal_Text.txt').write_text(text,encoding='utf-8')
    (PACKAGE/'Review_Notes/PORTAL_TEXT_CHECK.json').write_text(json.dumps({
        'status':'passed','word_counts':counts,'word_limits':limits,
        'scope':'Current manuscript alignment and published field lengths; video, forms and eligibility not certified'},indent=2))
    (PACKAGE/'FINAL_SUBMISSION_README.md').write_text("""# Final submission files

Use the following current files:

- Research paper: Upload_Files/SheafCode_Full_Empirical_Manuscript.pdf.
- Synopsis: Upload_Files/SheafCode_Full_Empirical_Synopsis.pdf.
- Abstract: Upload_Files/Project_Abstract_Expanded.txt.
- Other portal fields: Upload_Files/Portal_Text.txt.

Earlier PDFs in the upload folder are preserved references, not the designated
current submission. The combined historical reading copy is also a reference;
it contains older observations and is not the designated recent-data paper.

The manuscript has six selected propositions with supporting derivations,
the original recent benchmarks and twenty added evaluable external cohorts.
Review notes document every negative result and remaining scientific limitation.

Before submission, complete the project video, supervising guide and contribution
records, and any required registration/forms. IRIS requires a video URL for a
video of at most ninety seconds. Student identity and eligibility are not
inferred or certified by these files. Check the twelve-month observation window
at the actual submission date: the oldest included observations are
14 October 2025. The official guidelines currently display a deadline of
13 October 2026.
""",encoding='utf-8')
    print('Portal fields aligned:',counts,flush=True)

if __name__=='__main__': main()
