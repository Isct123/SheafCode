                                                                                         
from pathlib import Path
import json,os,subprocess,shutil,hashlib
import pymupdf

ROOT=Path(__file__).resolve().parents[1];PKG=ROOT/'IRIS_Submission_Recent_Data'
SRC=PKG/'Editable_Manuscript';OUT=PKG/'Upload_Files'

FIELDS={
'Introduction and objective': '''Pollution monitoring becomes less informative when stations stop reporting. Neighbouring stations and pre-outage history may help reconstruct the missing readings, but a plausible estimate is not automatically a trustworthy pollution decision. This project asks whether training-fitted spatial and temporal comparisons improve causal reconstruction against strong matched controls, and which additional measurements would settle a threshold decision despite noise and faulty queries. A second objective is to explain the limits: graph anchors describe remaining ambiguity, joint constraints can establish a decision without identifying every pollutant, and conflict witnesses identify useful repair readings. A stability-radius theorem then measures how far true concentrations can move away from declared archive scenarios before a fixed repair guarantee fails. The environmental application is computational monitoring continuity and measurement planning, rather than a demonstrated reduction in emissions.''',
'Innovation': '''The research novelty is a decision-first monitoring and repair framework. Joint pollutant constraints can preserve a decision when individual values remain ambiguous. Opposite-decision witnesses then identify robust measurements; a stability-radius theorem derives their exact continuous-box tolerance from sorted station margins and optimizes it under a budget. Adaptive policies choose later measurements from earlier responses. These linked algorithms and their evaluations are the contribution, with prior sheaf, robust sensor-selection and decision-tree methods credited. Guarantees specify the declared model, reading error and corruption limits.''',
'Methodology': '''Recent public records from OpenAQ, EPA, UK AURN and Australian NSW monitoring archives support the computational evaluation. The original recent Indian, US and UK benchmarks retain separate roles; a fixed exploratory expansion adds twenty evaluable external cohorts. Chronological training, selection, calibration and test blocks separate fitting, tuning and evaluation. Training availability determines station and channel membership. Training-fitted normalization, spatial comparisons, one-hour and twenty-four-hour relations and a full-state prior produce causal reconstructions without hidden target or future inputs. Controls include tuned full-covariance causal VAR, seasonal persistence, city mean, Gaussian conditioning, spatial sheaf and a causal random forest. Retrospective GRIN comparisons retain all seeds and are labelled separately. Paired station-day errors define the normalized RMSE metric; seven-day calendar-bin bootstrap intervals describe temporal sensitivity. Synthetic faults, natural-mask transplantation, interval coverage and threshold proxies retain unfavorable outcomes. Repair experiments evaluate all 414 fixed reference/noise/corruption settings on unchanged archived scalar scenarios. Exact bitmask optimization, adaptive dynamic programming and rational stability frontiers produce plans and failure witnesses. Independent small-bank implementations and solver-independent response replays test mathematical correctness. These archive-bank experiments do not establish future-state coverage or independently validated field repair.''',
'Results and conclusions': '''The fixed Indian causal comparison reduces normalized RMSE by 16.48% against tuned VAR on 77 paired station-days, with an exploratory weekly interval of 12.62 to 20.16%; physical PM2.5 error falls 10.45%. Across twenty external cohorts, sheaf has lower normalized error in thirteen against VAR and eighteen against the tested forest. All losses remain reported. Without corrupted queries, decision repair uses 18 versus 24 queries for whole-pattern identification across ten common-feasible settings. Adaptive repair reduces summed worst-case queries from 58 to 52 across fifteen one-corruption feasible settings; forty-six settings remain impossible. At the same fixed query count, stability optimization improves twelve of sixty-one no-corruption settings and four of fifteen one-corruption settings. These dependent, model-conditional results support monitoring and repair reasoning, not measured emissions reduction.''',
'Acknowledgements and reference links': '''Public observations are credited to OpenAQ and its original providers, EPA, DEFRA AURN and NSW monitoring services. CPCB materials and published GRIN code are acknowledged. The manuscript cites sheaf theory, sparse uniqueness, convex optimization, generalized test collection, robust sensor placement and adaptive decision-tree literature. Full references and source links accompany the paper. AI assistance in code, mathematical derivation, checking and writing is disclosed. Student contributions, external assistance and the supervising guide must be recorded accurately before submission.'''
}

def latex(s):
    return s.replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')

def main():
    abstract=(OUT/'Project_Abstract_Concise.txt').read_text(encoding='utf-8').strip()
    assert len(abstract.split())<=250
    limits={'Introduction and objective':(100,150),'Innovation':(50,100),'Methodology':(150,250),'Results and conclusions':(100,150),'Acknowledgements and reference links':(50,100)}
    counts={name:len(body.split()) for name,body in FIELDS.items()}
    for name,(lo,hi) in limits.items():assert lo<=counts[name]<=hi,(name,counts[name])
    title='SheafCode: Recovering Pollution Information During Sensor Outages'
    text='Project title: '+title+'\n\nAbstract\n'+abstract+'\n\n'+'\n\n'.join(name+'\n'+body for name,body in FIELDS.items())+'\n'
    (OUT/'Portal_Text_Concise.txt').write_text(text,encoding='utf-8')
    intro=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}\usepackage{microtype,hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{.5em}
\title{SheafCode: Recovering Pollution Information During Sensor Outages}
\author{}\date{8 October 2026}
\begin{document}\maketitle\vspace{-1.5em}
\textbf{Completed-study synopsis.} This summary describes executed research. It is not a backdated prospective research plan or approval record.
'''
    synopsis_summary='''SheafCode reconstructs missing pollution readings and asks which additional measurements would resolve a threshold decision. Its fixed Indian causal benchmark reduces normalized RMSE by 16.48% against tuned VAR; twenty external cohorts retain both gains and losses. Three core propositions motivate conditional decisions and repair. A stability-radius theorem extends fixed repair to declared continuous scenario boxes, while adaptive repair chooses subsequent readings from earlier answers. Tests and failure witnesses verify the algorithms under stated assumptions. Environmental value concerns monitoring continuity, rather than measured emissions reduction.'''
    source=intro+'\n'+r'\section*{Summary}'+'\n'+latex(synopsis_summary)+'\n'
    source+='\n'.join(r'\section*{'+latex(name)+'}\n'+latex(body) for name,body in FIELDS.items())
    source+=r'''
\small Current full report: \textit{SheafCode Concise Manuscript}. Source archives: \url{https://www.openaq.org/}, \url{https://www.epa.gov/outdoor-air-quality-data}, \url{https://uk-air.defra.gov.uk/}, and \url{https://www.airquality.nsw.gov.au/}.
\end{document}
'''
                                                                             
    source=source.replace('}\\n','}\n')
    tex=SRC/'concise_synopsis.tex';tex.write_text(source,encoding='utf-8')
    env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(ROOT/'audit_tools/tex_cache')
    proc=subprocess.run([str(ROOT/'audit_tools/tectonic/tectonic.exe'),'--only-cached','--keep-logs','--outdir',str(SRC),str(tex)],cwd=SRC,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (PKG/'Review_Notes/concise_synopsis_compile.txt').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    assert proc.returncode==0,proc.stderr
    pdf=SRC/'concise_synopsis.pdf';doc=pymupdf.open(pdf);assert len(doc)<=5
    assert '??' not in '\n'.join(page.get_text() for page in doc)
    log=(SRC/'concise_synopsis.log').read_text(encoding='utf-8',errors='replace');assert 'Overfull' not in log
    destination=OUT/'SheafCode_Concise_Research_Synopsis.pdf';shutil.copy2(pdf,destination)
    video='''VIDEO DRAFT — replace name/grade and rehearse to at most 90 seconds. Do not state your school or residence. Record only claims you understand and can defend.

I am [FULL NAME], in [GRADE]. My project asks how to keep pollution monitoring informative when stations stop reporting.

SheafCode compares neighbouring stations and their earlier readings to reconstruct missing measurements. I evaluated it against tuned statistical and machine-learning controls using recent public data from India, the United States, the United Kingdom and Australia.

The fixed Indian test reduced normalized error by 16.48 percent against the tuned causal VAR comparator. Across twenty external cohorts, the method beat that comparator in thirteen, so I report the seven losses too.

The mathematics asks a different question: which measurements are needed to settle a pollution decision? Opposite explanations create separating sets. A repair plan must hit them often enough to tolerate corrupted readings. My stability theorem calculates how much concentrations can move before that guarantee fails, using sorted station margins. Adaptive repair lets each answer choose the next station.

These are conditional monitoring and measurement-planning results. I have not measured an emissions reduction. The paper credits prior methods and discloses AI assistance and other contributions.
'''
    (OUT/'Video_Script_Concise.txt').write_text(video,encoding='utf-8')
    report={'date':'2026-10-08','status':'matching_companions_built','abstract_words':len(abstract.split()),'field_word_counts':counts,
            'word_limit_source':'Archived IRIS 2025-26 submission guidelines; current signed-in form may differ and must be checked.',
            'synopsis_pages':len(doc),'synopsis_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'paper_sha256':hashlib.sha256((OUT/'SheafCode_Concise_Manuscript.pdf').read_bytes()).hexdigest(),
            'scope':'Local draft files only; personal forms, video recording, contribution declarations and external submission remain user tasks.'}
    (PKG/'Review_Notes/CONCISE_COMPANIONS_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
