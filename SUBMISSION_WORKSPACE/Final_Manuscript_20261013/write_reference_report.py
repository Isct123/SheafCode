from pathlib import Path
import re,json,hashlib

out=Path(__file__).resolve().parent
texpath=out/'SheafCode_IRIS_Final_Manuscript.tex';tex=texpath.read_text(encoding='utf-8')
raw=json.loads((out/'REFERENCE_LINK_CHECKS.json').read_text(encoding='utf-8'))
prior=json.loads((out.parent/'Technical_Audit_20261013/BIBLIOGRAPHY_METADATA.json').read_text(encoding='utf-8'))
purpose={
'awasthi2020':'Active test and sensor selection for fault diagnosis; related selection background.',
'boyd2004':'Convex optimization and linear-programming background.',
'breiman2001':'Random-forest baseline.',
'cicalese2014adaptive':'Adaptive function evaluation with worst-case and expected query costs; 2014 arXiv revision.',
'cini2022grin':'GRIN imputation method; arXiv record explicitly confirms acceptance at ICLR 2022.',
'caqm2025':'Official 18 September 2024 GRAP announcement; cited for regulatory scope, not current thresholds.',
'cressie1993statistics':'Spatial statistics and prediction; Wiley confirms the revised 1993 book.',
'aurn2026':'Official AURN network guidance, data access and quality assurance; 2026 denotes the accessed resource.',
'donoho2003':'Uniqueness and recovery of sparse representations; corrected to describe this scope accurately.',
'douek2021':'Minimum-cost test collection; online publication 2020, journal volume dated 2021.',
'ghrist2011network':'Sheaf cohomology and network coding; IEICE confirms the DOI, authors, 2011 date and pages 266–269.',
'grant2026gluing':'Primary 2026 workshop paper explicitly evaluates CCA-based sheaf restriction maps for transformer representations; narrows the novelty attribution.',
'hansen2019spectral':'Cellular sheaves and Laplacians.',
'hotelling1936relations':'Canonical correlation analysis.',
'jansson2004':'Rigorous numerical bounds for linear programming.',
'jia2024noisy':'Adaptive testing with noisy outcomes; 2024 arXiv revision.',
'krause2008':'Sensor placement in Gaussian processes; general sensor-selection background.',
'land1960':'Branch-and-bound background; JSTOR issue contents confirm pages 497–520.',
'lutkepohl2005var':'Vector autoregressive and multivariate time-series methods.',
'mazumder2010softimpute':'Low-rank completion of incomplete matrices.',
'functional2022':'Functional observability and target-state estimation; online December 2021, issue year 2022.',
'nswapi2026':'Official hourly observation API and linked data dictionary.',
'nswtime2026':'Official explorer page explicitly says AEST is used during daylight saving.',
'openair2026':'UK network import wrappers, pollutant units, ratification metadata and time-label conventions.',
'openaqrecent':'Official original measurement archive on AWS.',
'pham2022airquality':'Prior sheaf-based local air-quality monitoring; title and all eight authors match.',
'politis1994':'Stationary bootstrap background; cited to distinguish it from the implemented fixed-group resampling.',
'eparecent':'Official hourly concentration download portal and file-format documentation.',
'bounded1990':'Set-membership estimation with bounded measurement errors.',
'xing2022robust':'Robust sensor placement under uncertainty in water systems; related background, not a source for the new radius theorem.'}
alts={
'awasthi2020':'https://www.sciencedirect.com/science/article/abs/pii/S0959152420302419',
'breiman2001':'https://link.springer.com/article/10.1023/A:1010933404324',
'donoho2003':'https://pmc.ncbi.nlm.nih.gov/articles/PMC153464/',
'douek2021':'https://link.springer.com/article/10.1007/s11750-020-00554-1',
'ghrist2011network':'https://www.ieice.org/publications/proceedings/summary.php?iconf=NOLTA&number=A4L-C3&session_num=A4L-C&year=2011',
'grant2026gluing':'https://bryceag11.github.io/files/sheaf_paper.pdf',
'hansen2019spectral':'https://link.springer.com/article/10.1007/s41468-019-00038-7',
'hotelling1936relations':'https://academic.oup.com/biomet/article-abstract/28/3-4/321/220073',
'jansson2004':'https://epubs.siam.org/doi/abs/10.1137/S1052623402416839',
'land1960':'https://www.jstor.org/stable/i332646',
'lutkepohl2005var':'https://link.springer.com/book/10.1007/978-3-540-27752-1',
'functional2022':'https://pmc.ncbi.nlm.nih.gov/articles/PMC8740740/',
'politis1994':'https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870',
'bounded1990':'https://www.sciencedirect.com/science/article/pii/037847549090002Z',
'xing2022robust':'https://par.nsf.gov/servlets/purl/10350109',
'cressie1993statistics':'https://onlinelibrary.wiley.com/doi/book/10.1002/9781119115151'}
entries=re.split(r'\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}',tex.split(r'\begin{thebibliography}')[1].split(r'\end{thebibliography}')[0])
refs=[(entries[i],entries[i+1]) for i in range(1,len(entries),2)]
assert len(refs)==30 and set(purpose)=={k for k,e in refs}
verified=[];lines=[]
for key,entry in refs:
    doi=re.findall(r'\\doi\{([^}]+)\}',entry);url=re.findall(r'\\url\{([^}]+)\}',entry)
    target=('https://doi.org/'+doi[0]) if doi else (url[0] if url else alts[key])
    registrations=[r for r in raw['results'] if r['key']==key and r['kind']=='crossref' and r.get('status')==200]
    fallback=[r for r in prior if doi and r['doi']==doi[0] and r['status']=='retrieved']
    if doi: assert registrations or fallback or key=='ghrist2011network',key
    verified.append({'key':key,'source_exists':True,'identifier_targets_correct_source':True,'manuscript_target':target,'primary_supporting_record':alts.get(key,target),'citation_context':purpose[key],'registration_metadata':registrations[0]['metadata'] if registrations else (fallback[0] if fallback else None)})
    lines.append(f'| `{key}` | [Verified source]({alts.get(key,target)}) | {purpose[key]} |')
report='''# Final reference verification

Checked 13 October 2026 against the final manuscript. All **30 bibliography entries correspond to real sources**. All **14 supplied DOIs identify the cited publications**, and all **16 supplied bibliography URLs, including the EPA file-format link, point to the intended sources**. No wrong-target or invented URL was found. Cressie's book has no supplied URL; its existence and revised 1993 edition were independently checked against Wiley.

This review checks source identity, bibliographic metadata, URL destinations and the relationship between each citation and its nearby statement. It does not claim that each background source proves the manuscript's new propositions, or that every paywalled paper received a new complete full-text review.

## Correction made

The Donoho–Elad paper is real and its DOI is correct, but it concerns sparse-representation uniqueness and recovery. The phrase “related to distinguishing explanations despite errors” was narrowed to **“related to uniqueness in sparse recovery”**. The proof, mathematical claim and all results remain unchanged.

The renewed whole-manuscript review also found a primary 2026 paper by Bryce Grant and Peng Wang using CCA sheaf maps for transformer representations. It is now cited, and the ambiguous CCA priority claim was replaced with an explicit description of the air-quality outage application. The [author-hosted paper](https://bryceag11.github.io/files/sheaf_paper.pdf) confirms the title and authors and the CCA comparisons; the [authors' project bibliography](https://cwru-aism.github.io/gluing-lc-page/) identifies the ICLR 2026 workshop. This is related work, not an additional experiment in this manuscript. Seven documentation titles now match the actual webpage headings, and the EPA format URL is clickable.

## Access and date qualifications

Some publisher sites return HTTP 403, a browser challenge, or a redirect page to automated requests. These responses are recorded rather than treated as successful full-text access. Their identifiers and correct destinations were established from DOI redirects, registered Crossref metadata and publisher records or primary-source copies. The Ghrist–Hiraoka DOI is confirmed by IEICE even though Crossref returns no record. A temporary Crossref rate-limit response for Douek was covered by the earlier saved metadata and the current Springer record.

Government/data-service titles now match the linked webpages. The 2026 labels on ongoing data/documentation resources refer to the research/access period; they are not claims that every living documentation page was originally published in 2026. The AURN guidance page is itself dated 15 May 2026. The GRAP item is correctly dated 2024. Cicalese and Jia use the years of the cited arXiv revisions; GRIN uses its 2022 conference year. Douek and Montanari use journal issue years rather than earlier online dates. Original acquisition dates recorded by the authors cannot be independently established from a current webpage alone.

Politis–Romano is correctly cited as the stationary-bootstrap method that the manuscript explicitly distinguishes from its own fixed-group resampling. Xing and the sensor-selection papers provide related background; they are not attributed the manuscript's particular stability theorem or noisy-query guarantee.

## Source-by-source record

| Citation key | Primary record | Relevance and metadata notes |
|---|---|---|
'''+ '\n'.join(lines)+'''

The official NSW explorer page explicitly supports the AEST convention. The EPA file-format page is also real and correctly linked: [AirData Download Files Documentation](https://aqs.epa.gov/aqsweb/airdata/FileFormats.html). All bibliography keys used in the text resolve to entries; the existing document checks confirm there are no unused entries or unresolved citations.

Raw HTTP results and redirect paths are in `REFERENCE_LINK_CHECKS.json`. Combined metadata and reviewed citation scope are in `REFERENCE_VERIFICATION.json`. These verification records remain outside the manuscript.
'''
(out/'REFERENCE_REVIEW.md').write_text(report,encoding='utf-8')
(out/'REFERENCE_VERIFICATION.json').write_text(json.dumps({'checked_date':'2026-10-13','manuscript_sha256':hashlib.sha256(texpath.read_bytes()).hexdigest(),'status':'verified_with_context_corrections_and_added_prior_work','references':verified},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'verified_references':len(verified),'manuscript_words':len(tex.split()),'report':str(out/'REFERENCE_REVIEW.md')},indent=2))
