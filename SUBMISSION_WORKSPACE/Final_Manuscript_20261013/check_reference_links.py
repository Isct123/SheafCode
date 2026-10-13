from pathlib import Path
import re,json,urllib.request,urllib.parse,urllib.error,concurrent.futures,hashlib

out=Path(__file__).resolve().parent
tex=(out/'SheafCode_IRIS_Final_Manuscript.tex').read_text(encoding='utf-8')
entries=re.split(r'\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}',tex.split(r'\begin{thebibliography}')[1].split(r'\end{thebibliography}')[0])
refs=[(entries[i],entries[i+1]) for i in range(1,len(entries),2)]
class RedirectRecorder(urllib.request.HTTPRedirectHandler):
    def __init__(self): self.redirects=[]
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        self.redirects.append({'status':code,'from':req.full_url,'to':newurl})
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def fetch(item):
    key,kind,url=item; redirects=RedirectRecorder();opener=urllib.request.build_opener(redirects)
    record={'key':key,'kind':kind,'url':url}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (compatible; ManuscriptReferenceCheck/1.0)'})
        with opener.open(req,timeout=25) as response:
            data=response.read(300000)
            record.update(status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type',''))
            if kind=='crossref':
                m=json.loads(data)['message']
                record['metadata']={k:m.get(k) for k in ['DOI','title','author','published','published-print','published-online','volume','issue','page','container-title','URL']}
            else:
                decoded=data.decode('utf-8',errors='replace')
                title=re.search(r'<title[^>]*>(.*?)</title>',decoded,re.I|re.S)
                record['page_title']=re.sub(r'\s+',' ',title[1]).strip() if title else None
                record['response_sha256']=hashlib.sha256(data).hexdigest()
    except urllib.error.HTTPError as e: record.update(status=e.code,final_url=e.url,error=str(e))
    except Exception as e: record.update(status='unavailable',error=str(e))
    record['redirects']=redirects.redirects
    return record

jobs=[]
for key,entry in refs:
    for doi in re.findall(r'\\doi\{([^}]+)\}',entry):
        jobs.append((key,'doi','https://doi.org/'+urllib.parse.quote(doi,safe='/():')))
        jobs.append((key,'crossref','https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')))
    for url in re.findall(r'\\url\{([^}]+)\}',entry):jobs.append((key,'url',url))
jobs.extend([
 ('eparecent','additional_url','https://aqs.epa.gov/aqsweb/airdata/FileFormats.html'),
 ('ghrist2011network','publisher','https://www.ieice.org/publications/proceedings/summary.php?iconf=NOLTA&number=A4L-C3&session_num=A4L-C&year=2011'),
 ('cressie1993statistics','publisher','https://onlinelibrary.wiley.com/doi/book/10.1002/9781119115151')])
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: results=list(pool.map(fetch,jobs))
(out/'REFERENCE_LINK_CHECKS.json').write_text(json.dumps({'checked_date':'2026-10-13','manuscript_sha256':hashlib.sha256((out/'SheafCode_IRIS_Final_Manuscript.tex').read_bytes()).hexdigest(),'bibliography_entries':len(refs),'results':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for r in results: print(json.dumps({k:r.get(k) for k in ['key','kind','status','final_url','page_title']},ensure_ascii=True))
