from pathlib import Path
import json,re,urllib.request,urllib.parse,concurrent.futures

out=Path(__file__).resolve().parent
tex=(out/'SheafCode_IRIS_Natural_Writing_Technically_Corrected.tex').read_text(encoding='utf-8')
dois=re.findall(r'\\doi\{([^}]+)\}',tex)
def fetch(doi):
    url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'SheafCode-Manuscript-Reference-Audit/1.0'})
        with urllib.request.urlopen(req,timeout=40) as response:
            m=json.load(response)['message']
        return {'doi':doi,'source':url,'title':m.get('title'),'authors':m.get('author'),'published':m.get('published'),'volume':m.get('volume'),'issue':m.get('issue'),'page':m.get('page'),'container-title':m.get('container-title'),'status':'retrieved'}
    except Exception as e:return {'doi':doi,'source':url,'status':'unavailable','error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(fetch,dois))
(out/'BIBLIOGRAPHY_METADATA.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
for r in records:
    print(r['doi'],r['status'],r.get('title'),r.get('published'),r.get('volume'),r.get('issue'),r.get('page'))
