                                                                        
import hashlib,json,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
COMMIT='4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e'
def main():
    out=HERE/'vendor/grin';out.mkdir(parents=True,exist_ok=True)
    tree=json.load(urllib.request.urlopen('https://api.github.com/repos/Graph-Machine-Learning-Group/grin/git/trees/'+COMMIT+'?recursive=1',timeout=60))
    manifest={}
    for item in tree['tree']:
        name=item['path']
        if item['type']!='blob' or not (name.startswith('lib/') and name.endswith('.py') or name.lower().startswith('license') or name in ['README.md','requirements.txt']):continue
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        data=urllib.request.urlopen('https://raw.githubusercontent.com/Graph-Machine-Learning-Group/grin/'+COMMIT+'/'+name,timeout=60).read()
        path.write_bytes(data);manifest[name]=hashlib.sha256(data).hexdigest()
    (out/'SOURCE.json').write_text(json.dumps({'repository':'https://github.com/Graph-Machine-Learning-Group/grin','commit':COMMIT,'files':manifest},indent=2))
    print('Downloaded pinned official GRIN:',len(manifest),'files',flush=True)
if __name__=='__main__':main()
