                                                                       
import csv,json,time,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from adaptive_repair import solve

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Research_Release/decision_repair_results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    banks=json.loads((OUT/'SCENARIO_BANKS.json').read_text())
    rows=list(csv.DictReader((OUT/'REPAIR_CONFIGURATIONS.csv').open(newline='',encoding='utf-8')))
    output=[];policies=[];checks=0;start=time.time()
    for row in rows:
        bank=banks[row['cohort']]; threshold=float(row['threshold']);eta=float(row['eta']);f=int(row['query_faults'])
        states=[];labels=[]
        for x in bank['pm25_vectors']:
            maximum=max(x)
            if maximum<=threshold-eta or maximum>threshold+eta:
                states.append(x);labels.append(int(maximum>threshold+eta))
        result={'status':'trivial_or_empty','cost':'','nodes':0,'policy_states_checked':0}
        if len(set(labels))==2:
            result=solve(states,labels,eta,faults=f,max_nodes=50000)
            if result['status']!='search_limit':
                assert (result['status']=='infeasible')==(row['decision_status']=='infeasible');checks+=1
            if result['status']=='optimal':
                assert result['cost']<=int(row['decision_queries']);checks+=1
                checks+=result['policy_states_checked']
        output.append({**{k:row[k] for k in ['cohort','threshold','eta','query_faults','scenarios','safe','exceedance','decision_queries']},
                       'adaptive_status':result['status'],'adaptive_queries':result.get('cost',''),
                       'nodes':result['nodes'],'policy_states_checked':result.get('policy_states_checked',0)})
        if result['status']=='optimal':
            policies.append({'configuration':len(output)-1,'cohort':row['cohort'],
                             'states':states,'labels':labels,'eta':eta,'faults':f,
                             'cost':result['cost'],'policy':result['policy']})
        if len(output)%18==0:print('Completed',len(output),'of',len(rows),flush=True)
    summary={}
    for f in ['0','1']:
        group=[r for r in output if r['query_faults']==f]
        completed=[r for r in group if r['adaptive_status']=='optimal']
        fixed=sum(int(r['decision_queries']) for r in completed);adaptive=sum(r['adaptive_queries'] for r in completed)
        summary[f]={'configurations':len(group),'optimal':len(completed),
                    'infeasible':sum(r['adaptive_status']=='infeasible' for r in group),
                    'search_limit':sum(r['adaptive_status']=='search_limit' for r in group),
                    'strictly_cheaper':sum(r['adaptive_queries']<int(r['decision_queries']) for r in completed),
                    'fixed_query_sum_common':fixed,'adaptive_query_sum_common':adaptive,
                    'saving_percent_common':100*(fixed-adaptive)/fixed if fixed else None}
    with (OUT/'ADAPTIVE_CONFIGURATIONS.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    (OUT/'ADAPTIVE_POLICIES.json').write_text(json.dumps(policies),encoding='utf-8')
    report={'status':'completed','summaries':summary,'checks':checks,'seconds':time.time()-start,
            'scope':'Exact worst-case policies over finite offline archive banks; dependent settings, conditional model membership.',
            'input_sha256':{p.name:sha(p) for p in [OUT/'SCENARIO_BANKS.json',OUT/'REPAIR_CONFIGURATIONS.csv']},
            'source_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('adaptive_repair.py'),Path(__file__).with_name('ADAPTIVE_PROTOCOL.md')]}}
    (OUT/'ADAPTIVE_SUMMARY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
