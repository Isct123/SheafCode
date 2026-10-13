                                                                                            
from pathlib import Path
import argparse,csv,json,sys
sys.path.insert(0,str(Path(__file__).parent))
from decision_repair import conflict_edges,master,plan

def main():
    candidate=Path(__file__).resolve().parents[2]/'decision_repair_results'
    default=candidate if candidate.exists() else Path(__file__).resolve().parent.parent/'results'
    parser=argparse.ArgumentParser();parser.add_argument('--results-dir',type=Path,default=default);args=parser.parse_args()
    banks=json.loads((args.results_dir/'SCENARIO_BANKS.json').read_text())
    rows=list(csv.DictReader((args.results_dir/'REPAIR_CONFIGURATIONS.csv').open(newline='',encoding='utf-8')))
    checks=0
    for row in rows:
        bank=banks[row['cohort']];threshold=float(row['threshold']);eta=float(row['eta']);f=int(row['query_faults'])
        states=[];labels=[]
        for state in bank['pm25_vectors']:
            maximum=max(state)
            if maximum<=threshold-eta or maximum>threshold+eta:
                states.append(state);labels.append(int(maximum>threshold+eta))
        assert len(states)==int(row['scenarios']) and labels.count(0)==int(row['safe']) and labels.count(1)==int(row['exceedance']);checks+=3
        if len(set(labels))<2:
            assert row['decision_status']=='trivial_or_empty';checks+=1;continue
        n=len(bank['stations']);edges=conflict_edges(states,labels,eta);r=plan(n,edges,faults=f)
        assert r['status']==row['decision_status'];checks+=1
        if r['status']=='optimal':
            assert r['cost']==int(row['decision_queries']);checks+=1
        full=master(n,[1]*n,[e for e,_ in conflict_edges(states,list(range(len(states))),eta)],2*f+1)
        assert ('infeasible' if full is None else 'optimal')==row['pattern_status'];checks+=1
        if full is not None:
            assert full[1]==int(row['pattern_queries']);checks+=1
    print(json.dumps({'status':'passed','configurations':len(rows),'checks':checks,'scope':'Replay of finite saved scenario banks; no model retraining or prospective predictions.'},indent=2))

if __name__=='__main__':main()
