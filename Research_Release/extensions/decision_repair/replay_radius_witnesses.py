                                                                                     
import csv,json,sys
from pathlib import Path
from fractions import Fraction as Q

def fraction(x):return Q(x['exact']) if isinstance(x,dict) else Q(x)

def main():
    directory=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[3]/'Research_Release/decision_repair_results'
    banks=json.loads((directory/'SCENARIO_BANKS.json').read_text())
    configs=list(csv.DictReader((directory/'REPAIR_CONFIGURATIONS.csv').open(newline='',encoding='utf-8')))
    witnesses=json.loads((directory/'RADIUS_BOUNDARY_WITNESSES.json').read_text());checks=0
    for record in witnesses:
        row=configs[record['configuration']];tau=float(row['threshold']);eta=Q.from_float(float(row['eta']));f=int(row['query_faults'])
        states=[];labels=[]
        for state in banks[row['cohort']]['pm25_vectors']:
            maximum=max(state)
            if maximum<=tau-float(eta) or maximum>tau+float(eta):
                states.append([Q.from_float(x) for x in state]);labels.append(int(maximum>tau+float(eta)))
        i,j=record['pair'];rho=fraction(record['radius']);w=record['counterexample']
        a=list(map(fraction,w['a']));b=list(map(fraction,w['b']))
        y={int(v):fraction(x) for v,x in w['response'].items()}
        assert labels[i]!=labels[j];checks+=1
        assert len(y)<=record['budget'];checks+=1
        assert all(abs(x-c)<=rho for x,c in zip(a,states[i]));checks+=1
        assert all(abs(x-c)<=rho for x,c in zip(b,states[j]));checks+=1
        assert sum(abs(a[v]-value)>eta for v,value in y.items())<=f;checks+=1
        assert sum(abs(b[v]-value)>eta for v,value in y.items())<=f;checks+=1
        semantic=min(abs(max(x)-Q.from_float(tau)) for x in states)
        if rho<semantic:
            assert int(max(a)>Q.from_float(tau))==labels[i];checks+=1
            assert int(max(b)>Q.from_float(tau))==labels[j];checks+=1
    report={'status':'passed','witnesses':len(witnesses),'checks':checks,
            'scope':'Solver-independent exact verification of saved box-boundary ambiguous responses; physical decision labels verified when inside semantic margin.'}
    (directory/'RADIUS_WITNESS_REPLAY.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
