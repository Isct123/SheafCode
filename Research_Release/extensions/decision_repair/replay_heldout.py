                                                                                      
from pathlib import Path
from fractions import Fraction as Q
import csv,json,hashlib,math,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    directory=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'Research_Release/operational_validation_results'
    lock=json.loads((directory/'PROTOCOL_LOCK.json').read_text());banks=json.loads((directory/'FROZEN_BANKS.json').read_text());checks=0
    for name,value in lock['input_sha256'].items():require(sha(ROOT/name)==value,'Changed input');checks+=1
    arrays={};maps={};membership={};clean_cache={}
    for name,bank in banks.items():
        folder=ROOT/'Research_Release/expanded_validation_results'/name
        if name in ['original_india','original_usa']:folder=ROOT/'Research_Release/recent_data_results'/('delhi' if name=='original_india' else 'epa_la')
        with np.load(folder/'dataset.npz',allow_pickle=False) as data:
            X=data['X'][:,:,list(map(str,data['channels'])).index('PM2.5')].copy();s=data['split'].copy();t=data['times'].copy();station=list(map(str,data['stations']))
        require(station==bank['stations'],'Station mismatch');checks+=1
        complete=np.isfinite(X).all(1);train=np.flatnonzero((s==0)&complete);cal=np.flatnonzero((s==3)&complete);test=np.flatnonzero((s==2)&complete)
        selected=train[np.unique(np.linspace(0,len(train)-1,min(24,len(train))).astype(int))]
        centres=np.array(bank['centres']);require(np.array_equal(centres,X[selected]),'Training centre mismatch');checks+=1
        require(bank['training_timestamps']==[str(x) for x in t[selected]],'Centre timestamps mismatch');checks+=1
        radii=[Q(x) for x in bank['radii_exact']]
        scores=np.min(np.max(abs(X[cal,None,:]-centres[None,:,:]),axis=2),axis=1)
        rank=math.ceil(.9*(len(cal)+1));raw=float(sorted(scores)[rank-1])
        require(bank['calibration_rank']==rank and bank['uncapped_calibrated_radius']==raw,'Calibration mismatch');checks+=1
        exact_centres=[[Q.from_float(float(x)) for x in row] for row in centres]
        for i,(row,label,radius) in enumerate(zip(exact_centres,bank['labels'],radii)):
            require(radius==min(Q.from_float(raw),Q(19,20)*abs(max(row)-35)),'Radius mismatch');checks+=1
            require(max(row)-radius>35 if label else max(row)+radius<=35,'Box changes its label');checks+=1
        order=sorted(range(len(station)),key=lambda v:(-float(np.mean(X[train,v]>35)),v));require(order==bank['query_order'],'Query order mismatch');checks+=1
        for kind,plans in bank['plans'].items():
            r=[Q(0)]*len(radii) if kind=='point' else radii
            for faults,plan in plans.items():
                required=2*int(faults)+1
                if not plan['feasible']:continue
                for i in range(len(centres)):
                    for j in range(i):
                        if bank['labels'][i]==bank['labels'][j]:continue
                        require(sum(abs(exact_centres[i][v]-exact_centres[j][v])>r[i]+r[j]+2 for v in plan['stations'])>=required,'Invalid fixed plan');checks+=1
        arrays[name]=X;maps[name]={str(t[ix]):int(ix) for ix in test}
        for ix in test:
            inside={}
            for kind in ['point','box']:
                rs=[Q(0)]*len(radii) if kind=='point' else radii
                delta=np.max(abs(centres-X[ix]),axis=1);floating=np.array([float(v) for v in rs]);poss=delta<=floating
                for j in np.flatnonzero(abs(delta-floating)<=1e-10):
                    poss[j]=all(abs(Q.from_float(float(x))-a)<=rs[j] for x,a in zip(X[ix],exact_centres[j]))
                inside[kind]=bool(poss.any())
            membership[(name,str(t[ix]))]=inside
    trials=0;sampled_model_checks=0;wrong={'point':0,'box':0};direct_wrong=0
    with (directory/'TRIALS.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            trials+=1;name=row['cohort'];stamp=row['timestamp'];bank=banks[name];x=arrays[name][maps[name][stamp]];faults=int(row['faults']);sign=int(row['error_sign']);truth=int(max(x)>35)
            require(int(row['truth'])==truth,'Truth mismatch');checks+=1
            key=(name,stamp,sign)
            if key not in clean_cache:
                y=x+sign
                for v in range(len(x)):
                    if abs(Q.from_float(float(y[v]))-Q.from_float(float(x[v])))>abs(sign):y[v]=np.nextafter(y[v],x[v])
                clean_cache[key]=y
            y=clean_cache[key].copy()
            if row['corrupt_station']!='':y[bank['stations'].index(row['corrupt_station'])]=-65. if row['corruption']=='low' else 135.
            initial=bank['plans']['box'][str(faults)]['stations'];order=bank['query_order'];policy=[v for v in order if v in initial]+[v for v in order if v not in initial]
            high=y>36.;safe=bool(np.all(y<=34.));positions=[k+1 for k,v in enumerate(policy) if high[v]]
            expected=1 if len(positions)>faults else (0 if faults==0 and safe else None)
            cost=positions[faults] if len(positions)>faults else len(x)
            require(row['direct_prediction']==('' if expected is None else str(expected)),'Direct label mismatch');checks+=1
            require(int(row['direct_queries'])==cost,'Direct query-cost mismatch');checks+=1
            simple_positions=[k+1 for k,v in enumerate(order) if high[v]]
            require(int(row['training_order_queries'])==(simple_positions[faults] if len(simple_positions)>faults else len(x)),'Training-order cost mismatch');checks+=1
            require(int(row['full_queries'])==len(x),'Full query count mismatch');checks+=1
            require(row['full_prediction']==row['direct_prediction'],'Control label mismatch');checks+=1
            if expected is not None and expected!=truth:direct_wrong+=1
            for kind in ['point','box']:
                inside=membership[(name,stamp)][kind];require(int(row[kind+'_member'])==int(inside),'Membership mismatch');checks+=1
                if row[kind+'_prediction']!='' and int(row[kind+'_prediction'])!=truth:
                    wrong[kind]+=1;require(not inside,'Wrong in-model decision');checks+=1
                if trials%313==0:
                    plan=bank['plans'][kind][str(faults)];poss=set()
                    if plan['feasible']:
                        for i,(centre,label) in enumerate(zip(bank['centres'],bank['labels'])):
                            radius=Q(0) if kind=='point' else Q(bank['radii_exact'][i])
                            misses=sum(abs(Q.from_float(float(centre[v]))-Q.from_float(float(y[v])))>radius+1 for v in plan['stations'])
                            if misses<=faults:poss.add(label)
                    predicted=next(iter(poss)) if len(poss)==1 else None
                    require(row[kind+'_prediction']==('' if predicted is None else str(predicted)),'Sampled exact model inference mismatch');checks+=1;sampled_model_checks+=1
    require(direct_wrong==0,'Wrong direct certificate')
    report={'status':'passed','trials':trials,'checks':checks,'sampled_exact_model_checks':sampled_model_checks,'model_wrong_decisions_outside_membership':wrong,'direct_wrong':direct_wrong,
            'scope':'Independent input/bank/calibration checks and every saved direct certificate/query cost; exact model inference sampled every 313th trial. No imports of repair implementation.'}
    (directory/'INDEPENDENT_REPLAY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
