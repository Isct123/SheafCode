                                                                                          
from pathlib import Path
from fractions import Fraction as Q
import argparse,csv,hashlib,json,math,time,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from decision_repair import master

ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'Research_Release/operational_validation_results'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)

def greedy(n,edges,required):
    if any(edge.bit_count()<required for edge in edges):return None
    mask=0
    while any((mask&e).bit_count()<required for e in edges):
        choices=[(sum(bool(e>>v&1) and (e&mask).bit_count()<required for e in edges),-v,v) for v in range(n) if not mask>>v&1]
        gain,_,v=max(choices)
        require(gain>0,'Greedy could not cover a feasible conflict')
        mask|=1<<v
    return mask

def simulated_readings(x,sign,bad,corruption):
    y=x+sign
                                                                             
    for v in range(len(x)):
        if abs(Q.from_float(float(y[v]))-Q.from_float(float(x[v])))>abs(sign):
            y[v]=np.nextafter(y[v],x[v])
    if bad is not None:y[bad]=-65. if corruption=='low' else 135.
    return y

def fit(train,cal,threshold=35.,eta=1.):
                                                                                   
    require(len(train)>0 and len(cal)>=20,'Insufficient complete train/calibration vectors')
    chosen=np.unique(np.linspace(0,len(train)-1,min(24,len(train))).astype(int))
    centres=train[chosen].copy();labels=(centres.max(1)>threshold).astype(int)
    scores=np.min(np.max(np.abs(cal[:,None,:]-centres[None,:,:]),axis=2),axis=1)
    rank=math.ceil(.9*(len(scores)+1));require(rank<=len(scores),'Calibration rank requires an infinite radius')
    raw=float(np.sort(scores)[rank-1])
    margins=np.abs(centres.max(1)-threshold)
    radii=np.minimum(raw,.95*margins)
    order=sorted(range(train.shape[1]),key=lambda v:(-float(np.mean(train[:,v]>threshold)),v))
    models={}
    exact_centres=[[Q.from_float(float(x)) for x in row] for row in centres]
    for kind,r in [('point',np.zeros(len(centres))),('box',radii)]:
        exact_r=[Q(0) if kind=='point' else min(Q.from_float(raw),Q(19,20)*abs(max(row)-Q.from_float(threshold))) for row in exact_centres]
                                                                                  
                                                                                             
        models[kind]={'centres':centres,'labels':labels,'radii':np.array([float(x) for x in exact_r]),
                      'exact_centres':exact_centres,'exact_radii':exact_r,'plans':{}}
        for faults in [0,1]:
            started=time.perf_counter();edges=[]
            for i in range(len(centres)):
                for j in range(i):
                    if labels[i]==labels[j]:continue
                    mask=sum(1<<v for v in range(train.shape[1]) if abs(exact_centres[i][v]-exact_centres[j][v])>exact_r[i]+exact_r[j]+2*Q.from_float(eta))
                    edges.append(mask)
            edges=sorted(set(edges));solution=master(train.shape[1],[1]*train.shape[1],edges,2*faults+1)
            selected=[] if solution is None else [v for v in range(train.shape[1]) if solution[0]>>v&1]
            exact_seconds=time.perf_counter()-started
            greedy_started=time.perf_counter();greedy_mask=greedy(train.shape[1],edges,2*faults+1);greedy_seconds=time.perf_counter()-greedy_started
            require((greedy_mask is None)==(solution is None),'Greedy feasibility mismatch')
            if greedy_mask is not None:
                require(greedy_mask.bit_count()>=len(selected),'Greedy beat the exact optimizer')
                require(all((greedy_mask&e).bit_count()>=2*faults+1 for e in edges),'Invalid greedy plan')
            models[kind]['plans'][faults]={'feasible':solution is not None,'stations':selected,'seconds':exact_seconds,'conflicts':len(edges),
                                         'greedy_queries':None if greedy_mask is None else greedy_mask.bit_count(),'greedy_seconds':greedy_seconds}
    return models,order,raw,rank,chosen

def membership(model,x):
    delta=np.max(abs(x-model['centres']),axis=1)
    inside=delta<=model['radii']
    near=np.flatnonzero(abs(delta-model['radii'])<=1e-10)
    for i in near:
        inside[i]=all(abs(Q.from_float(float(y))-z)<=model['exact_radii'][i] for y,z in zip(x,model['exact_centres'][i]))
    return bool(inside.any())

def infer_model(model,readings,faults,eta):
    selected=model['plans'][faults]['stations']
    if not model['plans'][faults]['feasible']:return None
    if not selected:
        labels=set(map(int,model['labels']));return next(iter(labels)) if len(labels)==1 else None
    distance=abs(model['centres'][:,selected]-readings[selected])
    tolerance=model['radii'][:,None]+eta
    mismatch=distance>tolerance
    near=np.argwhere(abs(distance-tolerance)<=1e-10)
    for i,k in near:
        v=selected[k]
        mismatch[i,k]=abs(model['exact_centres'][i][v]-Q.from_float(float(readings[v])))>model['exact_radii'][i]+Q.from_float(eta)
    labels=set(map(int,model['labels'][mismatch.sum(1)<=faults]))
    return next(iter(labels)) if len(labels)==1 else None

def direct(readings,order,faults,threshold,eta):
    high=0;all_safe=True
    for cost,v in enumerate(order,1):
        y=Q.from_float(float(readings[v]));threshold_q=Q.from_float(threshold);noise_q=Q.from_float(eta)
        high+=int(y-noise_q>threshold_q)
        all_safe &= y+noise_q<=threshold_q
        if high>faults:return 1,cost
    return (0 if faults==0 and all_safe else None),len(order)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    require(not (out/'SUMMARY.json').exists(),'Completed result exists; use a separate output directory to preserve evidence')
    with (ROOT/'Research_Release/expanded_validation_results/COHORT_RESULTS.csv').open(encoding='utf-8-sig',newline='') as f:
        cohorts=[(r['cohort'],ROOT/'Research_Release/expanded_validation_results'/r['cohort']/'dataset.npz') for r in csv.DictReader(f)]
    cohorts += [('original_india',ROOT/'Research_Release/recent_data_results/delhi/dataset.npz'),('original_usa',ROOT/'Research_Release/recent_data_results/epa_la/dataset.npz')]
    hashes={str(p.relative_to(ROOT)):sha(p) for _,p in cohorts}
    lock={'status':'frozen_before_followup_evaluation','input_sha256':hashes,'protocol_sha256':sha(Path(__file__).with_name('HELDOUT_PROTOCOL.md')),
          'code_sha256':sha(Path(__file__)),'threshold':35.,'eta':1.,'calibration_split':3,'bank_split':0,'test_split':2,
          'scope':'Exploratory chronological follow-up after earlier test access; not external preregistration.'}
    (out/'PROTOCOL_LOCK.json').write_text(json.dumps(lock,indent=2))
    summary=[];banks={};hour_rows=[];start=time.time();checks=0
    fields=['cohort','timestamp','faults','error_sign','corrupt_station','corruption','truth','point_member','box_member','point_prediction','box_prediction','direct_prediction','point_queries','box_queries','direct_queries','training_order_queries','full_queries','full_prediction']
    with (out/'TRIALS.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,path in cohorts:
            with np.load(path,allow_pickle=False) as data:
                pm=list(map(str,data['channels'])).index('PM2.5');X=data['X'][:,:,pm];split=data['split'];times=data['times'];stations=list(map(str,data['stations']))
                complete=np.isfinite(X).all(1)
                ids={part:np.flatnonzero(complete&(split==part)) for part in [0,3,2]}
                counts={str(part):int(len(ix)) for part,ix in ids.items()}
                exclusions={str(part):int(np.sum(split==part)-len(ids[part])) for part in ids}
                if not len(ids[0]) or len(ids[3])<20 or not len(ids[2]):
                    summary.append({'cohort':name,'status':'insufficient_complete_train_cal_test','complete_hours':counts,'excluded_hours':exclusions});continue
                require(all(times[split==a].max()<times[split==b].min() for a,b in [(0,3),(3,2)]),'Chronology violation')
                models,order,radius,rank,selected=fit(X[ids[0]],X[ids[3]])
                frozen={'stations':stations,'centres':models['box']['centres'].tolist(),'labels':models['box']['labels'].tolist(),
                        'radii_exact':[str(q) for q in models['box']['exact_radii']], 'uncapped_calibrated_radius':radius,'calibration_rank':rank,
                        'training_timestamps':[str(t) for t in times[ids[0][selected]]],'query_order':order,
                        'plans':{kind:{str(k):v for k,v in model['plans'].items()} for kind,model in models.items()}}
                banks[name]=frozen
                                                                         
                (out/'FROZEN_BANKS.json').write_text(json.dumps(banks,indent=2))
                rows=[]
                for index in ids[2]:
                    x=X[index];truth=int(max(x)>35.);members={k:membership(m,x) for k,m in models.items()}
                    hour_rows.append({'cohort':name,'timestamp':str(times[index]),'truth':truth,**{k+'_member':int(v) for k,v in members.items()}})
                    n=len(x)
                    trials=[(0,s,None,'none') for s in [-1,0,1]]
                    trials += [(1,s,v,c) for s in [-1,1] for v in range(n) for c in ['low','high']]
                    for faults,sign,bad,corruption in trials:
                        y=simulated_readings(x,sign,bad,corruption)
                        prediction={kind:infer_model(model,y,faults,1.) for kind,model in models.items()}
                        initial=models['box']['plans'][faults]['stations']
                        policy_order=[v for v in order if v in initial]+[v for v in order if v not in initial]
                        label,cost=direct(y,policy_order,faults,35.,1.)
                        full_label,_=direct(y,list(range(n)),faults,35.,1.)
                        simple_label,simple_cost=direct(y,order,faults,35.,1.)
                        require(label==full_label,'Sequential/full measurement decision mismatch');checks+=1
                        require(label==simple_label,'Training-order/direct label mismatch');checks+=1
                        require(label is None or label==truth,'Direct certificate was wrong');checks+=1
                        for kind in models:
                            require(not members[kind] or prediction[kind] is None or prediction[kind]==truth,'In-model decision incorrect');checks+=1
                        row={'cohort':name,'timestamp':str(times[index]),'faults':faults,'error_sign':sign,'corrupt_station':'' if bad is None else stations[bad],
                             'corruption':corruption,'truth':truth,'point_member':int(members['point']),'box_member':int(members['box']),
                             **{k+'_prediction':'' if prediction[k] is None else prediction[k] for k in models},
                             'direct_prediction':'' if label is None else label,
                             **{k+'_queries':'' if not models[k]['plans'][faults]['feasible'] else len(models[k]['plans'][faults]['stations']) for k in models},
                             'direct_queries':cost,'training_order_queries':simple_cost,'full_queries':n,'full_prediction':'' if full_label is None else full_label}
                        writer.writerow(row);rows.append(row)
                metrics={}
                for faults in [0,1]:
                    subset=[r for r in rows if r['faults']==faults];a={}
                    for kind in ['point','box','direct']:
                        issued=[r for r in subset if r[kind+'_prediction']!=''];wrong=sum(r[kind+'_prediction']!=r['truth'] for r in issued)
                        positives=[r for r in subset if r['truth']==1]
                        a[kind]={'trials':len(subset),'issued':len(issued),'wrong':wrong,'abstained':len(subset)-len(issued),
                                 'precision':(len(issued)-wrong)/len(issued) if issued else None,
                                 'sensitivity':sum(r[kind+'_prediction']==1 for r in positives)/len(positives) if positives else None,
                                 'mean_queries':float(np.mean([r[kind+'_queries'] for r in subset if r[kind+'_queries']!=''])) if any(r[kind+'_queries']!='' for r in subset) else None}
                    common=[r for r in subset if r['direct_prediction']!='']
                    a['query_saving_common_certified']=1-sum(r['direct_queries'] for r in common)/sum(r['full_queries'] for r in common) if common else None
                    metrics[str(faults)]=a
                member_rows=[r for r in hour_rows if r['cohort']==name]
                summary.append({'cohort':name,'status':'complete','stations':len(stations),'complete_hours':counts,'excluded_hours':exclusions,
                                'test_box_members':sum(r['box_member'] for r in member_rows),'test_point_members':sum(r['point_member'] for r in member_rows),
                                'calibration_box_members':sum(membership(models['box'],x) for x in X[ids[3]]),
                                'metrics':metrics})
                print(name,'test hours',len(ids[2]),'box coverage',summary[-1]['test_box_members']/len(ids[2]),flush=True)
    with (out/'HOUR_MEMBERSHIP.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['cohort','timestamp','truth','point_member','box_member']);writer.writeheader();writer.writerows(hour_rows)
    require(all(sha(ROOT/name)==value for name,value in hashes.items()),'Dataset changed during run')
    report={'status':'completed_and_checked','cohorts':summary,'attempted_cohorts':len(cohorts),'checks':checks,'seconds':time.time()-start,
            'scope':'Training/calibration-fitted scenario repair, later held-out query simulations, counted model violations and direct interval/counting certificates. Not field deployment or emissions reduction.'}
    (out/'SUMMARY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks':checks,'seconds':report['seconds']},indent=2))

if __name__=='__main__':main()
