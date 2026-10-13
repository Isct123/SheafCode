                                                                            
import itertools
import numpy as np
from .india import region_wide, complete_hourly, MaskedDataset, POLLUTANTS


def select_cohort(region, threshold=.5, min_training_observed_hours=720):
    wide,meta=region_wide(region)
    wide=complete_hourly(wide)
    T=len(wide); end=int(.6*T); va=int(.8*T)
    tr=wide.iloc[:end]
    candidates=[]
    for k in range(len(POLLUTANTS),1,-1):
        for P in itertools.combinations(POLLUTANTS,k):
            S=[]
            for s in wide.columns.get_level_values(0).unique():
                block=tr[s][list(P)]
                                                                               
                                                                     
                if (block.notna().mean()>=threshold).all() and (block.notna().sum()>=min_training_observed_hours).all():
                    S.append(s)
            if len(S)>=3:
                candidates.append((len(S)*k,k,tuple(P),S))
    if not candidates:
        raise ValueError(f'{region}: no cohort passes the fixed training coverage rule')
    _,_,P,S=max(candidates,key=lambda x:(x[0],x[1],x[2]))
    P=list(P); S=sorted(S)
    cols=[(s,p) for s in S for p in P]
    split=np.zeros(T,int); split[end:va]=1; split[va:]=2
    X=wide[cols].to_numpy().reshape(T,len(S),len(P))
    ds=MaskedDataset(region,S,P,meta.loc[S,['latitude','longitude']],meta.loc[S],X,wide.index,split)
    ds.selection={'rule':'fixed calendar, full-training-window coverage','threshold':threshold,
                  'train_end':str(wide.index[end-1]),'validation_end':str(wide.index[va-1]),
                  'timeline_start':str(wide.index[0]),'timeline_end':str(wide.index[-1]),
                  'stations':S,'pollutants':P,'n_train':end,'n_validation':va-end,'n_test':T-va,
                  'coverage':tr[cols].notna().mean().to_dict()}
                                                                   
    return ds.set_transform('none')
