                                                                             
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from fault_episodes import Decoder,week_ci

def main():
    choicepath=ROOT/'validation_results/fault_choices.json';choices=json.loads(choicepath.read_text());rows=[];rng=np.random.default_rng(9623)
    lock('raw_gain_lock.json',{'source_sha256':sha(__file__),'classifier_sha256':sha(choicepath),'seed':9623,'gain_factors':[1.2,1.5,2.0]})
    for region in ['delhi2025','bengaluru_raw']:
        ds,m,folder=local(region);Z=ds.Z();D=np.load(folder/'model.npz')['delta'];decoder=Decoder(Z[ds.split==0],D,ds.n,ds.p,choices[region])
        blocks=day_blocks(ds.times,ds.split,2);blocks=[blocks[i] for i in np.unique(np.linspace(0,len(blocks)-1,min(48,len(blocks))).astype(int))]
        for episode,(day,ix) in enumerate(blocks):
            ids=ix[24:];eligible=np.flatnonzero(np.isfinite(Z[ids]).all((0,2)))
            if len(eligible)<2:continue
            for count in [1,2]:
                F=tuple(sorted(rng.choice(eligible,count,replace=False).tolist()))
                for factor in [1.2,1.5,2.0]:
                    for j,t in enumerate(ids):
                        raw=ds.X[t,list(F)]*factor;transformed=ds._fwd(raw)-ds.mean[list(F)]
                        if ds.hod is not None:transformed-=ds.hod[ds.times[t].hour,list(F)]
                        y=Z[t].copy();y[list(F)]=transformed/ds.std[list(F)]
                        for method,support in decoder.predict(y).items():
                            rows.append({'region':region,'day':day,'timestamp':str(ds.times[t]),'episode':episode,'gain_factor':factor,'fault_count':count,
                                'method':method,'true_support':str(F),'estimated_support':str(support),'alarm':bool(support),'exact':support==F,
                                'week':str(ds.times[t].to_period('W-SUN'))})
        print(region,'raw-gain faults complete',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'raw_gain_fault_cases.csv',index=False);summary=[]
    for key,g in frame.groupby(['region','gain_factor','fault_count','method']):
        ci=week_ci(g);summary.append(dict(zip(['region','gain_factor','fault_count','method'],key))|{'cases':len(g),'episodes':g.episode.nunique(),
            'alarm_rate':g.alarm.mean(),'exact_end_to_end':g.exact.mean(),'exact_ci_low':ci[0],'exact_ci_high':ci[1]})
    pd.DataFrame(summary).to_csv(OUT/'raw_gain_fault_summary.csv',index=False)
    save('raw_gain_scope.json',{'status':'completed','case_rows':len(frame),'scope':'Artificial multiplicative calibration errors applied in original raw reading coordinates, then passed through the frozen transform. Same fixed classifier/thresholds and candidate universe; no retuning or field fault labels.'})

if __name__=='__main__':main()
