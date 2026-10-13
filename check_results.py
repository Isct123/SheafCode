import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common
import numpy as np
import pandas as pd
from reconstruction import predict,score

def main():
    parser=argparse.ArgumentParser(description='Recompute reported outage gains and replay the first recorded station-day for every cohort.')
    parser.add_argument('--results',type=Path,default=Path('results'))
    args=parser.parse_args();common.OUT=args.results.resolve();checks=[]
    for region in common.regions():
        folder=common.OUT/region
        frame=pd.read_csv(folder/'outage_cases.csv',dtype={'station':str})
        recorded=json.loads((folder/'reconstruction_summary.json').read_text())
        for scope,f in [('primary_every_third_day',frame[frame.primary]),('all_eligible_test_days',frame)]:
            result=common.paired(f,'sheaf_causal','var_causal')
            old=recorded[scope]['var_causal']
            assert result['cases']==old['cases']
            assert np.isclose(result['gain_percent'],old['gain_percent'],rtol=0,atol=1e-10)
            checks.append({'cohort':region,'scope':scope,**result})
        row=frame.iloc[0];ds=common.load(region);model=common.models(region)
        ix=dict(common.blocks(ds,2))[row.day];v=list(map(str,ds['stations'])).index(row.station)
        predicted=predict(ds['Z'][ix],v,24,model,ds)
        truth=ds['Z'][ix[24:],v]
        subset=frame[(frame.day==row.day)&(frame.station==row.station)]
        for r in subset.itertuples():
            replay=score(predicted[r.method],truth,ds['sd'][v])
            assert np.isclose(replay['mse'],r.mse,rtol=1e-9,atol=1e-10)
        hidden=ds['Z'][ix].copy();hidden[24:,v]=123456
        assert np.allclose(predicted['sheaf_causal'],model['sheaf'].causal(hidden,v))
    if not checks:raise RuntimeError('No recorded reconstruction results')
    print(json.dumps(common.clean({'status':'passed','comparisons':checks}),indent=2))

if __name__=='__main__':main()
