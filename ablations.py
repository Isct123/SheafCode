import copy
from common import *
from reconstruction import score

def identity_spatial(model, edges):
    result = copy.deepcopy(model)
    p, n = (result.p, result.n)
    rows = []
    for u, v in edges:
        block = np.zeros((p, n * p))
        block[:, u * p:(u + 1) * p] = np.eye(p)
        block[:, v * p:(v + 1) * p] = -np.eye(p)
        rows.append(block)
    result.D = np.vstack(rows)
    result.S = result.D.T @ result.D + result.lam * result.P
    result.precisions = {}
    return result

def main():
    for region in regions():
        ds=load(region);fitted=models(region);base=fitted['sheaf']
        identity=identity_spatial(base,fitted['edges'])
        no24=copy.deepcopy(base);no24.temporal.pop(24,None);no24.precisions={}
        rows=[];planned=blocks(ds,2);primary={day for day,ix in planned[::3]}
        for day,ix in planned:
            for v,station in enumerate(ds['stations']):
                truth=ds['Z'][ix[24:],v]
                if np.isfinite(truth).sum()<12*truth.shape[1]:continue
                for method,model in [('identity_spatial_fixed',identity),('without_24h',no24)]:
                    prediction=model.causal(ds['Z'][ix],v)
                    rows.append({'day':day,'station':str(station),'method':method,'primary':day in primary,**score(prediction,truth,ds['sd'][v])})
        frame=pd.DataFrame(rows);frame_save(region+'/ablation_cases.csv',frame)
        reference=pd.read_csv(OUT/region/'outage_cases.csv',dtype={'station':str})
        joined=pd.concat([frame,reference],ignore_index=True)
        save(region+'/ablation_summary.json',{scope:{name:paired(ff,'sheaf_causal',name) for name in ['identity_spatial_fixed','without_24h']} for scope,ff in [('primary',joined[joined.primary]),('all_days',joined)]})
