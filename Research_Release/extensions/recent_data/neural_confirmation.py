                                                                                
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from neural import official,cases,evaluate,torch

def main():
    GRIN,commit=official();region='delhi';ds=load(region);m=models(region);n,p=ds['Z'].shape[1:];adj=np.zeros((n,n),np.float32)
    for u,v in m['edges']:adj[u,v]=adj[v,u]=1
    saved=pd.read_csv(OUT/region/'grin_cases.csv',dtype={'station':str});choices=json.loads((OUT/region/'grin_choices.json').read_text());test=cases(ds,2);checks=[]
    for choice in choices:
        state=torch.load(OUT/region/f"grin_seed{choice['seed']}_width{choice['width']}_training.pt",map_location='cpu',weights_only=False);model=GRIN(adj,p,choice['width'],choice['width'],0.,d_emb=8);model.load_state_dict(state['best']);actual=pd.DataFrame(evaluate(model,test));joined=actual.merge(saved[saved.seed==choice['seed']],on=['day','station'],suffixes=('_replay','_saved'),validate='one_to_one')
        if len(joined)!=242 or not np.allclose(joined.mse_replay,joined.mse_saved,rtol=1e-7,atol=1e-9):raise AssertionError('Checkpoint predictions changed')
        checks.append({'seed':choice['seed'],'cases':len(joined),'maximum_absolute_mse_difference':float(abs(joined.mse_replay-joined.mse_saved).max()),'passed':True});print('Neural checkpoint replay passed, seed',choice['seed'],flush=True)
    save('NEURAL_CONFIRMATION_REPLAY.json',{'status':'passed','checks':checks,'source_sha256':sha(__file__),'vendor_commit':commit,'scope':'Every validation-chosen checkpoint re-evaluated on all test cases; no retraining, no seed selection and no independent neural architecture claimed'})
if __name__=='__main__':main()
