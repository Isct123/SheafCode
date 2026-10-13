                                                                         
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from graph_benchmark import official,torch
import numpy as np
def test_official_graph_model_shape_and_finite_gradient():
    GRIN=official();torch.manual_seed(1);model=GRIN(np.ones((3,3))-np.eye(3),2,8,8,0.,d_emb=2)
    x=torch.randn(2,8,3,2);mask=torch.ones_like(x,dtype=torch.bool);mask[:,4:,1]=False;x[~mask]=0
    pred,aux=model(x,mask=mask);assert pred.shape==x.shape
    loss=pred[~mask].square().mean()+sum(a[~mask].square().mean() for a in aux);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():out=model(x,mask=mask)
    assert out.shape==x.shape and torch.equal(out[mask],x[mask])
