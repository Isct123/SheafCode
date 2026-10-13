import json
from pathlib import Path
import numpy as np

                                                                          
                                                                         
D = np.eye(3)
fault = np.array([100., .1, 0.])
r = np.array([0., 0., 1.])
s = D @ fault + r
true_res = np.linalg.norm(s - D[:, [0, 1]] @ np.linalg.lstsq(D[:, [0, 1]], s, rcond=None)[0])
wrong_res = np.linalg.norm(s - D[:, [0, 2]] @ np.linalg.lstsq(D[:, [0, 2]], s, rcond=None)[0])
assert np.linalg.norm(fault) > 2*np.linalg.norm(r)
assert wrong_res < true_res

                                                                          
                                                                     
D = np.array([[1., -1., 0.], [0., 1., -1.], [-2., 0., 1.]])
sigma = float(np.linalg.svd(D, compute_uv=False)[-1])
f = np.array([100., .001, 0.]); r2 = D[:,2].copy(); s2=D@f+r2
true2=np.linalg.norm(s2-D[:,:2]@np.linalg.lstsq(D[:,:2],s2,rcond=None)[0])
wrong2=np.linalg.norm(s2-D[:,[0,2]]@np.linalg.lstsq(D[:,[0,2]],s2,rcond=None)[0])
assert np.linalg.norm(f)*sigma > 2*np.linalg.norm(r2)
assert wrong2 < true2

                                                                      
                                                                         
                                                                        
hist=[1.]; a=.1; errors=[]
for k in range(500):
    level=min(max(1-a,0.),1.)
    q=float(np.quantile(hist,level)) if level<1 else 1.5*max(hist)
    score=2*max(hist)
    err=float(score>q); errors.append(err)
    a += .005*(.1-err); hist.append(score)
assert sum(errors)==500

                                                                      
                                                                          
e=np.array([.2,.5,1.,2.,3.]); scale=7.
assert np.isclose(np.quantile(e/scale,.9)*scale,np.quantile(e,.9))

out={
 'robust_support_counterexample': {'true_residual':true_res,'wrong_residual':wrong_res,'stated_condition_holds':True},
 'actual_generic_sheaf_counterexample': {'sigma_min':sigma,'true_residual':true2,'wrong_residual':wrong2,'stated_condition_holds':True},
 'adaptive_endpoint_counterexample': {'n':500,'coverage':1-np.mean(errors),'final_alpha':a},
 'constant_certificate_scale_cancels':True,
}
print(json.dumps(out,indent=2))
(Path(__file__).parent/'claim_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
