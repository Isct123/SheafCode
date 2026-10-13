                                                                                 
import sys
from pathlib import Path
from fractions import Fraction as Q
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from heldout import direct,greedy,fit,simulated_readings

def test_direct_boundary_and_unknown_corruption():
    assert direct(np.array([34.,34.]),[0,1],0,35.,1.)==(0,2)
    assert direct(np.array([36.,34.]),[0,1],0,35.,1.)==(None,2)
    assert direct(np.array([36.1,34.]),[0,1],0,35.,1.)==(1,1)
                                                        
    assert direct(np.array([135.,0.]),[0,1],1,35.,1.)==(None,2)
    assert direct(np.array([135.,37.]),[0,1],1,35.,1.)==(1,2)
                                                                              
    assert direct(np.array([0.,0.]),[0,1],1,35.,1.)==(None,2)

def test_greedy_feasibility_and_trivial_bank():
    assert greedy(3,[3,6],1)==2
    assert greedy(3,[3,6],3) is None
    assert greedy(3,[],1)==0

def test_training_calibration_fit_and_label_preservation():
    train=np.array([[0.,0.],[40.,10.],[20.,45.],[2.,3.]])
    cal=np.repeat([[1.,1.],[41.,10.]],10,axis=0)
    models,order,_,rank,_=fit(train,cal)
    assert rank==19 and sorted(order)==[0,1]
    for centre,label,radius in zip(models['box']['exact_centres'],models['box']['labels'],models['box']['exact_radii']):
        assert (max(centre)-radius>Q(35)) if label else (max(centre)+radius<=Q(35))
    one_label,_,_,_,_=fit(np.repeat([[1.,2.]],4,axis=0),np.repeat([[1.,2.]],20,axis=0))
    assert one_label['box']['plans'][0]['stations']==[]

def test_simulated_clean_noise_respects_exact_bound():
    x=np.array([35.,34.99999999999999,1e-12,37.123456789])
    for sign in [-1,0,1]:
        readings=simulated_readings(x,sign,None,'none')
        assert all(abs(Q.from_float(float(a))-Q.from_float(float(b)))<=abs(sign) for a,b in zip(x,readings))
