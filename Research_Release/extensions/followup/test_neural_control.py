import numpy as np
from neural import features,fit,predict

def test_hidden_station_never_enters_features():
    z=np.arange(24,dtype=float).reshape(4,3,2);adj=np.ones((3,3))-np.eye(3)
    a=features(z,1,adj);z[:,1]=100000
    assert np.array_equal(a,features(z,1,adj))

def test_training_learns_nonlinear_signal():
    rng=np.random.default_rng(4);x=rng.normal(size=(400,3));y=np.tanh(x[:,:1])
    model=fit(x,y,16,0.,5,steps=300)
    assert np.mean((predict(x,model)-y)**2)<.03
