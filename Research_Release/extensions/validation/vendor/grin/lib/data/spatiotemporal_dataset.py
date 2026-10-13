import numpy as np
import pandas as pd
from einops import rearrange

from .temporal_dataset import TemporalDataset


class SpatioTemporalDataset(TemporalDataset):
    def __init__(self, data,
                 index=None,
                 trend=None,
                 scaler=None,
                 freq=None,
                 window=24,
                 horizon=24,
                 delay=0,
                 stride=1,
                 **exogenous):
           
                                                                               

                    
                                                                                                
                    
                                                             
                     
                                                                                            
                    
                                                                                                  
                       
                                                                                                  
                    
                                                                                      
                           
                                                                    
                      
                                                       
                       
                                                                  
                     
                                              
           
        super(SpatioTemporalDataset, self).__init__(data,
                                                    index=index,
                                                    trend=trend,
                                                    scaler=scaler,
                                                    freq=freq,
                                                    window=window,
                                                    horizon=horizon,
                                                    delay=delay,
                                                    stride=stride,
                                                    **exogenous)

    def __repr__(self):
        return "{}(n_samples={}, n_nodes={})".format(self.__class__.__name__, len(self), self.n_nodes)

    @property
    def n_nodes(self):
        return self.data.shape[1]

    @staticmethod
    def check_dim(data):
        if data.ndim == 2:                                              
            data = rearrange(data, 's (n f) -> s n f', f=1)
        elif data.ndim == 1:
            data = rearrange(data, '(s n f) -> s n f', n=1, f=1)
        elif data.ndim == 3:
            pass
        else:
            raise ValueError(f'Invalid data dimensions {data.shape}')
        return data

    def dataframe(self):
        if self.n_channels == 1:
            return pd.DataFrame(data=np.squeeze(self.data, -1), index=self.index)
        raise NotImplementedError()
