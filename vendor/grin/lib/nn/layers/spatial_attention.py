import torch.nn as nn

from einops import rearrange


class SpatialAttention(nn.Module):
    def __init__(self, d_in, d_model, nheads, dropout=0.):
        super(SpatialAttention, self).__init__()
        self.lin_in = nn.Linear(d_in, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)

    def forward(self, x, att_mask=None, **kwargs):
                                                     

             
                                                              
                                                               
                                                                                 

              
                                              
           
        b, s, n, f = x.size()
        x = rearrange(x, 'b s n f -> n (b s) f')
        x = self.lin_in(x)
        x = self.self_attn(x, x, x, attn_mask=att_mask)[0]
        x = rearrange(x, 'n (b s) f -> b s n f', b=b, s=s)
        return x
