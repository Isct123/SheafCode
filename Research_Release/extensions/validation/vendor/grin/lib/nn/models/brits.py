import torch
from torch import nn

from ..layers import BRITS


class BRITSNet(nn.Module):
    def __init__(self,
                 d_in,
                 d_hidden=64):
        super(BRITSNet, self).__init__()
        self.birits = BRITS(input_size=d_in,
                            hidden_size=d_hidden)

    def forward(self, x, mask=None, **kwargs):
                                       
        imputations, predictions = self.birits(x, mask=mask)
                                                               
        out = torch.mean(imputations, dim=1)                               
        predictions = torch.cat(predictions, dim=1)                                                           
                 
        imputations = torch.transpose(imputations, 0, 1)                                                
        predictions = torch.transpose(predictions, 0, 1)                                                
        return out, imputations, predictions

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-in', type=int)
        parser.add_argument('--d-hidden', type=int, default=64)
        return parser
