import torch
from torch import nn

from .gcrnn import GCGRUCell


class MPGRUImputer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 u_size=None,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 support_len=2,
                 n_nodes=None,
                 layer_norm=False,
                 autoencoder_mode=False):
        super(MPGRUImputer, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.ff_size = int(ff_size) if ff_size is not None else 0
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size                                         

                                                                 
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

                 
        if self.ff_size:
            self.pred_readout = nn.Sequential(
                nn.Conv1d(in_channels=self.hidden_size, out_channels=self.ff_size, kernel_size=1),
                nn.PReLU(),
                nn.Conv1d(in_channels=self.ff_size, out_channels=self.input_size, kernel_size=1)
            )
        else:
            self.pred_readout = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)

                                               
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

        self.autoencoder_mode = autoencoder_mode

    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, mask=None, u=None, h=None):
                                           
        *_, steps = x.size()

                                         
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

                                                                   
        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]

                       
        predictions, states = [], []
        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None
                                                               
            x_s_hat = self.pred_readout(h_s)
                                         
            predictions.append(x_s_hat)
            states.append(torch.stack(h, dim=0))
                                                          
            x_s = torch.where(m_s, x_s, x_s_hat)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)                                         
                                                                          
            h = self.update_state(inputs, h, adj)

                                                               
        if self.autoencoder_mode:
            states = states[1:] + [torch.stack(h, dim=0)]

                                                              
        predictions = torch.stack(predictions, dim=-1)
        states = torch.stack(states, dim=-1)

        return predictions, states
