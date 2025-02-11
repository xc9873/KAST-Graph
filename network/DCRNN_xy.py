import torch
from torch import nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, K:int, input_dim:int, hidden_dim:int, bias=True, activation=nn.ReLU):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.activation = activation() if activation is not None else None
        self.init_params(n_supports=K)

    def init_params(self, n_supports:int, b_init=0):
        self.W = nn.Parameter(torch.empty(n_supports*self.input_dim, self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)      # sampled from a normal distribution N(0, std^2), also known as Glorot initialization
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)

    def forward(self, G:torch.Tensor, x:torch.Tensor):
        '''
        Batch-wise graph convolution operation on given n support adj matrices
        :param G: support adj matrices - torch.Tensor (K, n_nodes, n_nodes)
        :param x: graph feature/signal - torch.Tensor (batch_size, n_nodes, input_dim)
        :return: hidden representation - torch.Tensor (batch_size, n_nodes, hidden_dim)
        '''
        assert self.K == G.shape[0]

        support_list = list()
        for k in range(self.K):
            support = torch.einsum('ij,bjp->bip', [G[k,:,:], x])
            support_list.append(support)
        support_cat = torch.cat(support_list, dim=-1)

        output = torch.einsum('bip,pq->biq', [support_cat, self.W])
        if self.bias:
            output += self.b
        output = self.activation(output) if self.activation is not None else output
        return output

    def __repr__(self):
        return self.__class__.__name__ + f'({self.K} * input {self.input_dim} -> hidden {self.hidden_dim})'


class DCGRU_Cell(nn.Module):
    def __init__(self, num_nodes:int, input_dim:int, hidden_dim:int, K:int, bias=True, activation=None):
        super(DCGRU_Cell, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv_gate = GCN(K=K,
                             input_dim=input_dim+hidden_dim,
                             hidden_dim=hidden_dim*2,       # for update_gate, reset_gate
                             bias=bias,
                             activation=activation)
        self.conv_cand = GCN(K=K,
                             input_dim=input_dim+hidden_dim,
                             hidden_dim=hidden_dim,       # for candidate
                             bias=bias,
                             activation=activation)

    def init_hidden(self, batch_size:int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(batch_size, self.num_nodes, self.hidden_dim))
        return hidden

    def forward(self, P:torch.Tensor, x_t:torch.Tensor, h_t_1:torch.Tensor):
        assert len(P.shape) == len(x_t.shape) == len(h_t_1.shape) == 3, 'DCGRU cell must take in 3D tensor as input [x, h]'

        x_h = torch.cat([x_t, h_t_1], dim=-1)
        x_h_conv = self.conv_gate(G=P, x=x_h)

        z, r = torch.split(x_h_conv, self.hidden_dim, dim=-1)
        update_gate = torch.sigmoid(z)
        reset_gate = torch.sigmoid(r)

        candidate = torch.cat([x_t, reset_gate*h_t_1], dim=-1)
        cand_conv = torch.tanh(self.conv_cand(G=P, x=candidate))

        h_t = (1.0 - update_gate) * h_t_1 + update_gate * cand_conv
        return h_t


class DCGRU_Encoder(nn.Module):
    def __init__(self, num_nodes:int, input_dim:int, hidden_dim, K:int, num_layers:int,
                 bias=True, activation=None, return_all_layers=False):
        super(DCGRU_Encoder, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(DCGRU_Cell(num_nodes=num_nodes,
                                             input_dim=cur_input_dim,
                                             hidden_dim=self.hidden_dim[i],
                                             K=K,
                                             bias=bias,
                                             activation=activation))

    def forward(self, P:torch.Tensor, x_seq:torch.Tensor, h_0_l=None):
        '''
            P: (K, N, N)
            x_seq: (B, T, N, C)
            h_0_l: [(B, N, C)] * L
            return - out_seq_lst: [(B, T, N, C)] * L
                     h_t_lst: [(B, N, C)] * L
        '''
        assert len(x_seq.shape) == 4, 'DCGRU must take in 4D tensor as input x_seq'
        batch_size, seq_len, _, _ = x_seq.shape
        if h_0_l is None:
            h_0_l = self._init_hidden(batch_size)

        out_seq_lst = list()    # layerwise output seq
        h_t_lst = list()        # layerwise last state
        in_seq_l = x_seq        # current input seq

        for l in range(self.num_layers):
            h_t = h_0_l[l]
            out_seq_l = list()
            for t in range(seq_len):
                h_t = self.cell_list[l](P=P, x_t=in_seq_l[:, t, :, :], h_t_1=h_t)
                out_seq_l.append(h_t)

            out_seq_l = torch.stack(out_seq_l, dim=1)  # (B, T, N, C)
            in_seq_l = out_seq_l  # update input seq

            out_seq_lst.append(out_seq_l)
            h_t_lst.append(h_t)

        if not self.return_all_layers:
            out_seq_lst = out_seq_lst[-1:]
            h_t_lst = h_t_lst[-1:]
        return out_seq_lst, h_t_lst

    def _init_hidden(self, batch_size:int):
        h_0_l = []
        for i in range(self.num_layers):
            h_0_l.append(self.cell_list[i].init_hidden(batch_size))
        return h_0_l

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class DCGRU_Decoder(nn.Module):       # projected output as input at the next timestep
    def __init__(self, num_nodes:int, out_horizon:int, out_dim:int, hidden_dim, K:int, num_layers:int,
                 bias=True, activation=None):
        super(DCGRU_Decoder, self).__init__()
        self.num_nodes = num_nodes
        self.out_horizon = out_horizon      # output steps
        self.out_dim = out_dim
        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        self.num_layers = num_layers
        self.bias = bias
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.out_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(DCGRU_Cell(num_nodes=num_nodes,
                                             input_dim=cur_input_dim,
                                             hidden_dim=self.hidden_dim[i],
                                             K=K,
                                             bias=bias,
                                             activation=activation))
        #self.out_projector = nn.Sequential(nn.Linear(in_features=self.hidden_dim[-1], out_features=out_dim, bias=bias), nn.ReLU())
        self.out_projector = nn.Linear(in_features=self.hidden_dim[-1], out_features=out_dim, bias=bias)

    def forward(self, P:torch.Tensor, x_t:torch.Tensor, h_0_l:list):
        '''
            P: (K, N, N)
            x_t: (B, N, C)
            h_0_l: [(B, N, C)] * L
        '''
        assert len(x_t.shape) == 3, 'DCGRU cell decoder must take in 3D tensor as input x_t'

        h_t_lst = list()        # layerwise hidden state
        x_in_l = x_t

        for l in range(self.num_layers):
            h_t_l = self.cell_list[l](P=P, x_t=x_in_l, h_t_1=h_0_l[l])
            h_t_lst.append(h_t_l)
            x_in_l = h_t_l      # update input for next layer

        output = self.out_projector(h_t_l)      # output
        return output, h_t_lst

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class DCRNN(nn.Module):
    def __init__(self, num_nodes:int, input_dim:int, hidden_dim, K:int,
                 num_layers:int, out_horizon:int, bias=True, activation=None):
        super(DCRNN, self).__init__()
        self.K = K
        self.encoder = DCGRU_Encoder(num_nodes=num_nodes, input_dim=input_dim, hidden_dim=hidden_dim, K=K,
                                     num_layers=num_layers, bias=bias, activation=activation, return_all_layers=True)
        self.decoder = DCGRU_Decoder(num_nodes=num_nodes, out_dim=input_dim, hidden_dim=hidden_dim, K=K,
                                     num_layers=num_layers, bias=bias, activation=activation, out_horizon=out_horizon)
        self.graph_encoder = nn.Linear(1, 10)

        self.fully2 = nn.Linear(28, 1)
        self.fully3 = nn.Linear(1+1, 1)

    def sim(self, z1, z2, b_cos):
        if b_cos:
            view1, view2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        score = view1 @ view2.T
        return score

    def InfoNCE(self, z1: torch.Tensor, z2: torch.Tensor, temperature=0.5, b_cos=True):
        f = lambda x: torch.exp(x / temperature)

        refl_sim = f(self.sim(z1, z1, b_cos))
        between_sim = f(self.sim(z1, z2, b_cos))
        loss = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        return loss.mean()

    def forward(self, graph, x, XY, adj_update=True, use_xy=True, moe=True, gcl=True):
        '''
            P: (K, N, N)
            x_seq: (B, T, N, C)
        '''
        x = x.permute(0, 2, 1, 3)
        assert len(x.shape) == 4, 'DCGRU must take in 4D tensor as input x_seq'

        # fixme
        graph = graph.unsqueeze(-1)
        if adj_update:
            graph_feature = self.graph_encoder(graph)
            graph_feature = (graph_feature - graph_feature.min()) / (graph_feature.max() - graph_feature.min())
            graph = (torch.eye(graph.shape[1]).cuda().unsqueeze(-1) +
                 0.0001 * graph + 0.0001 *  (graph_feature * graph_feature.transpose(0,1)).mean(dim=-1).unsqueeze(-1)).squeeze(-1)
        else:
            graph = graph.squeeze(-1)

        graph = graph.unsqueeze(0)
        assert graph.shape[0] == self.K

        # encoding
        _, h_t_lst = self.encoder(P=graph, x_seq=x, h_0_l=None)      # encoder returns layerwise last hidden state [(B, N, C)] * L
        # decoding
        deco_input = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), device=x.device)       # original initialization

        # gcl
        if gcl:
            hidden_lst = torch.stack(h_t_lst)
            hidden_lst = hidden_lst.transpose(0, 1)
            hidden_lst = hidden_lst.reshape(hidden_lst.shape[0], -1)
            # augmentation
            # random mask
            aug_hidden_lst = torch.dropout(hidden_lst, p=0.1, train=True)
            # add noise
            aug_hidden_lst = aug_hidden_lst + 0.001 * torch.randn_like(aug_hidden_lst)

            self.cl_loss = self.InfoNCE(hidden_lst, aug_hidden_lst)
        else:
            pass

        outputs = list()
        for t in range(self.decoder.out_horizon):
            output, h_t_lst = self.decoder(P=graph, x_t=deco_input, h_0_l=h_t_lst)
            deco_input = output     # update decoder input
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)   # (B, horizon, N, C)
        outputs = outputs.squeeze().permute(0, 2, 1)

        outputs = outputs.unsqueeze(-1)
        #print(outputs.shape)
        if use_xy:
            # 使用先验
            XY = self.fully2(XY.permute(0, 1, 3, 2)).relu()
            # TODO New_form = MOE
            if moe:
                gate = torch.sigmoid(self.fully3(torch.cat((outputs, XY), dim=-1)))
                outputs = outputs * gate + XY * (1-gate)
                outputs = outputs.squeeze()
            else:
                # TODO Origion_form
                outputs = self.fully3(torch.cat((outputs, XY), dim=-1)).squeeze()
        else:
            # 不使用先验
            outputs = outputs.squeeze()

        outputs = torch.clip(outputs, 0, 1)

        return outputs

'''
if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    od_flow = torch.randn(2, 12, 12, 1).cuda()
    graph = torch.randn(12, 12).cuda()
    net = DCRNN(num_nodes=12,
                input_dim=1,
                hidden_dim=64,
                K=1,
                num_layers=2,
                out_horizon=3).cuda()

    od_pre = net(graph,
                 od_flow)
    print(od_pre.shape)  #  B*T*N*N*1
'''