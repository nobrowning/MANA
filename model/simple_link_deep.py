import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from .data_arg import DataArgs


class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1:
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        else:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False


class Transformer(nn.Module):
    def __init__(self, n_channels, att_drop=0., act='none', num_heads=1):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C))


class L2Norm(torch.nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)


class SimpleLinkDeep(nn.Module):
    def __init__(self, dataset, data_args: DataArgs, nfeat, hidden, nclass,
                 dropout, input_drop, att_drop, n_layers,
                 act, residual=False, bns=False, **kwargs):
        super(SimpleLinkDeep, self).__init__()
        self.dataset = dataset
        self.residual = residual
        self.src_tgt_key = data_args.src_tgt_key
        self.dst_tgt_key = data_args.dst_tgt_key
        
        self.num_feats = data_args.dst_num_feats + data_args.src_num_feats
        self.num_scores = data_args.dst_num_feats*2 * data_args.src_num_feats*2

        self.embedings = nn.ModuleDict({})
        for k, v in data_args.src_data_size.items():
            if v != nfeat:
                self.embedings[k] = nn.Linear(v, nfeat)
        for k, v in data_args.dst_data_size.items():
            if v != nfeat:
                self.embedings[k] = nn.Linear(v, nfeat)

        self.feat_project_layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, self.num_feats, bias=True, cformat='channel-first'),
            # nn.LayerNorm([self.num_feats, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            Conv1d1x1(hidden, hidden, self.num_feats, bias=True, cformat='channel-first'),
            # nn.LayerNorm([self.num_feats, hidden]),
            L2Norm(),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        self.semantic_aggr_layers = Transformer(hidden, att_drop, act)
        if self.residual:
            self.res_fc = nn.Linear(nfeat*2, hidden, bias=False)

        def add_nonlinear_layers(nfeats, dropout, bns=False):
            if bns:
                return [
                    nn.BatchNorm1d(hidden),
                    nn.PReLU(),
                    nn.Dropout(dropout)
                ]
            else:
                return [
                    nn.PReLU(),
                    nn.Dropout(dropout)
                ]
        
        # lr_output_layers = [
        #     [nn.Linear(self.num_scores, hidden, bias=not bns)]  + add_nonlinear_layers(hidden, dropout, bns)
        #     for _ in range(n_layers-1)]
        # self.lr_output = nn.Sequential(*(
        #     [ele for li in lr_output_layers for ele in li] + [
        #     nn.Linear(hidden, 1, bias=False),
        #     # nn.BatchNorm1d(nclass)
        #     ]))
        self.lr_output = nn.Linear(self.num_scores, 1, bias=not bns)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_ratio = dropout
        self.input_drop = nn.Dropout(input_drop)
        self.reset_parameters()
    
    def dropout_channel(self, feats):
        if self.training:
            num_samples = int(feats.shape[1] * self.dropout_ratio)# self.dropout
            selected_idx = torch.randperm(feats.shape[1], dtype=torch.int64, device=feats.device)[:num_samples]
            feats[:,selected_idx,:] = 0
        return feats

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for k, lin in self.embedings.items():
            nn.init.xavier_uniform_(lin.weight, gain=gain)
            nn.init.uniform_(lin.bias, -0.5, 0.5)
        
        for layer in self.feat_project_layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()

        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)

        # for layer in self.lr_output:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight, gain=gain)
        #         if layer.bias is not None:
        #             nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.lr_output.weight, gain=gain)
        if self.lr_output.bias is not None:
            nn.init.zeros_(self.lr_output.bias)

    def forward(self, src_feats_dict, dst_feats_dict, batch_src_idx, batch_dst_idx):
        
        B = num_node = batch_dst_idx.shape[0]
        x = []
        for k, v in src_feats_dict.items():
            v = v[batch_src_idx]
            if k in self.embedings:
                v = self.embedings[k](v)
            if v.size(0) != B:
                v = v.unsqueeze(1).expand(-1, 2, -1).flatten(end_dim=1)
            x.append(v)
        for k, v in dst_feats_dict.items():
            v = v[batch_dst_idx]
            if k in self.embedings:
                v = self.embedings[k](v)
            x.append(v)
        
        x = torch.stack(x, dim=1)
        x = self.feat_project_layers(x)
        src_x = x[:,:len(src_feats_dict),:]
        dst_x = x[:,len(src_feats_dict):,:]
        
        src_x_ = F.normalize(self.semantic_aggr_layers(src_x), p=2, dim=-1)
        dst_x_ = F.normalize(self.semantic_aggr_layers(dst_x), p=2, dim=-1)
        
        src_x = torch.cat([src_x, src_x_], dim=1)
        dst_x = torch.cat([dst_x, dst_x_], dim=1)
        
        link_scores = torch.einsum('bch,bnh->bcn', src_x, dst_x).flatten(start_dim=1)
        link_scores = torch.tanh(link_scores)
        # link_scores = torch.sigmoid(link_scores)
        # link_scores = self.prelu(link_scores)
        
        link_scores = self.dropout(link_scores)
        link_score = self.lr_output(link_scores)
        
        return link_score.squeeze()