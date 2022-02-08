# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 20:29:08 2021

@author: user
"""
import torch
import torch.nn as nn, torch.nn.functional as F
import math 
from torch_scatter import scatter
from torch_geometric.utils import softmax
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing
from typing import Optional, List, Dict
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn import GATConv


class ContrastConv(MessagePassing):
    def __init__(self, in_channels):
        super(ContrastConv, self).__init__(aggr='mean')
        self.W = nn.Linear(2*in_channels, in_channels, bias=False)
    

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        tmp = self.W(torch.cat([x_i, x_j], dim=-1)) + x_i

        return tmp
    
    
class Att_Agg(MessagePassing):
    def __init__(self, in_channels):
        super(Att_Agg, self).__init__(aggr='mean') 
        self.W = nn.Linear(2*in_channels, in_channels, bias=False)
        self.act = nn.Sigmoid()
    

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        cos = torch.cosine_similarity(x_i, x_j, dim = 0)
        t = (1-cos)*x_j
        
        return t
    
class UniSAGEConv2(nn.Module):

    def __init__(self, in_channels, out_channels,  heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        # TODO: bias?
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.contrast_conv = ContrastConv(heads * out_channels)
        self.att_agg = Att_Agg(heads * out_channels)  

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, H, E,case):
        '''
        Parameters
        ----------
        X : TYPE
            Input node features.
        H : TYPE
            Hypergraph, [E,N].
        E : TYPE
            Edges between hypergraphs.
        '''


        X = self.W(X)

        Xe = torch.mm(H, X)/(torch.sum(H, dim=1, keepdim=True)+0.01) # [E, C]
        
        Xh = self.contrast_conv(Xe, E)
        Xc = torch.cat((X,Xh), dim = 0)
        Xv = self.att_agg(Xc,case.edge_node_he)
        Xn = torch.index_select(Xv,0,case.index)
        X = X + Xn
       
        return X
    
class TNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, nhead):
        """TNN

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        
        
        Conv = UniSAGEConv2
        self.dropout_rate = 0.0
        self.conv_out = Conv(nhid * nhead, nclass, heads=1, dropout=self.dropout_rate)
        self.convs = nn.ModuleList(
            [ Conv(nfeat, nhid, heads=nhead, dropout=self.dropout_rate)] +
            [Conv(nhid * nhead, nhid, heads=nhead, dropout=self.dropout_rate) for _ in range(nlayer-2)]
        )
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu']
        
    def forward(self, X, H, E,case):
        for conv in self.convs:
            X = conv(X, H, E,case)
            X = self.act(X)

        X = self.conv_out(X, H, E,case)  
        
        return X
        