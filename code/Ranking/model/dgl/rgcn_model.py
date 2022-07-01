"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RGCNBasisLayer as RGCNLayer, RGCNBasisLayer_1, RGCNBasisLayer_2

from .aggregators import SumAggregator, MLPAggregator, GRUAggregator, MLPAggregator_2, GRUAggregator_2


class RGCN(nn.Module):
    def __init__(self, params):
        super(RGCN, self).__init__()

        self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        # self.aggregator_type = params.gnn_agg_type
        self.has_attn = params.has_attn
        self.no_jk = params.no_jk

        self.device = params.device

        self.batchnorm_h = nn.BatchNorm1d(params.emb_dim)

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.num_rels, self.attn_rel_emb_dim, sparse=False)
            torch.nn.init.xavier_uniform_(self.attn_rel_emb.weight)
        else:
            self.attn_rel_emb = None

        # initialize aggregators for input and hidden layers
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        # initialize basis weights for input and hidden layers
        # self.input_basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.emb_dim))
        # self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim))

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def create_features(self):
        features = torch.arange(self.inp_dim).to(device=self.device)
        return features

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNLayer(self.inp_dim,
                         self.emb_dim,
                         # self.input_basis_weights,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         is_input_layer=True,
                         has_attn=self.has_attn,
                         no_jk=self.no_jk)

    def build_hidden_layer(self, idx):
        return RGCNLayer(self.emb_dim,
                         self.emb_dim,
                         # self.basis_weights,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         has_attn=self.has_attn,
                         no_jk=self.no_jk)

    # def forward(self, g):
    #     for layer in self.layers:
    #         layer(g, self.attn_rel_emb)
    #     return g.ndata.pop('h')

    def forward(self, g, norm):
        h_in = None  # for residual connection
        for layer in self.layers:
            if g.ndata.get('h') is not None:
                h_in = g.ndata['h']
            layer(g, self.attn_rel_emb, norm)
        h = g.ndata.pop('h')
        h = self.batchnorm_h(h)
        h = h_in + h  # for residual connection
        return h

class RGCN_1(nn.Module):
    def __init__(self, params):
        super(RGCN_1, self).__init__()

        self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.node_identifier_dropout
        self.edge_dropout = params.edge_dropout
        # self.aggregator_type = params.gnn_agg_type
        self.has_attn = params.has_attn
        self.gsn_type = params.gsn_type

        self.device = params.device
        self.num_substructure = params.num_substructure
        self.edge_num_substructure = params.edge_num_substructure

        self.batchnorm_h = nn.BatchNorm1d(params.emb_dim)
        self.no_jk = params.no_jk

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        # initialize aggregators for input and hidden layers
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        # initialize basis weights for input and hidden layers
        # self.input_basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.emb_dim))
        # self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim))

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def create_features(self):
        features = torch.arange(self.inp_dim).to(device=self.device)
        return features

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNBasisLayer_1(self.inp_dim,
                                self.emb_dim,
                                # self.input_basis_weights,
                                self.aggregator,
                                self.attn_rel_emb_dim,
                                self.aug_num_rels,
                                self.num_bases,
                                activation=F.relu,
                                dropout=self.dropout,
                                edge_dropout=self.edge_dropout,
                                is_input_layer=True,
                                has_attn=self.has_attn,
                                num_substructure=self.num_substructure,
                                edge_num_substructure=self.edge_num_substructure,
                                gsn_type=self.gsn_type,
                                no_jk=self.no_jk)

    def build_hidden_layer(self, idx):
        return RGCNBasisLayer_1(self.emb_dim,
                                self.emb_dim,
                                # self.basis_weights,
                                self.aggregator,
                                self.attn_rel_emb_dim,
                                self.aug_num_rels,
                                self.num_bases,
                                activation=F.relu,
                                dropout=self.dropout,
                                edge_dropout=self.edge_dropout,
                                has_attn=self.has_attn,
                                num_substructure=self.num_substructure,
                                edge_num_substructure=self.edge_num_substructure,
                                gsn_type=self.gsn_type,
                                no_jk=self.no_jk)

    def forward(self, g, norm):
        h_in = None  # for residual connection
        for layer in self.layers:
            if g.ndata.get('h') is not None:
                h_in = g.ndata['h']
            layer(g, self.attn_rel_emb, norm)
        h = g.ndata.pop('h')
        h = self.batchnorm_h(h)
        h = h_in + h  # for residual connection
        return h

class RGCN_2(nn.Module):
    def __init__(self, params):
        super(RGCN_2, self).__init__()

        self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.node_identifier_dropout
        self.edge_dropout = params.edge_dropout
        # self.aggregator_type = params.gnn_agg_type
        self.has_attn = params.has_attn

        self.device = params.device

        self.num_substructure = params.num_substructure
        self.edge_num_substructure = params.edge_num_substructure
        self.gsn_type = params.gsn_type

        self.batchnorm_h = nn.BatchNorm1d(params.emb_dim)

        self.no_jk = params.no_jk

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        # initialize aggregators for input and hidden layers
        if params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator_2(2 * self.emb_dim + self.num_substructure, self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator_2(self.emb_dim + self.num_substructure, self.emb_dim)
        else:
            raise ValueError('gnn_agg_type is error, gnn_agg_type:{}'.format(params.gnn_agg_type))

        # initialize basis weights for input and hidden layers
        # self.input_basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.emb_dim))
        # self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim))

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def create_features(self):
        features = torch.arange(self.inp_dim).to(device=self.device)
        return features

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNBasisLayer_2(self.inp_dim,
                                self.emb_dim,
                                # self.input_basis_weights,
                                self.aggregator,
                                self.attn_rel_emb_dim,
                                self.aug_num_rels,
                                self.num_bases,
                                activation=F.relu,
                                dropout=self.dropout,
                                edge_dropout=self.edge_dropout,
                                is_input_layer=True,
                                has_attn=self.has_attn,
                                num_substructure=self.num_substructure,
                                edge_num_substructure=self.edge_num_substructure,
                                gsn_type=self.gsn_type,
                                no_jk=self.no_jk)

    def build_hidden_layer(self, idx):
        return RGCNBasisLayer_2(self.emb_dim,
                                self.emb_dim,
                                # self.basis_weights,
                                self.aggregator,
                                self.attn_rel_emb_dim,
                                self.aug_num_rels,
                                self.num_bases,
                                activation=F.relu,
                                dropout=self.dropout,
                                edge_dropout=self.edge_dropout,
                                has_attn=self.has_attn,
                                num_substructure=self.num_substructure,
                                edge_num_substructure=self.edge_num_substructure,
                                gsn_type=self.gsn_type,
                                no_jk=self.no_jk)

    def forward(self, g, norm):
        h_in = None  # for residual connection
        for layer in self.layers:
            if g.ndata.get('h') is not None:
                h_in = g.ndata['h']
            layer(g, self.attn_rel_emb, norm)
        h = g.ndata.pop('h')
        h = self.batchnorm_h(h)
        h = h_in + h  # for residual connection
        return h
