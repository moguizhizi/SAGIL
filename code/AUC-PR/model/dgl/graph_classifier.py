from .rgcn_model import RGCN, RGCN_1, RGCN_2
from dgl import mean_nodes
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.relation_list = list(self.relation2id.values())
        self.no_jk = self.params.no_jk
        self.is_big_dataset = False
        self.link_mode = 6
        if self.params.dataset in ['fb237_v4', 'nell_v4', 'fb_new', 'WN_new']:
            self.is_big_dataset = True

        if params.version == 0:
            self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        elif params.version == 1:
            self.gnn = RGCN_1(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        elif params.version == 2:
            self.gnn = RGCN_2(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        else:
            raise NotImplementedError("gcn version is error, version:{}.".format(params.version))

        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.rel_depen = nn.ModuleList(
            [nn.Embedding(self.params.num_rels, self.params.num_rels) for _ in range(self.link_mode)])

        torch.nn.init.normal_(self.rel_emb.weight)
        for i in range(self.link_mode):
            torch.nn.init.normal_(self.rel_depen[i].weight)

        if params.six_mode:
            self.fc_reld = nn.ModuleList([nn.Linear(self.params.rel_emb_dim, self.params.rel_emb_dim)
                                          for _ in range(6)
                                          ])
        else:
            self.fc_reld = nn.ModuleList([nn.Linear(self.params.rel_emb_dim, self.params.rel_emb_dim)
                                          for _ in range(self.link_mode)
                                          ])

        self.conc = nn.Linear(self.params.rel_emb_dim * 2, self.params.rel_emb_dim)
        nn.init.kaiming_normal_(self.conc.weight, nonlinearity='relu')
        num_final_gcn = self.params.num_gcn_layers
        if self.no_jk:
            num_final_gcn = 1

        if self.params.add_ht_emb:
            if self.params.ablation == 0:
                self.fc_layer = nn.Linear(3 * num_final_gcn * self.params.emb_dim + self.params.rel_emb_dim, 1)
            elif self.params.ablation == 1:  # no-subg
                self.fc_layer = nn.Linear(2 * num_final_gcn * self.params.emb_dim + self.params.rel_emb_dim, 1)
            elif self.params.ablation == 2:  # no-ent
                self.fc_layer = nn.Linear(1 * num_final_gcn * self.params.emb_dim + self.params.rel_emb_dim, 1)
            elif self.params.ablation == 3:  # only-rel
                self.fc_layer = nn.Linear(self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(num_final_gcn * self.params.emb_dim + self.params.rel_emb_dim, 1)

    def convert2onehot(self, max_identifiers, identifiers_from_subgraph):
        identifiers_from_subgraph = identifiers_from_subgraph.tolist()
        max_identifiers = (torch.IntTensor(max_identifiers) + 1).tolist()
        one_hot_dim = np.sum(max_identifiers)
        one_hot_dim = int(one_hot_dim)
        one_hot_vector = np.zeros((len(identifiers_from_subgraph), one_hot_dim))
        identifiers_from_subgraph = np.array(identifiers_from_subgraph, dtype=int)
        for i in range(len(max_identifiers)):
            try:
                one_hot_vector[np.arange((len(identifiers_from_subgraph))), identifiers_from_subgraph[:, i]] = 1
            except Exception:
                raise Exception

        return torch.FloatTensor(list(one_hot_vector)).to(device=self.params.device)

    def init_identifiers_embedding(self, g):
        node_identifiers = g.ndata['identifier']
        edge_identifiers = g.edata['identifier']

        if self.params.one_hot_enable is True:
            node_identifiers = self.convert2onehot(self.params.max_nodes_identifiers, node_identifiers)
            edge_identifiers = self.convert2onehot(self.params.max_edge_identifiers, edge_identifiers)

        if self.params.activate_func_type == 'relu':
            node_identifiers = F.relu(node_identifiers)
        elif self.params.activate_func_type == 'sigmoid':
            node_identifiers = F.sigmoid(node_identifiers)
        elif self.params.activate_func_type == 'tanh':
            node_identifiers = F.tanh(node_identifiers)
        elif self.params.activate_func_type == 'mlp':
            node_identifiers = self.node_identifier_fc_layer(
                torch.FloatTensor(node_identifiers.tolist()).to(device=self.params.device))
        elif self.params.activate_func_type == 'normalize':
            node_identifiers = F.normalize(
                torch.FloatTensor(node_identifiers.tolist()).to(device=self.params.device), dim=1)
        elif self.params.activate_func_type == 'sin':
            node_identifiers = torch.sin(torch.FloatTensor(node_identifiers.tolist()).to(device=self.params.device))
        else:
            raise ValueError(
                "activate_func_type is error, activate_func_type is:{}".format(self.params.activate_func_type))

        if self.params.edge_identifier_activate_func_type == 'relu':
            edge_identifiers = F.relu(edge_identifiers)
        elif self.params.edge_identifier_activate_func_type == 'sigmoid':
            edge_identifiers = F.sigmoid(edge_identifiers)
        elif self.params.edge_identifier_activate_func_type == 'tanh':
            edge_identifiers = F.tanh(edge_identifiers)
        elif self.params.edge_identifier_activate_func_type == 'mlp':
            edge_identifiers = self.edge_identifier_fc_layer(
                torch.FloatTensor(edge_identifiers.tolist()).to(device=self.params.device))
        elif self.params.edge_identifier_activate_func_type == 'normalize':
            edge_identifiers = F.normalize(
                torch.FloatTensor(edge_identifiers.tolist()).to(device=self.params.device), dim=1)
        elif self.params.edge_identifier_activate_func_type == 'sin':
            edge_identifiers = torch.sin(torch.FloatTensor(edge_identifiers.tolist()).to(device=self.params.device))
        else:
            raise ValueError(
                "edge_identifiers is error, activate_func_type is:{}".format(self.params.edge_identifiers))

        g.ndata['identifier'] = node_identifiers
        g.edata['identifier'] = edge_identifiers

        if self.params.init_add_identifiers is True:
            node_feat = torch.cat((g.ndata['feat'], node_identifiers), dim=1)
            g.ndata['feat'] = node_feat

        return g

    def forward(self, data):
        g, rel_labels = data
        g = self.init_identifiers_embedding(g)

        local_g = g.local_var()
        in_deg = local_g.in_degrees(range(local_g.number_of_nodes())).float().numpy()
        in_deg[in_deg == 0] = 1
        node_norm = 1.0 / in_deg
        local_g.ndata['norm'] = node_norm
        local_g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
        norm = local_g.edata['norm'].to(device=self.params.device)

        g.ndata['h'] = self.gnn(g, norm)

        g_out = mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        u_node, v = head_ids, tail_ids
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        u_in_edge = g.in_edges(u_node, 'all')
        u_out_edge = g.out_edges(u_node, 'all')
        v_in_edge = g.in_edges(v, 'all')
        v_out_edge = g.out_edges(v, 'all')

        in_edge_out = torch.sparse_coo_tensor(torch.cat((u_in_edge[1].unsqueeze(0), u_in_edge[2].unsqueeze(0)), 0),
                                              torch.ones(len(u_in_edge[2])), size=torch.Size((num_nodes, num_edges)))
        out_edge_out = torch.sparse_coo_tensor(torch.cat((u_out_edge[0].unsqueeze(0), u_out_edge[2].unsqueeze(0)), 0),
                                               torch.ones(len(u_out_edge[2])), size=torch.Size((num_nodes, num_edges)))
        in_edge_in = torch.sparse_coo_tensor(torch.cat((v_in_edge[1].unsqueeze(0), v_in_edge[2].unsqueeze(0)), 0),
                                             torch.ones(len(v_in_edge[2])), size=torch.Size((num_nodes, num_edges)))
        out_edge_in = torch.sparse_coo_tensor(torch.cat((v_out_edge[0].unsqueeze(0), v_out_edge[2].unsqueeze(0)), 0),
                                              torch.ones(len(v_out_edge[2])), size=torch.Size((num_nodes, num_edges)))

        if self.is_big_dataset:  # smaller memory
            in_edge_out = self.sparse_index_select(in_edge_out, u_node).to(device=self.params.device)
            out_edge_out = self.sparse_index_select(out_edge_out, u_node).to(device=self.params.device)
            in_edge_in = self.sparse_index_select(in_edge_in, v).to(device=self.params.device)
            out_edge_in = self.sparse_index_select(out_edge_in, v).to(device=self.params.device)
        else:  # faster calculation
            in_edge_out = in_edge_out.to(device=self.params.device).to_dense()[u_node].to_sparse()
            out_edge_out = out_edge_out.to(device=self.params.device).to_dense()[u_node].to_sparse()
            in_edge_in = in_edge_in.to(device=self.params.device).to_dense()[v].to_sparse()
            out_edge_in = out_edge_in.to(device=self.params.device).to_dense()[v].to_sparse()

        ## six patterns
        edge_mode_5 = out_edge_out.mul(in_edge_in)
        edge_mode_6 = in_edge_out.mul(out_edge_in)
        out_edge_out = out_edge_out.sub(edge_mode_5)
        in_edge_in = in_edge_in.sub(edge_mode_5)
        in_edge_out = in_edge_out.sub(edge_mode_6)
        out_edge_in = out_edge_in.sub(edge_mode_6)
        # step by step
        # edge_connect_l = [in_edge_out, out_edge_out, in_edge_in, out_edge_in]
        edge_connect_l = [in_edge_out, out_edge_out, in_edge_in, out_edge_in, edge_mode_5, edge_mode_6]
        norm_sparse = []
        for i in range(self.link_mode):
            norm_sparse.append(
                torch.sparse.mm(edge_connect_l[i], torch.ones(num_edges, 1).to(device=self.params.device)) + 1e-30)

        if self.params.epoch <= (self.params.num_epochs + 1) // 2:
            rel_neighbor_embd = sum([torch.sparse.mm(edge_connect_l[i],
                                                     self.fc_reld[i](self.rel_emb(g.edata['type']))) * 1. / norm_sparse[
                                         i] for i in range(self.link_mode)]) * 1. / self.link_mode
        else:
            if self.rel_emb.weight.requires_grad:
                self.rel_emb.weight.requires_grad = False

            rel_neighbor_embd = sum([torch.sparse.mm(
                self.sparse_dense_mul(edge_connect_l[i],
                                      torch.softmax(self.rel_depen[i].weight, dim=1)[rel_labels][:, g.edata['type']]),
                self.fc_reld[i](self.rel_emb(g.edata['type']))) * 1. / norm_sparse[i] for i in
                                     range(self.link_mode)]) * 1. / self.link_mode

        rel_final_emb = self.conc(torch.cat([rel_neighbor_embd, self.rel_emb(rel_labels)], dim=-1))
        rel_final_emb = F.relu(rel_final_emb)
        rel_final_emb = F.normalize(rel_final_emb, p=2, dim=-1)

        if self.no_jk:
            if self.params.ablation == 0:
                g_rep = torch.cat([g_out.view(-1, self.params.emb_dim),
                                   head_embs.view(-1, self.params.emb_dim),
                                   tail_embs.view(-1, self.params.emb_dim),
                                   rel_final_emb], dim=1)
            elif self.params.ablation == 1:  # no-subg
                g_rep = torch.cat([head_embs.view(-1, self.params.emb_dim),
                                   tail_embs.view(-1, self.params.emb_dim),
                                   rel_final_emb], dim=1)
            elif self.params.ablation == 2:  # no-ent
                g_rep = torch.cat([g_out.view(-1, self.params.emb_dim),
                                   rel_final_emb], dim=1)
            elif self.params.ablation == 3:  # only-rel
                g_rep = torch.cat([rel_final_emb], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               rel_final_emb], dim=1)

        output = self.fc_layer(g_rep)
        # output_neg = self.fc_layer(g_rep_neg)
        output_neg = None
        return output, output_neg

    @staticmethod
    def sparse_dense_mul(s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    @staticmethod
    def sparse_index_select(s, idx):
        s = s.to(device="cpu")
        idx = idx.to(device="cpu")

        indices_s = s._indices()
        indice_new_1 = torch.tensor([])
        indice_new_2 = torch.tensor([])
        num_i = 0.0
        for itm in idx:
            mask = (indices_s[0] == itm)
            indice_tmp_1 = torch.ones(sum(mask)) * num_i
            indice_tmp_2 = indices_s[1][mask].float()
            indice_new_1 = torch.cat((indice_new_1, indice_tmp_1), dim=0)
            indice_new_2 = torch.cat((indice_new_2, indice_tmp_2), dim=0)
            num_i = num_i + 1.0
        indices_new = torch.cat((indice_new_1.unsqueeze(0), indice_new_2.unsqueeze(0)), dim=0).long()

        return torch.sparse.FloatTensor(indices_new, torch.ones(indices_new.shape[1]),
                                        torch.Size((len(idx), s.shape[1])))
