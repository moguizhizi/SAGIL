import os
import random
import argparse
import logging
import json
import time

import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
import torch
import numpy as np
import dgl

from subgraph_extraction.datasets import get_identifier
from utils import parsing_utils as parse

from utils.initialization_utils import set_params_from_model
from scipy.sparse import csc_matrix


def process_files(files, saved_relation2id, add_traspose_rels):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in saved_relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], saved_relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, identifiers, num_dim_edge_identifiers):
    global model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, identifiers_, num_dim_edge_identifiers_
    model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, identifiers_, num_dim_edge_identifiers_ = model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, identifiers,num_dim_edge_identifiers


def get_neg_samples_replacing_relation(test_links, adj_list, relations):
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        neg_triplet = [[], 0]
        neg_triplet[0].append([head, tail, rel])
        for neg_rel in relations:
            if neg_rel != rel and adj_list[neg_rel][head, tail] == 0: # for the negative samples
                neg_triplet[0].append([head, tail, neg_rel])

        neg_triplet[0] = np.array(neg_triplet[0])
        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_test_triples(test_links, adj_list, relations):
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    test_triples = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        test_triples_and_f = [[head, tail, rel], [0]*len(relations)]
        for neg_rel in relations:
            if neg_rel != rel and adj_list[neg_rel][head, tail] == 1:  # for the negative samples
                test_triples_and_f[1][neg_rel] = -1
        test_triples.append(test_triples_and_f)

    return test_triples


def get_rank(links_all):
    test_links = [np.array(links_all[0])]
    test_filter = np.array(links_all[1])
    data = get_subgraphs(test_links, adj_list_, dgl_adj_list_, model_.gnn.max_label_value, id2entity_, node_features_, kge_entity2id_)
    score_pos, score_neg = model_(data)
    scores = torch.cat((score_pos, score_neg.view(-1, 1)), dim=0)
    scores = torch.softmax(scores, dim=0).squeeze(1)
    scores = scores.detach().numpy()
    scores[1:] += test_filter[np.array(model_.neg_list)]
    rank = np.argwhere(np.argsort(scores)[::-1] == 0) + 1
    return scores, rank


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    Modified from dgl.contrib.data.knowledge_graph to node accomodate sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    # could pack these two into a function
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    return pruned_subgraph_nodes, pruned_labels


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def node_label_new(subgraph, max_distance=1):
    # an implementation of the proposed double-radius node labeling
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    # print(len(enclosing_subgraph_nodes))
    return labels, enclosing_subgraph_nodes


def ssp_multigraph_to_dgl(graph, edge_identifiers=None, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            if edge_identifiers is None:
                nx_triplets.append((src, dst, {'type': rel}))
            else:
                nx_triplets.append((src, dst, {'type': rel, 'identifier': edge_identifiers[(src, dst)]}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.DGLGraph(multigraph=True)
    if edge_identifiers is None:
        g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    else:
        g_dgl.from_networkx(g_nx, edge_attrs=['type', 'identifier'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def prepare_features(subgraph, n_labels, max_n_label,n_feats=None, identifiers=None):
    # One hot encode the node label feature and concat to n_featsure
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
    subgraph.ndata['identifier'] = identifiers

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)

    return subgraph

def identifiers_from_sub_graph(params, nodes, subgraph):
    subgraph_adj_list = []
    adj = subgraph.adjacency_matrix_scipy(transpose=True, fmt='coo', return_edge_ids=False)
    row = np.array(adj.tocoo().row.tolist())
    col = np.array(adj.tocoo().col.tolist())
    data = np.array(adj.tocoo().data.tolist())
    matrix = csc_matrix((data, (row, col)), shape=(len(nodes), len(nodes)))
    subgraph_adj_list.append(matrix)
    from subgraph_extraction.datasets import get_identifier
    subgraph_identifiers = get_identifier(subgraph_adj_list, len(nodes), params_)
    adj = []
    adj.append(row)
    adj.append(col)
    adj = torch.IntTensor(adj).T
    edge_identifiers = []
    for i in range(len(adj)):
        key = (adj[i].tolist()[0], adj[i].tolist()[1])
        edge_identifiers.append(subgraph_identifiers['edge_identifier'][key])
    pad_identifier = torch.zeros(num_dim_edge_identifiers_, dtype=int).unsqueeze(0)
    node_identifiers = subgraph_identifiers['node_identifier'].tolist()

    edge_identifiers, node_identifiers = clamp_process(params, edge_identifiers, node_identifiers)

    edge_identifiers, node_identifiers = cut_process(params, edge_identifiers, node_identifiers)

    return edge_identifiers, node_identifiers, pad_identifier

def clamp_process(params, edge_identifiers, nodes_identifiers):
    if params.clamp_enable is True:
        if len(nodes_identifiers) != 0:
            nodes_identifiers = torch.clamp(
                torch.FloatTensor(nodes_identifiers), min=params.node_identifier_min,
                max=params.node_identifier_max).tolist()
        if len(edge_identifiers) != 0:
            edge_identifiers = torch.clamp(
                torch.FloatTensor(edge_identifiers), min=params.edge_identifier_min,
                max=params.edge_identifier_max).tolist()

    return edge_identifiers, nodes_identifiers

def cut_process(params, edge_identifiers, nodes_identifiers):
    if params.cut_enable is True:
        temp_node_identifiers = torch.FloatTensor(nodes_identifiers)
        zeros = torch.zeros_like(temp_node_identifiers)
        filter_index = (temp_node_identifiers >= params.node_identifier_min) * (
                temp_node_identifiers <= params.node_identifier_max)
        nodes_identifiers = torch.where(filter_index, temp_node_identifiers, zeros).tolist()

        temp_edge_identifiers = torch.FloatTensor(edge_identifiers)
        zeros = torch.zeros_like(temp_edge_identifiers)
        filter_index = (temp_edge_identifiers >= params.edge_identifier_min) * (
                temp_edge_identifiers <= params.edge_identifier_max)
        edge_identifiers = torch.where(filter_index, temp_edge_identifiers, zeros).tolist()

    return edge_identifiers, nodes_identifiers


def get_subgraphs(all_links, adj_list, dgl_adj_list, max_node_label_value, id2entity, node_features=None, kge_entity2id=None):

    subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        nodes, node_labels = subgraph_extraction_labeling((head, tail), rel, adj_list, h=params_.hop, enclosing_sub_graph=params_.enclosing_sub_graph, max_node_label_value=max_node_label_value)

        subgraph = dgl.DGLGraph(dgl_adj_list.subgraph(nodes))
        subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).parent_eid]
        subgraph.ndata['entid'] = dgl_adj_list.subgraph(nodes).parent_nid
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        if params_.identifiers_from_all_graph is not True:
            edge_identifiers, node_identifiers, pad_identifier = identifiers_from_sub_graph(params_, nodes, subgraph)
            subgraph.edata['identifier'] = torch.IntTensor(edge_identifiers)
            node_identifiers = torch.IntTensor(node_identifiers)
            length = len(edge_identifiers)
        else:
            subgraph.edata['identifier'] = dgl_adj_list.edata['identifier'][dgl_adj_list.subgraph(nodes).parent_eid]
            pad_identifier = torch.zeros(len(dgl_adj_list.edata['identifier'][0]), dtype=int).unsqueeze(0)
            length = len(subgraph.edata['identifier'])
            node_identifiers = identifiers_['node_identifier'][nodes]

        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)

        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)
            if length == 0:
                subgraph.edata['identifier'] = pad_identifier
            else:
                subgraph.edata['identifier'][-1] = pad_identifier

        kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes] if kge_entity2id else None
        n_feats = node_features[kge_nodes] if node_features is not None else None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats,identifiers=node_identifiers)

        subgraphs.append(subgraph)
        r_labels.append(rel)

    batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)

    return (batched_graph, r_labels)


def save_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join('./data', params.dataset, 'ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')

def clamp_or_cut_for_all_graph_process(identifiers, params):
    node_identifier = identifiers['node_identifier'].tolist()
    edge_identifier = identifiers['edge_identifier']
    if params.clamp_enable is True or params.cut_enable is True:
        if params.cut_enable is not True:
            node_identifier = torch.clamp(torch.FloatTensor(node_identifier),
                                          min=params.node_identifier_min,
                                          max=params.node_identifier_max).tolist()

        else:
            temp_node_identifier = torch.FloatTensor(node_identifier)
            zeros = torch.zeros_like(temp_node_identifier)
            filter_index = (temp_node_identifier >= params.node_identifier_min) * (
                    temp_node_identifier <= params.node_identifier_max)
            node_identifier = torch.where(filter_index, temp_node_identifier, zeros).tolist()

        identifiers['node_identifier'] = torch.FloatTensor(node_identifier)

        if params.cut_enable is not True:
            for key, value in edge_identifier.edge_identifiers.items():
                value = torch.clamp(torch.FloatTensor(value),
                                    min=params.edge_identifier_min,
                                    max=params.edge_identifier_max).tolist()
                edge_identifier[key] = value
        else:
            for key, value in edge_identifier.items():
                temp_edge_identifiers = list(value)
                temp_edge_identifiers = torch.FloatTensor(temp_edge_identifiers)
                zeros = torch.zeros_like(temp_edge_identifiers)
                filter_index = (temp_edge_identifiers >= params.edge_identifier_min) * (
                        temp_edge_identifiers <= params.edge_identifier_max)
                temp_edge_identifiers = torch.where(filter_index, temp_edge_identifiers, zeros).tolist()
                edge_identifier[key] = temp_edge_identifiers

        identifiers['edge_identifier'] = edge_identifier

def main(params):
    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, model.relation2id, params.add_traspose_rels)

    test_path = os.path.join(os.path.dirname(params.file_paths['graph']), 'id2ent.json')
    with open(test_path, 'w') as f:
        json.dump(id2entity, f)
    test_path = os.path.join(os.path.dirname(params.file_paths['graph']), 'id2rel.json')
    with open(test_path, 'w') as f:
        json.dump(id2relation, f)

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if params.add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    identifiers = get_identifier(adj_list_aug, len(entity2id), params)
    num_dim_edge_identifiers = 0
    if params.identifiers_from_all_graph is not True:
        num_dim_edge_identifiers = len(list(identifiers['edge_identifier'].values())[0])
        identifiers['node_identifier'] = None
        identifiers['edge_identifier'] = None
    else:
        clamp_or_cut_for_all_graph_process(identifiers, params)

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug, identifiers['edge_identifier'])

    node_features, kge_entity2id = (None, None)

    relations = list(relation2id.values())
    test_triplets_and_filter = get_test_triples(triplets['links'], adj_list, relations)

    ranks = []
    all_head_scores = []

    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=16, initializer=intialize_worker, initargs=(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id,
                         identifiers, num_dim_edge_identifiers)) as p:
        for head_scores, head_rank in tqdm(p.imap(get_rank, test_triplets_and_filter), total=len(test_triplets_and_filter)):
            ranks.append(head_rank)
            all_head_scores += head_scores.tolist()

    mp.set_start_method('fork', force=True)

    isHit1List = [x for x in ranks if x <= 1]
    isHit3List = [x for x in ranks if x <= 3]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) * 1.0 / len(ranks)
    hits_3 = len(isHit3List) * 1.0 / len(ranks)
    hits_10 = len(isHit10List) * 1.0 / len(ranks)

    mrr = np.mean(1.0 / np.array(ranks)).item()

    logger.info('test metrics. MRR: %.4f, H@1: %.4f, H@3: %.4f, H@10: %.4f' % (mrr, hits_1, hits_3, hits_10))

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="FB237_v2",
                        help="Path to dataset")
    parser.add_argument("--mode", "-m", type=str, default="sample", choices=["sample", "all", "ruleN"],
                        help="Negative sampling mode")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=2,
                        help="How many hops to go while eextracting subgraphs?")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='Whether to append adj matrix list with symmetric relations?')
    parser.add_argument('--final_model', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--has_attn', '-attn', type=parse.str2bool, default=True,
                        help='whether to have attn in model or not')

    #train
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')

    # GSN
    parser.add_argument('--id_type', type=str, default='complete_graph_chosen_k')
    parser.add_argument('--induced', type=parse.str2bool, default=False)
    parser.add_argument('--edge_automorphism', type=str, default='induced')
    parser.add_argument('--k', type=parse.str2list2int, default=[3])
    parser.add_argument('--id_scope', type=str, default='local')
    parser.add_argument('--custom_edge_list', type=parse.str2ListOfListsOfLists2int, default=None)
    parser.add_argument('--directed', type=parse.str2bool, default=True)
    parser.add_argument('--directed_orbits', type=parse.str2bool, default=True)
    parser.add_argument('--init_add_identifiers', type=parse.str2bool, default=False)
    parser.add_argument("--version", type=int, default=0, help="GCN version[0,1,2]")
    parser.add_argument("--num_substructure", type=int, default=0)
    parser.add_argument("--edge_num_substructure", type=int, default=0)
    parser.add_argument("--gsn_type", type=int, default=0, help="gsn type[0,1,2]")
    parser.add_argument('--hit_eval', type=parse.str2bool, default=False)
    parser.add_argument('--identifiers_from_all_graph', type=parse.str2bool, default=False,
                        help='whether to get identifiers from all graph or not')
    parser.add_argument("--init_identifiers_dim", type=int, default=32)
    parser.add_argument('--activate_func_type', type=str,
                        choices=['relu', 'sigmoid', 'tanh', 'mlp', 'normalize', 'sin'],
                        default='relu')
    parser.add_argument('--edge_identifier_activate_func_type', type=str,
                        choices=['relu', 'sigmoid', 'tanh', 'mlp', 'normalize', 'sin'],
                        default='relu')
    parser.add_argument('--one_hot_enable', type=parse.str2bool, default=True,
                        help='whether to enable one hot or not')
    parser.add_argument("--node_identifier_dropout", type=float, default=0.5,
                        help="Dropout rate in node_identifier of the subgraphs")
    parser.add_argument("--edge_identifier_dropout", type=float, default=0.5,
                        help="Dropout rate in edge_identifier of the subgraphs")
    parser.add_argument('--clamp_enable', type=parse.str2bool, default=False)
    parser.add_argument('--cut_enable', type=parse.str2bool, default=True)
    parser.add_argument("--node_identifier_min", type=int, default=0)
    parser.add_argument("--node_identifier_max", type=int, default=16)
    parser.add_argument("--edge_identifier_min", type=int, default=0)
    parser.add_argument("--edge_identifier_max", type=int, default=32)

    params = parser.parse_args()

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.file_paths = {
        'graph': os.path.join('../../data', params.dataset, 'train.txt'),
        'links': os.path.join('../../data', params.dataset, 'test.txt')
    }

    # params.final_model = True
    if params.final_model:
        params.model_path = os.path.join('experiments', params.experiment_name, 'graph_classifier_chk.pth')
    else:
        params.model_path = os.path.join('experiments', params.experiment_name, 'best_graph_classifier.pth')

    file_handler = logging.FileHandler(os.path.join('experiments', params.experiment_name, f'log_rank_test_{time.time()}.txt'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    model = torch.load(params.model_path, map_location=params.device)
    model.params.device = params.device
    model.params.num_neg_samples_per_link = model.params.num_rels - 1
    set_params_from_model(params, model)
    

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    main(params)