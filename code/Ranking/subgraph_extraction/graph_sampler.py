import struct
import logging
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.sparse as ssp
import torch
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius, ssp_multigraph_to_dgl
import copy
import dgl
from scipy.sparse import csc_matrix


def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges


    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    return pos_edges


def links2subgraphs(A, graphs, params, max_label_value=None, identifiers=None):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []
    max_num_dim_edge_identifiers_list = []
    max_nodes_identifiers_list = []
    max_edge_identifiers_list = []

    ssp_graph = copy.copy(A)
    if params.add_traspose_rels:
        ssp_graph_t = [adj.T for adj in A]
        ssp_graph += ssp_graph_t

    # the effective number of relations after adding symmetric adjacency matrices and/or self connections
    graph = ssp_multigraph_to_dgl(ssp_graph)

    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 200.0
    links_length = 0
    for split_name, split in graphs.items():
        links_length += len(split['pos']) * 2
    map_size = links_length * BYTES_PER_DATUM

    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env, graph):

        mp.set_start_method('spawn', force=True)

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value,graph)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])
                enc_ratios.append(datum['enc_ratio'])
                num_pruned_nodes.append(datum['num_pruned_nodes'])

                num_dim_edge_identifiers = datum['num_dim_edge_identifiers']
                if num_dim_edge_identifiers is not None:
                    max_num_dim_edge_identifiers_list.append(num_dim_edge_identifiers)

                max_nodes_identifiers = datum['max_nodes_identifiers']
                if max_nodes_identifiers is not None:
                    max_nodes_identifiers_list.append(max_nodes_identifiers)
                max_edge_identifiers = datum['max_edge_identifiers']
                if max_edge_identifiers is not None:
                    max_edge_identifiers_list.append(max_edge_identifiers)

                del datum['max_nodes_identifiers']
                del datum['max_edge_identifiers']
                del datum['num_dim_edge_identifiers']

                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))
            p.close()
            p.join()

        mp.set_start_method('fork', force=True)

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env, graph)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']
    if params.identifiers_from_all_graph is not True:
        max_edge_identifiers = None if len(max_edge_identifiers_list) == 0 else list(
            np.max(max_edge_identifiers_list, axis=0))
        max_nodes_identifiers = None if len(max_nodes_identifiers_list) == 0 else list(
            np.max(max_nodes_identifiers_list, axis=0))

        max_num_dim_edge_identifiers = 0 if len(max_num_dim_edge_identifiers_list) == 0 else np.max(
            max_num_dim_edge_identifiers_list)
    else:
        max_edge_identifiers = torch.max(torch.FloatTensor(list(identifiers['edge_identifier'].values())), dim=0)[0].tolist()
        max_nodes_identifiers = torch.max(identifiers['node_identifier'], dim=0)[0].tolist()
        max_num_dim_edge_identifiers = len(max_edge_identifiers)

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(),
                (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(),
                (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))

        # save edge identifiers
        if identifiers['edge_identifier'] is not None:
            datum = {'edge_identifiers': identifiers['edge_identifier']}
            txn.put('edge_identifiers'.encode(), serialize(datum))

        # save node identifiers
        if identifiers['node_identifier'] is not None:
            datum = {'node_identifier': identifiers['node_identifier'].tolist()}
            txn.put('node_identifier'.encode(), serialize(datum))

        datum = {'max_edge_identifiers': max_edge_identifiers}
        txn.put('max_edge_identifiers'.encode(), serialize(datum))

        datum = {'max_nodes_identifiers': max_nodes_identifiers}
        txn.put('max_nodes_identifiers'.encode(), serialize(datum))

        # max_num_dim_edge_identifiers
        bit_max_num_dim_edge_identifiers = int.bit_length(int(max_num_dim_edge_identifiers))
        txn.put('max_num_dim_edge_identifiers'.encode(),
                (int(max_num_dim_edge_identifiers)).to_bytes(bit_max_num_dim_edge_identifiers, byteorder='little'))


def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A, params, max_label_value, graph):
    global A_, params_, max_label_value_, graph_
    A_, params_, max_label_value_, graph_ = A, params, max_label_value, graph


def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A_,
                                                                                               params_.hop,
                                                                                               params_.enclosing_sub_graph,
                                                                                               params_.max_nodes_per_hop)

    nodes_identifiers = []
    edge_identifiers = []
    num_dim_edge_identifiers = 0
    max_nodes_identifiers = None
    max_edge_identifiers = None
    subgraph_identifiers = None
    if params_.identifiers_from_all_graph is not True:
        adj_list = []
        subgraph = dgl.DGLGraph(graph_.subgraph(nodes))
        adj = subgraph.adjacency_matrix_scipy(transpose=True, fmt='coo', return_edge_ids=False)
        row = np.array(adj.tocoo().row.tolist())
        col = np.array(adj.tocoo().col.tolist())
        data = np.array(adj.tocoo().data.tolist())
        matrix = csc_matrix((data, (row, col)), shape=(len(nodes), len(nodes)))
        adj_list.append(matrix)
        from .datasets import get_identifier
        subgraph_identifiers = get_identifier(adj_list, len(nodes), params_)

        adj = []
        adj.append(row)
        adj.append(col)
        adj = torch.IntTensor(adj).T
        for i in range(len(adj)):
            key = (adj[i].tolist()[0], adj[i].tolist()[1])
            edge_identifiers.append(subgraph_identifiers['edge_identifier'][key])


    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    if subgraph_identifiers is not None:
        nodes_identifiers = subgraph_identifiers['node_identifier'].tolist()
        edge_identifier_list = list(subgraph_identifiers['edge_identifier'].values())
        num_dim_edge_identifiers = 0 if len(edge_identifier_list) == 0 else len(edge_identifier_list[0])
        max_nodes_identifiers = np.max(nodes_identifiers, axis=0)
        max_edge_identifiers = None if len(edge_identifier_list) == 0 else np.max(edge_identifier_list, axis=0)

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels,
             'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes,
             'nodes_identifiers_from_subgraph': nodes_identifiers,
             'edge_identifiers_from_subgraph': edge_identifiers,
             'num_dim_edge_identifiers': num_dim_edge_identifiers,
             'max_nodes_identifiers': max_nodes_identifiers,
             'max_edge_identifiers': max_edge_identifiers}

    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

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

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes


def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes
