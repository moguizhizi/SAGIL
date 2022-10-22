import torch
from torch_geometric.utils import remove_self_loops
from utils.graph_utils import incidence_matrix


def subgraph_counts2ids(count_fn, adj_list, subgraph_dicts, subgraph_params):
    # 获取edge_index
    edge_index = incidence_matrix(adj_list).nonzero()
    edge_index = torch.LongTensor(edge_index)


    #### Remove self loops and then assign the structural identifiers by computing subgraph isomorphisms ####

    edge_list = []
    for i in range(len(edge_index[0])):
        row = int(edge_index[0][i])
        col = int(edge_index[1][i])
        edge_list.append((row, col))
    edge_index = edge_list

    num_nodes = subgraph_params['num_nodes']
    identifiers = None
    for subgraph_dict in subgraph_dicts:
        kwargs = {'subgraph_dict': subgraph_dict,
                  'induced': subgraph_params['induced'],
                  'num_nodes': num_nodes,
                  'directed': subgraph_params['directed']}
        if subgraph_params['vertex'] is True:
            counts = count_fn(edge_index, **kwargs)
            identifiers = counts if identifiers is None else torch.cat((identifiers, counts), 1)
        else:
            counts, index2edge = count_fn(edge_index, **kwargs)
            identifiers = counts if identifiers is None else torch.cat((identifiers, counts), 1)

    # 每次返回的index2edge都一致
    if subgraph_params['vertex'] is False:
        edge_identifiers = {}
        for i in range(len(identifiers)):
            edge_identifiers[index2edge[i]] = identifiers[i].tolist()
        identifiers = edge_identifiers

    return identifiers
