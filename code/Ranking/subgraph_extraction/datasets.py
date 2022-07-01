from torch.utils.data import Dataset
import json
import dgl
import copy
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, save_to_file, plot_rel_dist
from .graph_sampler import *
from utils.ids_utils import subgraph_counts2ids
from utils.graph_processing_utils import get_custom_edge_list, subgraph_isomorphism_edge_counts, \
    subgraph_isomorphism_vertex_counts
from utils.graph_processing_utils import induced_edge_automorphism_orbits, automorphism_orbits, edge_automorphism_orbits
from utils.graph_utils import deserialize, deserialize_edge_identifiers, deserialize_node_identifiers, \
    deserialize_for_subgraph
import networkx as nx
import os


def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):
    testing = 'test' in splits
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths,
                                                                                       saved_relation2id)

    identifiers = {}
    identifiers['node_identifier'] = None
    identifiers['edge_identifier'] = None
    if params.identifiers_from_all_graph is True:
        identifiers = get_identifier(adj_list, len(entity2id), params)

    data_path = os.path.join(params.main_dir, f'../../data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}

    # Sample train and valid/test links
    for split_name, split in graphs.items():
        split['pos'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link,
                                  max_size=split['max_size'],
                                  constrained_neg_prob=params.constrained_neg_prob)

    if testing:
        directory = os.path.join(params.main_dir, '../data/{}/'.format(params.dataset))
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'],
                     id2entity, id2relation)

    links2subgraphs(adj_list, graphs, params, max_label_value, identifiers)


def get_node_identifier(adj_list, num_nodes, params):
    extract_id_fn = subgraph_counts2ids
    ###### choose the function that computes the subgraph isomorphisms #######
    count_fn = subgraph_isomorphism_vertex_counts
    ###### choose the substructures: usually loaded from networkx,
    ###### except for 'all_simple_graphs' where they need to be precomputed,
    ###### or when a custom edge list is provided in the input by the user
    if params.id_type in ['cycle_graph',
                          'path_graph',
                          'complete_graph',
                          'binomial_tree',
                          'star_graph',
                          'nonisomorphic_trees']:
        params.k = params.k[0]
        k_max = params.k
        k_min = 2 if params.id_type == 'star_graph' else 3
        params.custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), params.id_type)

    elif params.id_type in ['cycle_graph_chosen_k',
                            'path_graph_chosen_k',
                            'complete_graph_chosen_k',
                            'binomial_tree_chosen_k',
                            'star_graph_chosen_k',
                            'nonisomorphic_trees_chosen_k']:
        params.custom_edge_list = get_custom_edge_list(params.k, params.id_type.replace('_chosen_k', ''))

    elif params.id_type in ['all_simple_graphs']:
        params.k = params.k[0]
        k_max = params.k
        k_min = 3
        filename = os.path.join(params.root_folder, 'all_simple_graphs')
        params.custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), filename=filename)

    elif params.id_type in ['all_simple_graphs_chosen_k']:
        filename = os.path.join(params.root_folder, 'all_simple_graphs')
        params.custom_edge_list = get_custom_edge_list(params.k, filename=filename)

    elif params.id_type in ['diamond_graph']:
        params.k = None
        graph_nx = nx.diamond_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['bull_graph']:
        params.k = None
        graph_nx = nx.bull_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['chvatal_graph']:
        params.k = None
        graph_nx = nx.chvatal_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['cubical_graph']:
        params.k = None
        graph_nx = nx.cubical_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['desargues_graph']:
        params.k = None
        graph_nx = nx.desargues_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['dodecahedral_graph']:
        params.k = None
        graph_nx = nx.dodecahedral_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['frucht_graph']:
        params.k = None
        graph_nx = nx.frucht_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['heawood_graph']:
        params.k = None
        graph_nx = nx.heawood_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['hoffman_singleton_graph']:
        params.k = None
        graph_nx = nx.hoffman_singleton_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['house_graph']:
        params.k = None
        graph_nx = nx.house_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['house_x_graph']:
        params.k = None
        graph_nx = nx.house_x_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['octahedral_graph']:
        params.k = None
        graph_nx = nx.octahedral_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['tetrahedral_graph']:
        params.k = None
        graph_nx = nx.tetrahedral_graph()
        params.custom_edge_list = [list(graph_nx.edges)]


    elif params.id_type == 'custom':
        assert params.custom_edge_list is not None, "Custom edge list must be provided."

    else:
        raise NotImplementedError("Identifiers {} are not currently supported.".format(params.id_type))

    if params.edge_automorphism == 'induced' or params.edge_automorphism == 'line_graph':
        automorphism_fn = automorphism_orbits
    else:
        raise NotImplementedError

    subgraph_params = {'induced': params.induced,
                       'edge_list': params.custom_edge_list,
                       'directed': False,
                       'directed_orbits': False,
                       'num_nodes': num_nodes,
                       'vertex': True}
    ### compute the orbits of earch substructure in the list, as well as the vertex automorphism count
    subgraph_dicts = []
    orbit_partition_sizes = []
    if 'edge_list' not in subgraph_params:
        raise ValueError('Edge list not provided.')
    for edge_list in subgraph_params['edge_list']:
        subgraph, orbit_partition, orbit_membership, aut_count = \
            automorphism_fn(edge_list=edge_list,
                            directed=subgraph_params['directed'],
                            directed_orbits=subgraph_params['directed_orbits'])
        subgraph_dicts.append({'subgraph': subgraph, 'orbit_partition': orbit_partition,
                               'orbit_membership': orbit_membership, 'aut_count': aut_count})
        orbit_partition_sizes.append(len(orbit_partition))
    node_identifier = extract_id_fn(count_fn, adj_list, subgraph_dicts, subgraph_params)

    return node_identifier


def get_undirected_edge_identifier(adj_list, num_nodes, params):
    extract_id_fn = subgraph_counts2ids
    ###### choose the function that computes the subgraph isomorphisms #######
    count_fn = subgraph_isomorphism_edge_counts
    ###### choose the substructures: usually loaded from networkx,
    ###### except for 'all_simple_graphs' where they need to be precomputed,
    ###### or when a custom edge list is provided in the input by the user
    if params.id_type in ['cycle_graph',
                          'path_graph',
                          'complete_graph',
                          'binomial_tree',
                          'star_graph',
                          'nonisomorphic_trees']:
        params.k = params.k[0]
        k_max = params.k
        k_min = 2 if params.id_type == 'star_graph' else 3
        params.custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), params.id_type)

    elif params.id_type in ['cycle_graph_chosen_k',
                            'path_graph_chosen_k',
                            'complete_graph_chosen_k',
                            'binomial_tree_chosen_k',
                            'star_graph_chosen_k',
                            'nonisomorphic_trees_chosen_k']:
        params.custom_edge_list = get_custom_edge_list(params.k, params.id_type.replace('_chosen_k', ''))

    elif params.id_type in ['all_simple_graphs']:
        params.k = params.k[0]
        k_max = params.k
        k_min = 3
        filename = os.path.join(params.root_folder, 'all_simple_graphs')
        params.custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), filename=filename)

    elif params.id_type in ['all_simple_graphs_chosen_k']:
        filename = os.path.join(params.root_folder, 'all_simple_graphs')
        params.custom_edge_list = get_custom_edge_list(params.k, filename=filename)

    elif params.id_type in ['diamond_graph']:
        params.k = None
        graph_nx = nx.diamond_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['bull_graph']:
        params.k = None
        graph_nx = nx.bull_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['chvatal_graph']:
        params.k = None
        graph_nx = nx.chvatal_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['cubical_graph']:
        params.k = None
        graph_nx = nx.cubical_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['desargues_graph']:
        params.k = None
        graph_nx = nx.desargues_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['dodecahedral_graph']:
        params.k = None
        graph_nx = nx.dodecahedral_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['frucht_graph']:
        params.k = None
        graph_nx = nx.frucht_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['heawood_graph']:
        params.k = None
        graph_nx = nx.heawood_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['hoffman_singleton_graph']:
        params.k = None
        graph_nx = nx.hoffman_singleton_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['house_graph']:
        params.k = None
        graph_nx = nx.house_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['house_x_graph']:
        params.k = None
        graph_nx = nx.house_x_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['octahedral_graph']:
        params.k = None
        graph_nx = nx.octahedral_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['tetrahedral_graph']:
        params.k = None
        graph_nx = nx.tetrahedral_graph()
        params.custom_edge_list = [list(graph_nx.edges)]



    elif params.id_type == 'custom':
        assert params.custom_edge_list is not None, "Custom edge list must be provided."

    else:
        raise NotImplementedError("Identifiers {} are not currently supported.".format(params.id_type))

    if params.edge_automorphism == 'induced':
        automorphism_fn = induced_edge_automorphism_orbits
    elif params.edge_automorphism == 'line_graph':
        automorphism_fn = edge_automorphism_orbits
    else:
        raise NotImplementedError
    subgraph_params = {'induced': params.induced,
                       'edge_list': params.custom_edge_list,
                       'directed': False,
                       'directed_orbits': False,
                       'num_nodes': num_nodes,
                       'vertex': False}
    ### compute the orbits of earch substructure in the list, as well as the vertex automorphism count
    subgraph_dicts = []
    orbit_partition_sizes = []
    if 'edge_list' not in subgraph_params:
        raise ValueError('Edge list not provided.')
    for edge_list in subgraph_params['edge_list']:
        subgraph, orbit_partition, orbit_membership, aut_count = \
            automorphism_fn(edge_list=edge_list,
                            directed=subgraph_params['directed'],
                            directed_orbits=subgraph_params['directed_orbits'])
        subgraph_dicts.append({'subgraph': subgraph, 'orbit_partition': orbit_partition,
                               'orbit_membership': orbit_membership, 'aut_count': aut_count})
        orbit_partition_sizes.append(len(orbit_partition))
    undirected_edge_identifier = extract_id_fn(count_fn, adj_list, subgraph_dicts, subgraph_params)

    return undirected_edge_identifier


def get_direct_edge_identifier(adj_list, num_nodes, params):
    extract_id_fn = subgraph_counts2ids
    ###### choose the function that computes the subgraph isomorphisms #######
    count_fn = subgraph_isomorphism_edge_counts
    ###### choose the substructures: usually loaded from networkx,
    ###### except for 'all_simple_graphs' where they need to be precomputed,
    ###### or when a custom edge list is provided in the input by the user
    if params.id_type in ['cycle_graph',
                          'path_graph',
                          'complete_graph',
                          'binomial_tree',
                          'star_graph',
                          'nonisomorphic_trees']:
        params.k = params.k[0]
        k_max = params.k
        k_min = 2 if params.id_type == 'star_graph' else 3
        params.custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), params.id_type)

    elif params.id_type in ['cycle_graph_chosen_k',
                            'path_graph_chosen_k',
                            'complete_graph_chosen_k',
                            'binomial_tree_chosen_k',
                            'star_graph_chosen_k',
                            'nonisomorphic_trees_chosen_k']:
        params.custom_edge_list = get_custom_edge_list(params.k, params.id_type.replace('_chosen_k', ''))

    elif params.id_type in ['all_simple_graphs']:
        params.k = params.k[0]
        k_max = params.k
        k_min = 3
        filename = os.path.join(params.root_folder, 'all_simple_graphs')
        params.custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), filename=filename)

    elif params.id_type in ['all_simple_graphs_chosen_k']:
        filename = os.path.join(params.root_folder, 'all_simple_graphs')
        params.custom_edge_list = get_custom_edge_list(params.k, filename=filename)

    elif params.id_type in ['diamond_graph']:
        params.k = None
        graph_nx = nx.diamond_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['bull_graph']:
        params.k = None
        graph_nx = nx.bull_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['chvatal_graph']:
        params.k = None
        graph_nx = nx.chvatal_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['cubical_graph']:
        params.k = None
        graph_nx = nx.cubical_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['desargues_graph']:
        params.k = None
        graph_nx = nx.desargues_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['dodecahedral_graph']:
        params.k = None
        graph_nx = nx.dodecahedral_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['frucht_graph']:
        params.k = None
        graph_nx = nx.frucht_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['heawood_graph']:
        params.k = None
        graph_nx = nx.heawood_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['hoffman_singleton_graph']:
        params.k = None
        graph_nx = nx.hoffman_singleton_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['house_graph']:
        params.k = None
        graph_nx = nx.house_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['house_x_graph']:
        params.k = None
        graph_nx = nx.house_x_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['octahedral_graph']:
        params.k = None
        graph_nx = nx.octahedral_graph()
        params.custom_edge_list = [list(graph_nx.edges)]

    elif params.id_type in ['tetrahedral_graph']:
        params.k = None
        graph_nx = nx.tetrahedral_graph()
        params.custom_edge_list = [list(graph_nx.edges)]



    elif params.id_type == 'custom':
        assert params.custom_edge_list is not None, "Custom edge list must be provided."

    else:
        raise NotImplementedError("Identifiers {} are not currently supported.".format(params.id_type))
    if params.edge_automorphism == 'induced':
        automorphism_fn = induced_edge_automorphism_orbits
    elif params.edge_automorphism == 'line_graph':
        automorphism_fn = edge_automorphism_orbits
    else:
        raise NotImplementedError
    subgraph_params = {'induced': params.induced,
                       'edge_list': params.custom_edge_list,
                       'directed': True,
                       'directed_orbits': True,
                       'num_nodes': num_nodes,
                       'vertex': False}
    ### compute the orbits of earch substructure in the list, as well as the vertex automorphism count
    subgraph_dicts = []
    orbit_partition_sizes = []
    if 'edge_list' not in subgraph_params:
        raise ValueError('Edge list not provided.')
    for edge_list in subgraph_params['edge_list']:
        subgraph, orbit_partition, orbit_membership, aut_count = \
            automorphism_fn(edge_list=edge_list,
                            directed=subgraph_params['directed'],
                            directed_orbits=subgraph_params['directed_orbits'])
        subgraph_dicts.append({'subgraph': subgraph, 'orbit_partition': orbit_partition,
                               'orbit_membership': orbit_membership, 'aut_count': aut_count})
        orbit_partition_sizes.append(len(orbit_partition))
    direct_edge_identifier = extract_id_fn(count_fn, adj_list, subgraph_dicts, subgraph_params)

    return direct_edge_identifier


def get_identifier(adj_list, num_nodes, params):
    identifier = {}

    params_node = copy.copy(params)
    params_undirected_edge = copy.copy(params)
    params_direct_edge = copy.copy(params)
    node_identifier = get_node_identifier(adj_list, num_nodes, params_node)

    if params.directed is True:
        edge_identifier = get_direct_edge_identifier(adj_list, num_nodes, params_direct_edge)
    else:
        edge_identifier = get_undirected_edge_identifier(adj_list, num_nodes, params_undirected_edge)

    identifier['node_identifier'] = node_identifier
    identifier['edge_identifier'] = edge_identifier

    return identifier

class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None,
                 add_traspose_rels=False, num_neg_samples_per_link=1, file_name='', init_add_identifiers=False,
                 hit_eval=False,
                 identifiers_from_all_graph=False, clamp_enable=False, cut_enable=False,
                 node_identifier_min=0, node_identifier_max=0,
                 edge_identifier_min=0, edge_identifier_max=0):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        ##### del neg
        # self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name

        self.init_add_identifiers = init_add_identifiers
        self.hit_eval = hit_eval
        self.identifiers_from_all_graph = identifiers_from_all_graph
        self.clamp_enable = clamp_enable
        self.cut_enable = cut_enable
        self.node_identifier_min = node_identifier_min
        self.node_identifier_max = node_identifier_max
        self.edge_identifier_min = edge_identifier_min
        self.edge_identifier_max = edge_identifier_max
        log_max_edge_identifiers = None
        log_max_nodes_identifiers = None

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

            self.edge_identifiers = None
            if txn.get('edge_identifiers'.encode()) is not None:
                record = deserialize_edge_identifiers(txn.get('edge_identifiers'.encode()))
                self.edge_identifiers = record['edge_identifiers']

            self.node_identifiers = None
            if txn.get('node_identifier'.encode()) is not None:
                record = deserialize_node_identifiers(txn.get('node_identifier'.encode()))
                self.node_identifiers = record['node_identifier']

            self.num_dim_edge_identifiers = 0
            if txn.get('max_num_dim_edge_identifiers'.encode()) is not None:
                self.num_dim_edge_identifiers = int.from_bytes(
                    txn.get('max_num_dim_edge_identifiers'.encode()),
                    byteorder='little')

            self.max_edge_identifiers = None
            if txn.get('max_edge_identifiers'.encode()) is not None:
                self.max_edge_identifiers = deserialize_for_subgraph(txn.get('max_edge_identifiers'.encode()))[0]
                log_max_edge_identifiers = self.max_edge_identifiers

            self.max_nodes_identifiers = None
            if txn.get('max_nodes_identifiers'.encode()) is not None:
                self.max_nodes_identifiers = deserialize_for_subgraph(txn.get('max_nodes_identifiers'.encode()))[0]
                log_max_nodes_identifiers = self.max_nodes_identifiers

            if self.clamp_enable is True or self.cut_enable is True:
                self.max_nodes_identifiers = (
                        torch.ones(len(self.max_nodes_identifiers)) * self.node_identifier_max).tolist()
                self.max_edge_identifiers = (
                        torch.ones(len(self.max_edge_identifiers)) * self.edge_identifier_max).tolist()

                if self.node_identifiers is not None:
                    if self.cut_enable is not True:
                        self.node_identifiers = torch.clamp(torch.FloatTensor(self.node_identifiers),
                                                            min=self.node_identifier_min,
                                                            max=self.node_identifier_max).tolist()

                    else:
                        temp_node_identifiers = torch.FloatTensor(self.node_identifiers)
                        zeros = torch.zeros_like(temp_node_identifiers)
                        filter_index = (temp_node_identifiers >= self.node_identifier_min) * (
                                temp_node_identifiers <= self.node_identifier_max)
                        self.node_identifiers = torch.where(filter_index, temp_node_identifiers, zeros).tolist()

                if self.edge_identifiers is not None:
                    if self.cut_enable is not True:
                        for key, value in self.edge_identifiers.items():
                            value = torch.clamp(torch.FloatTensor(value),
                                                min=self.edge_identifier_min,
                                                max=self.edge_identifier_max).tolist()
                            self.edge_identifiers[key] = value
                    else:
                        for key, value in self.edge_identifiers.items():
                            temp_edge_identifiers = list(value)
                            temp_edge_identifiers = torch.FloatTensor(temp_edge_identifiers)
                            zeros = torch.zeros_like(temp_edge_identifiers)
                            filter_index = (temp_edge_identifiers >= self.edge_identifier_min) * (
                                    temp_edge_identifiers <= self.edge_identifier_max)
                            temp_edge_identifiers = torch.where(filter_index, temp_edge_identifiers, zeros).tolist()
                            self.edge_identifiers[key] = temp_edge_identifiers

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")
        logging.info(
            f"max edge identifiers : {log_max_edge_identifiers}, max nodes identifiers : {log_max_nodes_identifiers}")

        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        self.relation_list = list(id2relation.keys())
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)

        self.graph = ssp_multigraph_to_dgl(ssp_graph, self.edge_identifiers)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            record = deserialize(txn.get(str_id))
            nodes_pos = record['nodes']
            r_label_pos = record['r_label']
            g_label_pos = record['g_label']
            n_labels_pos = record['n_label']
            identifiers_pos = None
            if self.node_identifiers is not None:
                identifiers_pos = torch.FloatTensor(self.node_identifiers)[nodes_pos].tolist()
            nodes_identifiers_from_subgraph_pos = record['nodes_identifiers_from_subgraph']
            edge_identifiers_from_subgraph_pos = record['edge_identifiers_from_subgraph']

            edge_identifiers_from_subgraph_pos, nodes_identifiers_from_subgraph_pos = self.clamp_process(
                edge_identifiers_from_subgraph_pos, nodes_identifiers_from_subgraph_pos)

            edge_identifiers_from_subgraph_pos, nodes_identifiers_from_subgraph_pos = self.cut_process(
                edge_identifiers_from_subgraph_pos, nodes_identifiers_from_subgraph_pos)
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos, identifiers=identifiers_pos,
                                                   nodes_identifiers_from_subgraph=nodes_identifiers_from_subgraph_pos,
                                                   edge_identifiers_from_subgraph=edge_identifiers_from_subgraph_pos,
                                                   num_dim_edge_identifiers=self.num_dim_edge_identifiers)

        return subgraph_pos, g_label_pos, r_label_pos

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels, identifiers=None,
                           nodes_identifiers_from_subgraph=None,
                           edge_identifiers_from_subgraph=None,
                           num_dim_edge_identifiers=0):
        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        if self.identifiers_from_all_graph is True:
            subgraph.edata['identifier'] = self.graph.edata['identifier'][self.graph.subgraph(nodes).parent_eid]
            pad_identifier = torch.zeros(num_dim_edge_identifiers, dtype=int).unsqueeze(0)
            length = len(self.graph.edata['identifier'][self.graph.subgraph(nodes).parent_eid])
        else:
            pad_identifier = torch.zeros(num_dim_edge_identifiers, dtype=int).unsqueeze(0)
            if subgraph.number_of_edges() != 0:
                subgraph.edata['identifier'] = torch.IntTensor(edge_identifiers_from_subgraph)
            length = len(edge_identifiers_from_subgraph)

        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            if length == 0:
                subgraph.edata['identifier'] = pad_identifier
            else:
                subgraph.edata['identifier'][-1] = pad_identifier

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats, identifiers,
                                              nodes_identifiers_from_subgraph=nodes_identifiers_from_subgraph)

        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None, identifiers=None,
                              nodes_identifiers_from_subgraph=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats

        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        if identifiers is not None and len(identifiers) != 0:
            subgraph.ndata['identifier'] = torch.IntTensor(identifiers)
        elif nodes_identifiers_from_subgraph is not None:
            subgraph.ndata['identifier'] = torch.IntTensor(nodes_identifiers_from_subgraph)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def clamp_process(self, edge_identifiers_from_subgraph, nodes_identifiers_from_subgraph):
        if self.clamp_enable is True:
            if len(nodes_identifiers_from_subgraph) != 0:
                nodes_identifiers_from_subgraph = torch.clamp(
                    torch.FloatTensor(nodes_identifiers_from_subgraph), min=self.node_identifier_min,
                    max=self.node_identifier_max).tolist()
            if len(edge_identifiers_from_subgraph) != 0:
                edge_identifiers_from_subgraph = torch.clamp(
                    torch.FloatTensor(edge_identifiers_from_subgraph), min=self.edge_identifier_min,
                    max=self.edge_identifier_max).tolist()

        return edge_identifiers_from_subgraph, nodes_identifiers_from_subgraph

    def cut_process(self, edge_identifiers_from_subgraph, nodes_identifiers_from_subgraph):
        if self.cut_enable is True:
            temp_node_identifiers = torch.FloatTensor(nodes_identifiers_from_subgraph)
            zeros = torch.zeros_like(temp_node_identifiers)
            filter_index = (temp_node_identifiers >= self.node_identifier_min) * (
                    temp_node_identifiers <= self.node_identifier_max)
            nodes_identifiers_from_subgraph = torch.where(filter_index, temp_node_identifiers, zeros).tolist()

            temp_edge_identifiers = torch.FloatTensor(edge_identifiers_from_subgraph)
            zeros = torch.zeros_like(temp_edge_identifiers)
            filter_index = (temp_edge_identifiers >= self.edge_identifier_min) * (
                    temp_edge_identifiers <= self.edge_identifier_max)
            edge_identifiers_from_subgraph = torch.where(filter_index, temp_edge_identifiers, zeros).tolist()

        return edge_identifiers_from_subgraph, nodes_identifiers_from_subgraph