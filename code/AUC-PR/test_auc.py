
import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning
import numpy as np
from utils.initialization_utils import initialize_experiment, initialize_model, set_params_from_model
from utils.graph_utils import collate_dgl_test, move_batch_to_device_dgl_test

from managers.evaluator_test import Evaluator
from warnings import simplefilter
from utils import parsing_utils as parse
from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
import shutil

def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    graph_classifier = initialize_model(params, None, load_model=True)
    set_params_from_model(params, graph_classifier)

    logging.info('============ Initialized logger ============')
    logging.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                           in sorted(dict(vars(params)).items())))
    logging.info('============================================')

    logging.info(f"Device: {params.device}")

    all_auc = []
    auc_mean = 0

    all_auc_pr = []
    auc_pr_mean = 0
    for r in range(1, params.runs + 1):
        params.db_path = os.path.join(params.main_dir,
                                      f'../../data/{params.dataset}/test_subgraphs_{params.experiment_name}_{params.constrained_neg_prob}_en_{params.enclosing_sub_graph}')

        if os.path.exists(params.db_path):
            shutil.rmtree(params.db_path)
        generate_subgraph_datasets(params, splits=['test'],
                                   saved_relation2id=graph_classifier.relation2id,
                                   max_label_value=graph_classifier.gnn.max_label_value)

        test = SubgraphDataset(params.db_path, 'test_pos', 'test_neg', params.file_paths, graph_classifier.relation2id,
                               add_traspose_rels=params.add_traspose_rels,
                               num_neg_samples_per_link=params.num_neg_samples_per_link,
                               use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                               kge_model=params.kge_model, file_name=params.test_file,
                               init_add_identifiers=params.init_add_identifiers,
                               identifiers_from_all_graph=params.identifiers_from_all_graph,
                               clamp_enable=params.clamp_enable, cut_enable=params.cut_enable,
                               node_identifier_min=params.node_identifier_min,
                               node_identifier_max=params.node_identifier_max,
                               edge_identifier_min=params.edge_identifier_min,
                               edge_identifier_max=params.edge_identifier_max)

        test_evaluator = Evaluator(params, graph_classifier, test)

        result = test_evaluator.eval(save=True)
        logging.info('\nTest Set Performance:' + str(result))
        all_auc.append(result['auc'])
        auc_mean = auc_mean + (result['auc'] - auc_mean) / r

        all_auc_pr.append(result['auc_pr'])
        auc_pr_mean = auc_pr_mean + (result['auc_pr'] - auc_pr_mean) / r

    auc_std = np.std(all_auc)
    auc_pr_std = np.std(all_auc_pr)
    avg_auc = np.mean(all_auc)
    avg_auc_pr = np.mean(all_auc_pr)

    # logging.info('\nAvg test Set Performance -- mean auc :' + str(avg_auc) + ' std auc: ' + str(auc_std))
    logging.info('\nAvg test Set Performance -- mean auc_pr :' + str(avg_auc_pr) + ' std auc_pr: ' + str(auc_pr_std))
    logging.info(f'auc_pr: {avg_auc_pr: .4f}')

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str, default="Toy",
                        help="Dataset string")
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--test_file", "-t", type=str, default="test",
                        help="Name of file containing test triplets")
    parser.add_argument("--runs", type=int, default=1,
                        help="How many runs to perform for mean and std?")
    parser.add_argument("--gpu", "-g", type=int, default=4,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=100000,
                        help="Set maximum number of links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument('--has_attn', '-attn', type=parse.str2bool, default=True,
                        help='whether to have attn in model or not')

    # GSN
    parser.add_argument('--id_type', type=str, default='cycle_graph')
    parser.add_argument('--induced', type=parse.str2bool, default=False)
    parser.add_argument('--edge_automorphism', type=str, default='induced')
    parser.add_argument('--k', type=parse.str2list2int, default=[3])
    parser.add_argument('--id_scope', type=str, default='local')
    parser.add_argument('--custom_edge_list', type=parse.str2ListOfListsOfLists2int, default=None)
    parser.add_argument('--directed', type=parse.str2bool, default=True)
    parser.add_argument('--directed_orbits', type=parse.str2bool, default=True)
    parser.add_argument('--init_add_identifiers', type=parse.str2bool, default=True)
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
    parser.add_argument('--cut_enable', type=parse.str2bool, default=False)
    parser.add_argument("--node_identifier_min", type=int, default=0)
    parser.add_argument("--node_identifier_max", type=int, default=16)
    parser.add_argument("--edge_identifier_min", type=int, default=0)
    parser.add_argument("--edge_identifier_max", type=int, default=32)

    params = parser.parse_args()
    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../../data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'test': os.path.join(params.main_dir, '../../data/{}/{}.txt'.format(params.dataset, params.test_file))
    }

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl_test
    params.move_batch_to_device = move_batch_to_device_dgl_test

    main(params)
