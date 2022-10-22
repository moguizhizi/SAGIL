#!/usr/bin/env bash

DATASET=$1
MR=$2
NUM_NEG=$3
HOP=$4
LR=$5
L2=$6
EXP_NAME=$7
EPOCH=$8
GPU=$9
INIT_ADD_IDENTIFIERS=${10}
CUT_ENABLE=${11}
VERSION=${12}
GSN_TYPE=${13}
ONE_HOT_ENABLE=${14}
ID_TYPE=${15}
K=${16}
ACTIVATE_FUNC_TYPE=${17}
EDGE_IDENTIFIER_ACTIVATE_FUNC_TYPE=${18}
MIN_NODE=${19}
MAX_NODE=${20}
MIN_EDGE=${21}
MAX_EDGE=${22}
GNN_AGG_TYPE=${23}

python train.py -d "$DATASET" -e "$DATASET"_"$EXP_NAME"_"$EPOCH" --num_epochs $EPOCH -g $GPU --hop $HOP --lr $LR --margin $MR \
--num_neg_samples_per_link $NUM_NEG --l2 $L2 --num_gcn_layers 2 --no_jk --init_add_identifiers $INIT_ADD_IDENTIFIERS \
--cut_enable $CUT_ENABLE --version $VERSION --gsn_type $GSN_TYPE --one_hot_enable $ONE_HOT_ENABLE --id_type $ID_TYPE \
--k $K --activate_func_type $ACTIVATE_FUNC_TYPE --edge_identifier_activate_func_type $EDGE_IDENTIFIER_ACTIVATE_FUNC_TYPE \
--node_identifier_max $MAX_NODE --edge_identifier_max $MAX_EDGE --node_identifier_min $MIN_NODE --edge_identifier_min $MIN_EDGE \
--gnn_agg_type $GNN_AGG_TYPE
python test_ranking.py -d "$DATASET"_"ind" -e "$DATASET"_"$EXP_NAME"_"$EPOCH" --hop $HOP --disable_cuda