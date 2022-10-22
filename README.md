# Substructure-Aware Subgraph Reasoning for Inductive Relation Prediction

This repository contains the code and the datasets of **Substructure-Aware Subgraph Reasoning for Inductive Relation Prediction**. Kai Sun, HuaJie Jiang, YongLi Hu, BaoCai Yin

## Dependencies
The code is based on Python 3.7. You can use the following command to create a environment and enter it.
```shell script
conda create --name SASP_ENV python=3.7
source activate SASP_ENV
```
All the required packages can be installed by running 
```shell script
pip install -r requirements.txt
```
To test the code, run the following commands.

```shell script
cd code/AUC-PR
bash run_once.sh WN18RR_v1 8 1 2 0.01 0.01 demo_test 10 4 False True 1 1 False complete_graph_chosen_k 3 relu normalize 0 16 0 16 sum
```

Notice that, for the first time you run the code, it would take some time to sample the subgraph. 

## Reproduce the Results

To reproduce the results, run the following commands. 

```shell script
#################################### AUC-PR ####################################
cd code/AUC-PR
bash run_five.sh WN18RR_v1 8 1 2 0.01 0.01 demo 10 0 True True 1 1 True complete_graph_chosen_k 3 relu relu 0 16 0 32 sum
bash run_five.sh WN18RR_v2 8 1 2 0.01 0.01 demo 10 0 True True 1 2 True complete_graph_chosen_k 3 relu relu 0 16 0 32 sum
bash run_five.sh WN18RR_v3 8 1 2 0.01 0.01 demo 10 0 True True 1 0 True complete_graph_chosen_k 3 relu relu 0 32 0 32 sum
bash run_five.sh WN18RR_v4 8 1 2 0.01 0.01 demo 10 0 True True 1 1 True complete_graph_chosen_k 3 relu relu 0 16 0 32 sum

bash run_five.sh fb237_v1 16 1 2 0.01 0.01 demo 10 0 True True 1 0 False complete_graph_chosen_k 3 relu relu 0 16 0 32
bash run_five.sh fb237_v2 16 1 2 0.01 0.01 demo 10 0 True True 1 1 True complete_graph_chosen_k 4 relu relu 0 32 0 96 sum
bash run_five.sh fb237_v3 16 1 2 0.01 0.01 demo 10 0 False True 2 1 False complete_graph_chosen_k 4 relu normalize 0 16 0 16 gru
bash run_five.sh fb237_v4 16 1 2 0.01 0.01 demo 10 0 True True 1 1 True complete_graph_chosen_k 4 relu relu 0 16 0 32 sum

bash run_five.sh nell_v1 10 1 2 0.01 0.01 demo 10 0 True True 0 0 True complete_graph_chosen_k 4 relu relu 0 16 0 32 sum
bash run_five.sh nell_v2 10 1 2 0.01 0.01 demo 10 0 True True 0 0 True complete_graph_chosen_k 3 relu relu 0 32 0 32 sum
bash run_five.sh nell_v3 10 1 2 0.01 0.01 demo 10 0 True True 1 1 True complete_graph_chosen_k 4 relu relu 0 32 0 16 sum
bash run_five.sh nell_v4 10 1 2 0.01 0.1 demo 10 0 True True 0 0 True complete_graph_chosen_k 5 relu relu 0 64 0 32 sum

#################################### Ranking #############################
cd code/Ranking
bash run_five.sh WN18RR_v1 8 8 2 0.01 0.01 demo 10 0 True True 1 0 False complete_graph_chosen_k 3 relu relu
bash run_five.sh WN18RR_v4 8 8 2 0.01 0.01 demo 10 0 True True 1 0 False complete_graph_chosen_k 3 relu relu

bash run_five.sh fb237_v1 16 8 2 0.005 0.01 demo 10 0 False True 1 1 False complete_graph_chosen_k 3 relu relu 0 16 0 32 sum
bash run_five.sh fb237_v4 16 8 2 0.005 0.01 demo 10 0 True True 0 0 True complete_graph_chosen_k 3 relu relu 16 32

bash run_five.sh nell_v1 10 8 2 0.01 0.01 demo 10 0 True True 0 0 True complete_graph_chosen_k 3 relu relu
bash run_five.sh nell_v4 16 8 2 0.008 0.01 demo 5 0 True True 0 0 False complete_graph_chosen_k 3 relu relu 0 128 0 32 sum
```

## Acknowledgement

We refer to the code of [GraIL](https://github.com/kkteru/grail). Thanks for their contributions.
