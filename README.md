# Faster Local Solvers for Graph Diffusion Equations
This repository contains an implementation of [Faster Local Solvers for Graph Diffusion Equations](https://arxiv.org/pdf/2410.21634) by Jiahe Bai, Baojian Zhou, Deqing Yang, Yanghua Xiao, to appear at NIPS 2024.

## Code

The `ppr_solver.py`, `katz_solver.py`, `hk_solver.py`, and `InstantGNN.py` are main algorithm codes and others are preprocessing and experiment codes.

## Requirement

The `requirements.txt` file lists all Python libraries that our code depend on, and they will be installed using:
```
pip install -r requirements.txt
```

## Requirement

The `requirements.txt` file lists all Python libraries that our code depend on, and they will be installed using:
```
pip install -r requirements.txt
```

## Running the demo

We provide a PPR experiment demo. You can run this command:
```
python demo.py --dataset wiki-talk --eps 1e-7 --alpha 0.1 --opt_omega True --test_num 50
```
This code will randomly select 50 points from the wiki-talk dataset and compute the PPR vectors using all methods we have implemented. It will output the average running time and number of operations for each method. The output looks like:
```
wiki-talk:
local_sor_opers : 13487658.94
local_sor_algo_times : 1.4423852968867867
global_sor_opers : 206194825.54
global_sor_algo_times : 3.5041825445089487
local_gs_opers : 10859586.04
local_gs_algo_times : 1.0632941502891482
global_gs_opers : 184662562.26
global_gs_algo_times : 3.037390498318709
local_gd_opers : 10350181.24
local_gd_algo_times : 1.2028513477742673
global_gd_opers : 362303734.32
global_gd_algo_times : 10.632122597224079
local_ch_opers : 48239385.42
local_ch_algo_times : 3.2669592004548758
global_ch_opers : 280855608.0
global_ch_algo_times : 11.393353031598963
local_gd_gpu_algo_times : 0.12180607559625059
global_gd_gpu_algo_times : 1.7817821394117213
```
## Testing the impact of $\epsilon$ on the computation of PPR

We provide an experiment code for testing the impact of $\epsilon$ on the computation of PPR. You can run this command:
```
python test_epsilon.py --dataset wiki-talk --alpha 0.1 --opt_omega True --test_num 50
```
The results will be stored in `./results/wiki-talk_ppr_exp_ratio_result.npy`. To visualize some results, you can run this command:
```
python visualize_wiki-talk_epsilon_GD.py
```
Then you can get `wiki-talk_ppr_exp_gd_ratio.png` under folder `figs`. It's the first figure of Figure 6 in our paper and it looks like:

<img src="figs/wiki-talk_ppr_exp_gd_ratio.png" width=50%>

## Running the experiments on GNN models

We provide an experiment code for GNN models on ogbn-arxiv dataset. To use it, you should run this command first:
```
python data_process_GNN.py --dataset 'ogbn-arxiv' --delnum 950000 --snapshots 16
```
This code will construct a dynamic graph procedure and store it in `./dataset/ogbn-arxiv-exp.npz`. Then you can run this command for dynamic PPR approximating and GNN models training:
```
python instantGNN-arxiv-exp.py
```
The results will be stored in `./results/instantGNN_ogbn-arxiv_fwd_result.npz` and `./results/instantGNN_ogbn-arxiv_sor_result.npz`. To visualize it,  you can run this command:
```
python visualize_ogbn-arxiv_InstantGNN.py
```
Then you can get `ogbn-arxiv_instantGNN_runtime.png` and `ogbn-arxiv_instantGNN_accuracy.png` under folder `figs`. It's Figure 8 in our paper and it looks like:

<img src="figs/ogbn-arxiv_instantGNN_runtime.png" width=55%><img src="figs/ogbn-arxiv_instantGNN_accuracy.png" width=35%>
