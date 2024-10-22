# Faster Local Solvers for Graph Diffusion Equations
This repository contains an implementation of Faster Local Solvers for Graph Diffusion Equations by Jiahe Bai, Baojian Zhou, Deqing Yang, Yanghua Xiao, to appear at NIPS 2024.

## Code

The ppr_solver.py, katz_solver.py, hk_solver.py, and InstantGNN.py are main algorithm codes and others are preprocessing codes.

## Running the demo

We provide a PPR experiment demo. You can run this command:
```
python demo.py --dataset dataset_name --eps 1e-7 --alpha 0.1 --opt_omega True --test_num 50
```
