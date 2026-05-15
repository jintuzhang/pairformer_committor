# PairFormer Committor

This repository contains all the input files and data related to the paper
"[Navigating committor landscape of biomolecules with a general pairwise interaction model.]()".

The following directories contains the training script, simulation setup and
models for the different systems presented in the manuscript:
```bash
.
├── alad
├── chignolin
└── calixarene
```
Besides, the `plumed_pytorch_pairformer` directory contains the cpp interface
for `PLUMED`, the `reweight` directory contains a script to reweight OPES
simulations (it was taken from
[here](https://github.com/invemichele/opes/blob/master/postprocessing/FES_from_Reweighting.py)),
and the `env.yaml` file contains the `conda` environment.

The definition and training of Pairformer committor model is available through
[a fork of the mlcolvar library](https://github.com/jintuzhang/mlcolvar).
