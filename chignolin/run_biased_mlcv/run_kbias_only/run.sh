#!/bin/bash
#SBATCH -J host
#SBATCH -o host.log
#SBATCH -e host.err
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH -w gpu04

export OMP_NUM_THREADS=1
export OPENMM_NUM_THREADS=1
export PLUMED_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

source /home/zhangjintu/softwares/gromacs/bin/GMXRC
source /home/zhangjintu/softwares/plumed.python.mace/bin/activate

export LD_PRELOAD="/home/zhangjintu/softwares/plumed.python.mace/lib/libplumed.so /home/zhangjintu/softwares/plumed.python.mace/lib/libplumedKernel.so /home/zhangjintu/softwares/miniforge3/envs/MACE/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so /home/zhangjintu/softwares/miniforge3/envs/MACE/lib/python3.12/site-packages/torch/lib/libc10.so /home/zhangjintu/softwares/miniforge3/envs/MACE/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so /home/zhangjintu/softwares/miniforge3/envs/MACE/lib/python3.12/site-packages/torch/lib/libc10_cuda.so /home/zhangjintu/softwares/miniforge3/envs/MACE/lib/python3.12/site-packages/torch/lib/libtorch.so"

gmx grompp -f md.mdp -c ../../data/gmx/ts_1.gro -p ../../data/gmx/topol.top -o chignolin.tpr
# gmx grompp -f md.mdp -c ../../data/gmx/ts_2.gro -p ../../data/gmx/topol.top -o chignolin.tpr
gmx mdrun -deffnm chignolin -plumed plumed.dat -gpu_id 0
