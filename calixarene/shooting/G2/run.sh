#!/bin/bash

export OMP_NUM_THREADS=1
export OPENMM_NUM_THREADS=1
export PLUMED_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

source ~/jintu/bin/activate

mkdir -p $(seq 0 5)

for i in $(seq 0 5)
do
	cd $i
	mkdir -p $(seq 0 19)
	for j in $(seq 0 19)
	do
		cd $j
		gmx grompp -f ../../md.mdp -c ../../data/$i.gro -p ../../../../../data/G2/topol.top -o G2.tpr
		gmx mdrun -deffnm G2 -plumed ../../plumed.dat -gpu_id 0
		cd ../
	done
	cd ../
done
