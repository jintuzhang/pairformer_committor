#!/bin/sh

echo 0 | gmx trjconv -pbc whole -f ../run_biased_mlcv/G2/G2.trr -o G2_whole.xtc -s ../run_biased_mlcv/G2/G2.tpr
echo 0 | gmx trjconv -pbc whole -f ../run_biased_mlcv/G5/G5.trr -o G5_whole.xtc -s ../run_biased_mlcv/G5/G5.tpr
echo 0 | gmx trjconv -pbc whole -f ../run_biased_mlcv/G6/G6.trr -o G6_whole.xtc -s ../run_biased_mlcv/G6/G6.tpr
