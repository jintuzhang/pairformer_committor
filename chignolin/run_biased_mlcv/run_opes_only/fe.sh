#!/bin/sh
export PLUMED_MAXBACKUP=1000000
rm -rf grids
mkdir grids
reweight.py --colvar colvar --cv z_fitted --outfile grids/ff.dat --sigma 0.01 --bin 200 --temp 340 --stride 1000
python ./fe.py > fe

rm -rf grids
