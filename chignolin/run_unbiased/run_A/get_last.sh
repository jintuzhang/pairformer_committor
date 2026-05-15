#!/bin/sh

# grep "^#" ./colvar > colvar.last
# tail -n 10000 ./colvar >> colvar.last
# mdconvert ./traj.dcd -fo traj.last.dcd -i 2500:12500

python -c "import numpy; import mdtraj; print('err.: ', (numpy.loadtxt('colvar')[:, 1] - mdtraj.compute_distances(mdtraj.load('traj.dcd', top='../../data/gmx/folded.gro'), [[4,146]], periodic=False).T[0]).max(), ' ')"
