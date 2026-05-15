import plumed
import mdtraj as md


d = plumed.read_as_pandas('colvar')
t: md.Trajectory = md.load('chignolin.xtc', top='../../data/gmx/folded.gro')
d_3_6 = md.compute_distances(t, [[52, 85], [53, 85]]).min(axis=1)
d['d_asp3od_thr6n'] = d_3_6
plumed.write_pandas(d, 'colvar')
