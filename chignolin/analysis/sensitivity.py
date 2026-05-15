import torch
import numpy as np
import pandas as pd

import plumed
import mdtraj as md
import mlcolvar.pairformer as mpair
from mlcolvar.utils import io as ffio

mpair.utils.torch_tools.set_default_dtype('float32')


dataset = mpair.utils.io.create_dataset_from_trajectories(
    trajectories=[
        '../run_unbiased/run_A/traj.dcd',
        '../run_unbiased/run_B/traj.dcd',
        '../data/chignolin_traindata_pkang/iter_5_clean/0/traj.dcd',
        '../data/chignolin_traindata_pkang/iter_5_clean/1/traj.dcd',
        '../data/chignolin_traindata_pkang/iter_5_clean/2/traj.dcd',
        '../data/chignolin_traindata_pkang/iter_5_clean/3/traj.dcd',
    ],
    top=[
        '../data/gmx/folded.gro',
        '../data/gmx/unfolded.gro',
        '../data/gmx/rmsd_ref_ca.pdb',
        '../data/gmx/rmsd_ref_ca.pdb',
        '../data/gmx/rmsd_ref_ca.pdb',
        '../data/gmx/rmsd_ref_ca.pdb',
    ],
    cutoff=-1,  # should be negative
    create_labels=True,
    system_selection='protein and backbone and not type H',
    node_embeddings=['atom_names', 'residue_names'],
    n_workers=128,
    no_pbc=True,
)

ff_dataset, ff_dataframe = ffio.create_dataset_from_files(
    file_names=[
        '../run_unbiased/run_A/colvar',
        '../run_unbiased/run_B/colvar',
        '../data/chignolin_traindata_pkang/iter_5_clean/0/colvar',
        '../data/chignolin_traindata_pkang/iter_5_clean/1/colvar',
        '../data/chignolin_traindata_pkang/iter_5_clean/2/colvar',
        '../data/chignolin_traindata_pkang/iter_5_clean/3/colvar',
    ],
    return_dataframe=True,
)
dataframe = ff_dataframe.fillna({
        'w1.bias': 0,
        'w2.bias': 0,
        'w3.bias': 0,
        'potential.bias': 0,
        'opes.bias': 0,
})
bias_w1 = torch.tensor(dataframe['w1.bias'].values)
bias_w2 = torch.tensor(dataframe['w2.bias'].values)
bias_w3 = torch.tensor(dataframe['w3.bias'].values)
bias_opes = torch.tensor(dataframe['opes.bias'].values)
bias_k = torch.tensor(dataframe['potential.bias'].values)
bias = bias_w1 + bias_w2 + bias_w3 + bias_opes + bias_k
dataset = mpair.cvs.committor.compute_committor_weights(
    dataset, bias, 1 / (0.008314 * 340)
)

mpair.data.save_dataset(dataset, 'dataset.pt')

dataset = mpair.data.load_dataset('dataset.pt')
dataset = dataset[20002:]

masses = mpair.data.atomic.get_masses(dataset.mapping_names['atom_names'])

cv = mpair.cvs.PairCommittor(
    atomic_masses=masses,
    mapping_names=dataset.mapping_names,
    model_name='PairFormerModel',
    model_options={
        'n_layers': 2,
        'cutoff': 24.0,  # A
        'n_bases': 16,
        'n_polynomials': 5,
        'n_embedding_pair': 8,
        'triangle_attention': 'torch',
        'triangle_multiplicative': None,
    },
    extra_loss_options={
        'alpha': 1.0,
        'gamma': 10000.0,
        'sigmoid_p': 3.0,
        'n_bootstrap': 5,
        'exclude_boundary_in_loss_v': True,
    },
    optimizer_options={
        'optimizer': {'lr': 5E-5, 'weight_decay': 5E-7},
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR,
            'gamma': 0.99997
        }
    }
)

cv.load_state_dict(torch.load('./model.pt'))
cv = cv.to('cuda').eval()

cv_values = mpair.explain.utils.get_dataset_cv_values(
    cv,
    dataset,
    device='cuda',
    batch_size=10000,
)[:, 0].flatten()
index = np.arange(len(dataset))[((cv_values > -0.5) & (cv_values < 0.5))]
dataset = dataset[index]

s = mpair.explain.pair_sensitivity(
    cv,
    dataset,
    device='cuda',
    batch_size=3000,
    component=1
)['sensitivities']
np.savetxt('sensitivity.dat', s)

p = np.array(np.where(s > 0.0005)).T
p = p[p[:, 1] > p[:, 0]]
_, lengths = mpair.explain.utils.get_dataset_cv_gradients_pair(
    cv,
    dataset,
    device='cuda',
    batch_size=3000,
    return_lengths=True,
)
lengths = lengths[:, p[:, 0], p[:, 1]]

atom_indices_plumed = md.load(
    '../../data/gmx/plumed_topo.pdb'
).top.select(
    'protein and backbone and not type H'
) + 1

d = pd.DataFrame()
d['time'] = np.arange(len(lengths))
d['z'] = cv_values[index]
for i, pp in enumerate(p):
    d[
        f'd_{atom_indices_plumed[pp[0]]}_{atom_indices_plumed[pp[1]]}'
    ] = lengths[:, i]
d.set_index('time')
plumed.write_pandas(d, 'analysis.dat')
