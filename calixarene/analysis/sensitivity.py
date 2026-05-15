import torch
import numpy as np

import mlcolvar.pairformer as mpair
from mlcolvar.utils import io as ffio

mpair.utils.torch_tools.set_default_dtype('float32')


dataset = mpair.utils.io.create_dataset_from_trajectories(
    trajectories=[
        '../data/G2/plumed_topo.pdb',
        '../data/G5/plumed_topo.pdb',
        '../data/G6/plumed_topo.pdb',
    ],
    top=[
        '../data/G2/plumed_topo.pdb',
        '../data/G5/plumed_topo.pdb',
        '../data/G6/plumed_topo.pdb',
    ],
    cutoff=10.0,  # Ang
    system_selection=(
        '(resname G2 G5 G6 and not type H) or '
        + '(resname OCB and name C1 C10 C16 C28 C8 C18 C26 C32 C46 C52 C58 C64)'
    ),
    environment_selection='type O and resname HOH',
    center_selections=[
        'resname OCB and name C1 C10 C16 C28',
        'resname OCB and name C8 C18 C26 C32',
        'resname OCB and name C46 C52 C58 C64',
    ],
    n_atoms_padded_environment=180,
    node_embeddings=['atom_names', 'residue_names'],
    n_workers=1,
)

masses = mpair.data.atomic.get_masses(dataset.mapping_names['atom_names'])

cv = mpair.cvs.PairCommittor(
    mapping_names=dataset.mapping_names,
    atomic_masses=masses,
    model_options={
        'n_layers': 2,
        'cutoff': 20.0,  # Ang, for basis set
        'n_bases': 16,
        'n_polynomials': 5,
        'n_embedding_pair': 8,
        'triangle_attention': 'torch',
        'triangle_multiplicative': None,
        'pair_transition': True,
        'cn_options': {  # RATIONAL D_0=0.0 D_MAX=1.0 R_0=0.25 NN=2 MM=6
            'n': 2,
            'm': 6,
            'r_0': 2.5,  # Ang
            'd_0': 0.0,  # Ang
            'd_max': dataset.cutoff,
            'n_centers': dataset[0]['centers'].shape[1],
        },
    },
    extra_loss_options={
        'alpha': 1.0,
        'gamma': 10000.0,
        'sigmoid_p': 3.0,
        'n_bootstrap': 50,
        'exclude_boundary_in_loss_v': True,
    },
    optimizer_options={
        'optimizer': {'lr': 1E-5, 'weight_decay': 1E-6},
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR,
            'gamma': 0.999995
        }
    }
)

cv.load_state_dict(torch.load('../train/model.pt'))
cv = cv.eval().to('cuda')


dataset = mpair.utils.io.create_dataset_from_trajectories(
    trajectories=[
        'G2_whole.xtc',
        '../data/G5/plumed_topo.pdb',
        '../data/G6/plumed_topo.pdb',
    ],
    top=[
        '../data/G2/plumed_topo.pdb',
        '../data/G5/plumed_topo.pdb',
        '../data/G6/plumed_topo.pdb',
    ],
    cutoff=10.0,  # Ang
    system_selection=(
        '(resname G2 G5 G6 and not type H) or '
        + '(resname OCB and name C1 C10 C16 C28 C8 C18 C26 C32 C46 C52 C58 C64)'
    ),
    environment_selection='type O and resname HOH',
    center_selections=[
        'resname OCB and name C1 C10 C16 C28',
        'resname OCB and name C8 C18 C26 C32',
        'resname OCB and name C46 C52 C58 C64',
    ],
    n_atoms_padded_environment=180,
    node_embeddings=['atom_names', 'residue_names'],
    n_workers=64,
)
dataset = dataset[:-2]
ff_dataset, ff_dataframe = ffio.create_dataset_from_files(
    file_names=['../run_biased_mlcv/run_opes_only/G2/colvar'],
    return_dataframe=True
)
dataframe = ff_dataframe.fillna(
    {'w_1.bias': 0, 'w_f.bias': 0, 'opes.bias': 0, 'w_cap.bias': 0}
)
bias_w1 = torch.tensor(dataframe['w_1.bias'].values)
bias_wf = torch.tensor(dataframe['w_f.bias'].values)
bias_opes = torch.tensor(dataframe['opes.bias'].values)
bias_c = torch.tensor(dataframe['w_cap.bias'].values)
bias = bias_w1 + bias_wf + bias_opes + bias_c
dataset = mpair.cvs.committor.compute_committor_weights(
    dataset, bias, 1 / (0.008314 * 300)
)
dataset = dataset[2000:]

s = mpair.explain.pair_sensitivity(
    cv,
    dataset,
    device='cuda',
    batch_size=3000,
)['sensitivities']
np.savetxt('sensitivity.g2.dat', s)


dataset = mpair.utils.io.create_dataset_from_trajectories(
    trajectories=[
        '../data/G2/plumed_topo.pdb',
        'G5_whole.xtc',
        '../data/G6/plumed_topo.pdb',
    ],
    top=[
        '../data/G2/plumed_topo.pdb',
        '../data/G5/plumed_topo.pdb',
        '../data/G6/plumed_topo.pdb',
    ],
    cutoff=10.0,  # Ang
    system_selection=(
        '(resname G2 G5 G6 and not type H) or '
        + '(resname OCB and name C1 C10 C16 C28 C8 C18 C26 C32 C46 C52 C58 C64)'
    ),
    environment_selection='type O and resname HOH',
    center_selections=[
        'resname OCB and name C1 C10 C16 C28',
        'resname OCB and name C8 C18 C26 C32',
        'resname OCB and name C46 C52 C58 C64',
    ],
    n_atoms_padded_environment=180,
    node_embeddings=['atom_names', 'residue_names'],
    n_workers=64,
)
dataset = dataset[1:-1]
ff_dataset, ff_dataframe = ffio.create_dataset_from_files(
    file_names=['../run_biased_mlcv/run_opes_only/G5/colvar'],
    return_dataframe=True
)
dataframe = ff_dataframe.fillna(
    {'w_1.bias': 0, 'w_f.bias': 0, 'opes.bias': 0, 'w_cap.bias': 0}
)
bias_w1 = torch.tensor(dataframe['w_1.bias'].values)
bias_wf = torch.tensor(dataframe['w_f.bias'].values)
bias_opes = torch.tensor(dataframe['opes.bias'].values)
bias_c = torch.tensor(dataframe['w_cap.bias'].values)
bias = bias_w1 + bias_wf + bias_opes + bias_c
dataset = mpair.cvs.committor.compute_committor_weights(
    dataset, bias, 1 / (0.008314 * 300)
)

s = mpair.explain.pair_sensitivity(
    cv,
    dataset,
    device='cuda',
    batch_size=3000,
)['sensitivities']
np.savetxt('sensitivity.g5.dat', s)


dataset = mpair.utils.io.create_dataset_from_trajectories(
    trajectories=[
        '../data/G2/plumed_topo.pdb',
        '../data/G5/plumed_topo.pdb',
        'G6_whole.xtc',
    ],
    top=[
        '../data/G2/plumed_topo.pdb',
        '../data/G5/plumed_topo.pdb',
        '../data/G6/plumed_topo.pdb',
    ],
    cutoff=10.0,  # Ang
    system_selection=(
        '(resname G2 G5 G6 and not type H) or '
        + '(resname OCB and name C1 C10 C16 C28 C8 C18 C26 C32 C46 C52 C58 C64)'
    ),
    environment_selection='type O and resname HOH',
    center_selections=[
        'resname OCB and name C1 C10 C16 C28',
        'resname OCB and name C8 C18 C26 C32',
        'resname OCB and name C46 C52 C58 C64',
    ],
    n_atoms_padded_environment=180,
    node_embeddings=['atom_names', 'residue_names'],
    n_workers=64,
)
dataset = dataset[2:]
ff_dataset, ff_dataframe = ffio.create_dataset_from_files(
    file_names=['../run_biased_mlcv/run_opes_only/G6/colvar'],
    return_dataframe=True
)
dataframe = ff_dataframe.fillna(
    {'w_1.bias': 0, 'w_f.bias': 0, 'opes.bias': 0, 'w_cap.bias': 0}
)
bias_w1 = torch.tensor(dataframe['w_1.bias'].values)
bias_wf = torch.tensor(dataframe['w_f.bias'].values)
bias_opes = torch.tensor(dataframe['opes.bias'].values)
bias_c = torch.tensor(dataframe['w_cap.bias'].values)
bias = bias_w1 + bias_wf + bias_opes + bias_c
dataset = mpair.cvs.committor.compute_committor_weights(
    dataset, bias, 1 / (0.008314 * 300)
)

s = mpair.explain.pair_sensitivity(
    cv,
    dataset,
    device='cuda',
    batch_size=3000,
)['sensitivities']
np.savetxt('sensitivity.g6.dat', s)
