import torch
import numpy as np
import mlcolvar.pairformer as mpair

mpair.utils.torch_tools.set_default_dtype('float32')


dataset = mpair.utils.io.create_dataset_from_trajectories(
    trajectories=['../run_biased_mlcv/k_bias/traj.dcd'],
    top=['../data/plumed_topo.pdb'],
    create_labels=True,
    system_selection='not type H',
    node_embeddings=['atom_names', 'residue_names'],
    no_pbc=True,
)

masses = mpair.data.atomic.get_masses(dataset.mapping_names['atom_names'])

cv = mpair.cvs.PairCommittor(
    atomic_masses=masses,
    mapping_names=dataset.mapping_names,
    model_name='PairFormerModel',
    model_options={
        'n_layers': 3,
        'cutoff': 10.0,  # A
        'triangle_attention': 'torch',
        'triangle_multiplicative': None,
    },
    extra_loss_options={
        'alpha': 1.0,
        'gamma': 10000.0,
        'delta_f': -9 / 2.49,
        'sigmoid_p': 3.0,
        'n_bootstrap': 20,
        'exclude_boundary_in_loss_v': True,
    },
    optimizer_options={
        'optimizer': {'lr': 1E-4, 'weight_decay': 1E-6},
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR,
            'gamma': 0.99993
        }
    }
)

cv.load_state_dict(torch.load('../train/model.pt'))
cv = cv.to('cuda').eval()

s = mpair.explain.pair_sensitivity(
    cv,
    dataset,
    device='cuda',
    batch_size=3000,
    component=1
)['sensitivities']

np.savetxt('sensitivity.dat', s)
