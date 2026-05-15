import torch
import mlcolvar.pairformer as mpair

mpair.utils.torch_tools.set_default_dtype('float64')


dataset = mpair.utils.io.create_dataset_from_trajectories(
    trajectories=['../data/plumed_topo.pdb'],
    top=['../data/plumed_topo.pdb'],
    create_labels=True,
    system_selection='not type H',
    node_embeddings=['atom_names', 'residue_names'],
    no_pbc=True,
)
masses = mpair.data.atomic.get_masses(dataset.mapping_names['atom_names'])


cv = mpair.cvs.PairCommittor(
    mapping_names=dataset.mapping_names,
    atomic_masses=masses,
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

cv.load_state_dict(torch.load('./model.pt'))
cv = cv.eval()

mpair.utils.export.export(
    cv,
    dataset[0],
    file_name='model.pt2',
)

mpair.utils.export.export(
    cv,
    dataset[0],
    file_name='model.k_bias.pt2',
    k_bias_options={
        'calculate_k_bias': True,
        'kb_epsilon': 1E-10,
        'kb_lambda': -3.0,
        'kb_weighted': True,
        'kb_truncated': False,
    },
)
