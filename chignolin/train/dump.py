import torch
import mlcolvar.pairformer as mpair

mpair.utils.torch_tools.set_default_dtype('float32')

dataset = mpair.utils.io.create_dataset_from_trajectories(
    trajectories=['../data/gmx/folded.gro'],
    top=['../data/gmx/folded.gro'],
    cutoff=-1,  # should be negative
    create_labels=True,
    system_selection='protein and backbone and not type H',
    node_embeddings=['atom_names', 'residue_names'],
    no_pbc=True,
)
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

mpair.utils.export.export(
    cv,
    dataset[0],
    file_name='model.pt2',
)
mpair.utils.export.export(
    cv,
    dataset[0],
    file_name='model.k_bias.large.pt2',
    k_bias_options={
        'calculate_k_bias': True,
        'kb_epsilon': 1E-7,
        'kb_lambda': -3.4,
        'kb_weighted': True,
        'kb_truncated': False,
        'kb_truncated_cos': True,
    },
)
