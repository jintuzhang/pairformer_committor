import torch
import mlcolvar.pairformer as mpair
from mlcolvar.utils import io as ffio

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlcolvar.utils.trainer import MetricsCallback

mpair.utils.torch_tools.set_default_dtype('float32')


dataset = mpair.utils.io.create_dataset_from_trajectories(
    trajectories=[
        [
            '../../iter_0/run_A/G2/traj.dcd',
            '../../iter_0/run_A/G5/traj.dcd',
            '../../iter_0/run_A/G6/traj.dcd',
        ],
        [
            '../../iter_0/run_B/G2/traj.dcd',
            '../../iter_0/run_B/G5/traj.dcd',
            '../../iter_0/run_B/G6/traj.dcd',
        ],
        '../../iter_0/run_biased/G2/traj.last.dcd',
        '../../iter_0/run_biased/G5/traj.last.dcd',
        '../../iter_0/run_biased/G6/traj.last.dcd',
        '../../iter_1/run_opes_kbias/G2_gmx/traj.last.dcd',
        '../../iter_1/run_opes_kbias/G5_gmx/traj.last.dcd',
        '../../iter_1/run_opes_kbias/G6_gmx/traj.last.dcd',
    ],
    top=[
        [
            '../../data/G2/plumed_topo.pdb',
            '../../data/G5/plumed_topo.pdb',
            '../../data/G6/plumed_topo.pdb',
        ],
        [
            '../../data/G2/plumed_topo.pdb',
            '../../data/G5/plumed_topo.pdb',
            '../../data/G6/plumed_topo.pdb',
        ],
        '../../data/G2/plumed_topo.pdb',
        '../../data/G5/plumed_topo.pdb',
        '../../data/G6/plumed_topo.pdb',
        '../../data/G2/plumed_topo.pdb',
        '../../data/G5/plumed_topo.pdb',
        '../../data/G6/plumed_topo.pdb',
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

ff_dataset, ff_dataframe = ffio.create_dataset_from_files(
    file_names=[
        '../../iter_0/run_A/G2/colvar',
        '../../iter_0/run_A/G5/colvar',
        '../../iter_0/run_A/G6/colvar',
        '../../iter_0/run_B/G2/colvar',
        '../../iter_0/run_B/G5/colvar',
        '../../iter_0/run_B/G6/colvar',
        '../../iter_0/run_biased/G2/colvar.last',
        '../../iter_0/run_biased/G5/colvar.last',
        '../../iter_0/run_biased/G6/colvar.last',
        '../../iter_1/run_opes_kbias/G2_gmx/colvar.last',
        '../../iter_1/run_opes_kbias/G5_gmx/colvar.last',
        '../../iter_1/run_opes_kbias/G6_gmx/colvar.last',
    ],
    return_dataframe=True,
)
dataframe = ff_dataframe.fillna(
    {'w_1.bias': 0, 'w_f.bias': 0, 'opes.bias': 0, 'v_k.bias': 0}
)
bias_w1 = torch.tensor(dataframe['w_1.bias'].values)
bias_wf = torch.tensor(dataframe['w_f.bias'].values)
bias_opes = torch.tensor(dataframe['opes.bias'].values)
bias_k = torch.tensor(dataframe['v_k.bias'].values)
bias = bias_w1 + bias_wf + bias_opes + bias_k
dataset = mpair.cvs.committor.compute_committor_weights(
    dataset, bias, 1 / (0.008314 * 300)
)

mpair.data.save_dataset(dataset, 'dataset.pt')

dataset = mpair.data.load_dataset('dataset.pt')

datamodule = mpair.data.PairDataModule(
    dataset,
    batch_size=3000,
    lengths=(0.95, 0.05),
    random_split=True,
    shuffle=False,
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


metrics = MetricsCallback()
early_stopping = EarlyStopping(
    monitor='train_loss', patience=100, min_delta=1E-4, stopping_threshold=-20
)
early_stop_val = EarlyStopping(
    monitor='valid_loss', min_delta=1E-5, patience=300, mode='min'
)
checkpointing = ModelCheckpoint(
    monitor='train_loss',
    dirpath='./checkpoints',
    filename='{epoch}',
    every_n_epochs=5,
    save_top_k=200,
)

trainer = Trainer(
    callbacks=[metrics, early_stopping, checkpointing],
    logger=None,
    accelerator='gpu',
    max_epochs=2000,
)

trainer.fit(cv, datamodule)
cv = cv.eval()

torch.save(cv.state_dict(), 'model.pt')
