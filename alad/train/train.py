import torch
import mlcolvar.pair as mpair
from mlcolvar.utils import io as ffio

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlcolvar.utils.trainer import MetricsCallback

mpair.utils.torch_tools.set_default_dtype('float64')


dataset = mpair.utils.io.create_dataset_from_trajectories(
    trajectories=[
        '../iter_0/run_A/traj.dcd',
        '../iter_0/run_B/traj.dcd',
        '../iter_1/traj_all/traj.last.dcd',
    ],
    top=[
        '../data/plumed_topo.pdb',
        '../data/plumed_topo.pdb',
        '../data/plumed_topo.pdb',
    ],
    cutoff=-1,  # should be negative
    create_labels=True,
    system_selection='not type H',
    node_embeddings=['atom_names', 'residue_names'],
    no_pbc=True,
)

ff_dataset, ff_dataframe = ffio.create_dataset_from_files(
    file_names=[
        '../iter_0/run_A/colvar',
        '../iter_0/run_B/colvar',
        '../iter_1/traj_all/colvar.last',
    ],
    return_dataframe=True
)

dataframe = ff_dataframe.fillna({'opes.bias': 0, 'pf.kbias': 0})
try:
    bias_opes = torch.tensor(dataframe['opes.bias'].values)
except KeyError:
    bias_opes = torch.zeros(len(dataframe))
try:
    bias_k = torch.tensor(dataframe['pf.kbias'].values)
except KeyError:
    bias_k = torch.zeros(len(dataframe))
bias = bias_opes + bias_opes

dataset = mpair.cvs.committor.compute_committor_weights(dataset, bias, 1/2.49)
masses = mpair.data.atomic.get_masses(dataset.mapping_names['atom_names'])

datamodule = mpair.data.PairDataModule(
    dataset,
    batch_size=5000,
    lengths=(0.95, 0.05),
    random_split=True,
    shuffle=False,
)


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


metrics = MetricsCallback()
early_stopping = EarlyStopping(
    monitor='train_loss',
    patience=100,
    min_delta=1E-5,
    stopping_threshold=-16.4,
)
early_stop_val = EarlyStopping(
    monitor='valid_loss',
    min_delta=1E-5,
    patience=300,
    mode='min',
)
checkpointing = ModelCheckpoint(
    monitor='train_loss',
    dirpath='./checkpoints',
    filename='{epoch}',
    every_n_epochs=25,
    save_top_k=50,
)

trainer = Trainer(
    callbacks=[metrics, early_stopping, checkpointing],
    logger=None,
    accelerator='gpu',
    max_epochs=15000
)

trainer.fit(cv, datamodule)
cv = cv.eval()

torch.save(cv.state_dict(), 'model.pt')
