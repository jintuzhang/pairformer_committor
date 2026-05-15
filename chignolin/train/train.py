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
        '../run_unbiased/run_A/traj.dcd',
        '../run_unbiased/run_B/traj.dcd',
        '../data/chignolin_traindata_pkang/iter_4_clean/0/traj.dcd',
        '../data/chignolin_traindata_pkang/iter_4_clean/1/traj.dcd',
        '../data/chignolin_traindata_pkang/iter_4_clean/2/traj.dcd',
        '../data/chignolin_traindata_pkang/iter_4_clean/3/traj.dcd',
    ],
    top=[
        '../data/gmx/folded.gro',
        '../data/gmx/unfolded.gro',
        '../data/gmx/rmsd_ref_ca.pdb',
        '../data/gmx/rmsd_ref_ca.pdb',
        '../data/gmx/rmsd_ref_ca.pdb',
        '../data/gmx/rmsd_ref_ca.pdb',
    ],
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
        '../data/chignolin_traindata_pkang/iter_4_clean/0/colvar',
        '../data/chignolin_traindata_pkang/iter_4_clean/1/colvar',
        '../data/chignolin_traindata_pkang/iter_4_clean/2/colvar',
        '../data/chignolin_traindata_pkang/iter_4_clean/3/colvar',
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

# dataset = mpair.data.load_dataset('dataset.pt')

masses = mpair.data.atomic.get_masses(dataset.mapping_names['atom_names'])

datamodule = mpair.data.PairDataModule(
    dataset,
    batch_size=3000,
    lengths=(0.95, 0.05),
    random_split=True,
    shuffle=False,
)


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
        'optimizer': {'lr': 5E-5, 'weight_decay': 2E-6},
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR,
            'gamma': 0.99997
        }
    }
)


metrics = MetricsCallback()
early_stopping = EarlyStopping(
    monitor='train_loss', patience=100, min_delta=1E-5, stopping_threshold=-16
)
early_stop_val = EarlyStopping(
    monitor='valid_loss', min_delta=1E-5, patience=300, mode='min'
)
checkpointing = ModelCheckpoint(
    monitor='train_loss',
    dirpath='./checkpoints',
    filename='{epoch}',
    every_n_epochs=5,
    save_top_k=500,
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
