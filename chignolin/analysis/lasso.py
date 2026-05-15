import torch
import numpy as np

from cycler import cycler
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import mlcolvar.graph as mg
from mlcolvar.explain.lasso import lasso_regression
from mlcolvar.utils.io import create_dataset_from_files

dataset_ff, df = create_dataset_from_files(
    './analysis.dat', return_dataframe=True, filter_args={'regex': 'd.*'}
)
dataset_ff['target'] = torch.tensor(df['z'].values)

regressor, feats, coeffs = lasso_regression(dataset_ff, plot=True)

num_features = [5, 7, 10, 15, 20]
num_features_path_ = np.count_nonzero(
    np.abs(regressor.coefs_paths_.T) > 1E-4, axis=1
)

print(num_features_path_)

selected_alphas = []
for num_feat in num_features:
    id = np.argwhere(num_features_path_ == num_feat).max()
    alpha = regressor.alphas_[id]
    selected_alphas.append(alpha)
    print(f'num_feat={num_feat} --> alpha={alpha:.2e}')

with torch.no_grad():
    X = dataset_ff['data'].numpy()
    y = dataset_ff['target'].numpy()

X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

# test different values of alpha
fig, ax = plt.subplots()
ax.set_prop_cycle(cycler(color=plt.get_cmap('Dark2').colors))

for alpha in selected_alphas:
    regressor, feats, coeffs = lasso_regression(
        dataset_ff,
        alphas=[alpha],
        print_info=True,
        plot=False,
    )
    y_pred = regressor.predict(X)
    ax.scatter(
        y_pred[::5],
        y[::5],
        s=5,
        label=f'n_features={len(coeffs)} (alpha={alpha:.2e})',
        alpha=0.5
    )
    # print equation
    equation = "y="
    for f, c in zip(feats, coeffs):
        equation += f"+{c:.3f}*{f} " if c > 0 else f"{c:.3f}*{f} "

    print(equation[:-1])
    print('\n')

ax.set_xlabel('Linear model')
ax.set_ylabel('MLCV')
ax.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linewidth=2,
    color='lightgrey',
    zorder=0,
    linestyle='dashed'
)
ax.legend(frameon=False)
plt.show()
