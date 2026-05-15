import plumed
import numpy as np

for i in [2, 5, 6]:
    results = []
    for j in [1, 2, 3]:
        d = plumed.read_as_pandas(f'G{i}_gmx_t2_{j}/colvar')[0:]
        ratio = (d['V2'] < 0.3).sum() / (d['V2'] > 0.5).sum()
        results.append(ratio)
    print(
        'G{:d} (dry path): {:.3f} +- {:.3f}'.format(
            i, np.mean(results), np.std(results)
        )
    )
