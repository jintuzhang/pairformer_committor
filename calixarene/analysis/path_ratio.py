import plumed
import numpy as np

for i in [2, 5, 6]:
    results = []
    for j in [1, 2, 3]:
        d = plumed.read_as_pandas(f'G{i}/{j}/colvar')[100:]
        w = np.exp(d['pf.kbias'] / 2.5)
        w = w / sum(w)
        w_new = np.exp(-d['pf.kbias'] / 2.5 / 1.2)
        ratio = (
            ((d['V2'] < 0.5) * w * w_new).sum()
            / ((d['V2'] > 1) * w * w_new).sum()
        )
        results.append(ratio)
    print(
        'G{:d} (dry path): {:.3f} +- {:.3f}'.format(
            i, np.mean(results), np.std(results)
        )
    )
