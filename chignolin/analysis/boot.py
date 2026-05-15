import numpy as np

n_1 = [
    [4, 8, 5],
    [0, 4, 1],
    [3, 1, 5],
    [4, 4, 5],
    [3, 3, 6],
    [5, 2, 2],
    [4, 4, 7],
    [7, 4, 4],
    [10, 8, 7],
    [2, 8, 6],
    [5, 3, 4],
    [13, 16, 12],
    [12, 8, 9],
    [4, 7, 8],
    [14, 14, 15],
    [7, 2, 7],
    [11, 13, 14],
    [5, 6, 8],
]
n_0 = [
    [11, 10, 11],
    [18, 14, 15],
    [13, 18, 13],
    [14, 12, 15],
    [13, 17, 12],
    [12, 15, 15],
    [12, 9, 10],
    [11, 9, 15],
    [2, 6, 6],
    [15, 7, 8],
    [10, 12, 8],
    [2, 2, 4],
    [0, 2, 4],
    [7, 7, 9],
    [4, 2, 3],
    [6, 5, 4],
    [2, 1, 3],
    [6, 4, 5],
]

group_obs = []
for i in range(18):
    group_obs.append(
        [np.array([1] * n_1[i][j] + [0] * n_0[i][j]) for j in range(3)]
    )
group_missing = 20 - np.array(n_0) + np.array(n_1)
group_pi = [[np.mean(obs) for obs in group_obs[i]] for i in range(18)]

B = 10000

all_group_means = [np.zeros((B, 3)) for i in range(18)]

for k in range(18):
    for b in range(B):
        for i, (obs, n_missing, pi) in enumerate(
            zip(group_obs[k], group_missing[k], group_pi[k])
        ):
            if n_missing > 0:
                missing_values = np.random.binomial(1, pi, n_missing)
            else:
                missing_values = np.array([])

            full_data = np.concatenate([obs, missing_values])
            all_group_means[k][b, i] = np.mean(full_data)

    group_mean_est = np.mean(all_group_means[k], axis=0)
    group_se_est = np.std(all_group_means[k], axis=0, ddof=1)

    overall_mean = np.mean(group_mean_est)
    overall_se = np.std(np.mean(all_group_means, axis=1), ddof=1)
    print(f"{k:2d}: mean: {overall_mean:.3f}, std: {overall_se:.3f}")
