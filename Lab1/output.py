import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, chi2


def scatter_diagram(seq):
    plt.scatter(seq[:-1], seq[1:])
    plt.show()


def moment_expected_value(eps, seq):
    statistic = np.abs(seq.mean() - 0.5) * (12 * seq.size) ** 0.5
    bound = norm.ppf(1 - eps / 2)

    return 0 if statistic < bound else 1, statistic, bound, eps


def moment_dispersion(eps, seq):
    statistic = np.abs(seq.var(ddof=1) - 1 / 12) * (seq.size - 1) / seq.size / \
                (0.0056 / seq.size + 0.0028 / seq.size ** 2 - 0.0083 / seq.size ** 3) ** 0.5
    bound = norm.ppf(1 - eps / 2)

    return 0 if statistic < bound else 1, statistic, bound, eps


def covariance(eps, seq, count):
    c, r = np.array([2 ** 0.5, *([1] * count)]), np.array([1 / 12, *([0] * count)])

    m = seq.mean()
    r_appr = np.array(
        [1 / (seq.size - j - 1) * (seq[:seq.size - j] * seq[j:]).sum() - seq.size / (seq.size - 1) * m ** 2 for j in range(count + 1)])

    statistic = np.abs(r_appr - r)
    bound = c * norm.ppf(1 - eps / 2) / 12 / (seq.size - 1) ** 0.5
    hyp = (~(statistic < bound)).astype(int)

    return [(hyp[i], statistic[i], bound[i], eps) for i in range(count + 1)]


def chi_square(eps, seq):
    count = 1000

    p = np.full(count, 1 / count)
    n = np.full(count, 0)

    for el in seq:
        for i in range(count):
            if el < (i + 1) / count:
                n[i] += 1
                break

    statistic = seq.size * ((n / seq.size - p) ** 2 / p).sum()
    bound = chi2.ppf(1 - eps, count - 1)

    return 0 if statistic < bound else 1, statistic, bound, eps


def concat(arr):
    return ','.join(map(lambda s: f'{s:.10f}', arr))


def output(gen, n, output_file):
    seq = np.array([gen.generate() for _ in range(n)]) / gen.m

    scatter_diagram(seq)

    test_seq1 = np.array([gen.generate() for _ in range(10 ** 4)]) / gen.m
    test_seq2 = np.array([gen.generate() for _ in range(10 ** 5)]) / gen.m

    eps = 0.05
    covar_count = 10

    with open(output_file, 'w') as lc_output:
        lc_output.writelines([
            f'{seq.mean()}\n',
            concat(moment_expected_value(eps, test_seq1)), '\n',
            concat(moment_expected_value(eps, test_seq2)), '\n',
            f'{seq.var(ddof=1)}\n',
            concat(moment_dispersion(eps, test_seq1)), '\n',
            concat(moment_dispersion(eps, test_seq2)), '\n',
            '\n'.join([concat(stat) for stat in covariance(eps, test_seq1, covar_count)]), '\n',
            '\n'.join([concat(stat) for stat in covariance(eps, test_seq2, covar_count)]), '\n',
            concat(chi_square(eps, test_seq1)), '\n',
            concat(chi_square(eps, test_seq2)), '\n'
        ])
