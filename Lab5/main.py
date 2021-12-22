import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt


rand = lambda: np.random.rand(1)[0]


def err(est, sol):
    return np.sum((est - sol) ** 2)


# обработка вероятностей цепей Маркова
def get_states_distribution(state_probs):
    state_probs = np.array(state_probs)
    j = np.argsort(state_probs)
    if state_probs.ndim > 1:
        i = np.mgrid[0:state_probs.shape[0], 0:state_probs.shape[1]][0]
        return {'states': j, 'distribution': np.cumsum(state_probs[i, j], axis=1)}
    return {'states': j, 'distribution': np.cumsum(state_probs[j])}


# определить промежуток для св
def get_bin(state_distribution):
    return np.digitize(rand(), state_distribution)


# генерация цепей Маркова
def chain_generator(pi, P):
    distribution = get_states_distribution(pi)
    state = distribution['states'][get_bin(distribution['distribution'])]
    distribution = get_states_distribution(P)

    while True:
        yield state
        state = distribution['states'][state][get_bin(distribution['distribution'][state])]


A_init = np.array([[1.2, -0.3, 0.4],
                   [0.4, 0.7, -0.2],
                   [0.2, -0.3, 0.9]])
f = np.array([-4, 2, 0])
n = f.size

A = np.eye(n) - A_init
pi = np.ones(n) / n
P = np.ones((n, n)) / n
G = np.nan_to_num(A / P, posinf=0)


# решение
def solve(L, N):
    solution = []
    h, r = np.zeros(n), range(1, N + 1)

    for i in range(n):
        h[i - 1 if i >= 1 else 0] = 0
        h[i] = 1
        var_sum = 0

        for _ in range(L):
            generator = chain_generator(pi, P)
            chain = np.array([next(generator) for _ in range(N + 1)])

            Q = list(accumulate(r, lambda Q_prev, i: Q_prev * G[chain[i - 1], chain[i]],
                                initial=h[chain[0]] / pi[chain[0] if pi[chain[0]] > 0 else 0]))
            var_sum += np.sum(np.array(Q) * f[chain])

        solution.append(var_sum / L)

    return np.array(solution)


solution = np.linalg.solve(A_init, f)
estimated_solution = solve(10000, 100)

print(f'solution: {solution}', f'true solution: {estimated_solution}',
      f'err: {err(estimated_solution, solution)}', sep='\n')

N = np.arange(10, 101, 10)
errs = np.array([err(solve(10000, n), solution) for n in N])
plt.plot(N, errs)
plt.show()

L = np.arange(1000, 10001, 1000)
errs = np.array([err(solve(l, 100), solution) for l in L])
plt.plot(L, errs)
plt.show()
