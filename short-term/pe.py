import numpy as np
from math import factorial

def permutation_entropy(time_series, order=3, delay=1, normalize=False):
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))

    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')


    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)


    _, c = np.unique(hashval, return_counts=True)

    p = np.true_divide(c, c.sum())

    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe



def _embed(x, order=3, delay=1):

    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

