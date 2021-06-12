"""OPTIMAL EXPERIMENTAL DESIGN"""

import numpy as np

from wzk import random_subset, mp_wrapper
from wzk.mpl import new_fig, subplot_grid


def covariance(jac, prior_sigma):
    s = jac.shape
    jac2 = jac.reshape(s[:-3] + (s[-3]*s[-2], s[-1]))
    return np.linalg.inv((np.swapaxes(jac2, -1, -2) @ jac2 + np.eye(s[-1]) * prior_sigma))


def covariance_wrapper(jac, prior_sigma):
    def __covariance(idx):
        return covariance(jac=jac[idx], prior_sigma=prior_sigma)

    return __covariance


def task_a_optimality_wrapper(jac_calset, prior_sigma, jac_test=None, sum_jac_jac=None, verbose=0):
    """
    Simplification as used by Carrillo 2013.
    Results are the same, just much faster to compute
    """

    if sum_jac_jac is None:
        sum_jac_jac = (jac_test.transpose(0, 2, 1) @ jac_test).mean(axis=0)
    # Is equivalent
    #  s = jac_test.shape
    #  jac_test2 = jac_test.reshape(s[:-3] + (s[-3]*s[-2], s[-1]))
    #  sum_jac_jac2 = (np.swapaxes(jac_test2, -1, -2) @ jac_test2)  / len(jac_test)
    #  np.allclose(sum_jac_jac, sum_jac_jac2)

    if verbose > 0:
        print('sum(J.T @ J)', sum_jac_jac)

    cov_wrapper = covariance_wrapper(jac=jac_calset, prior_sigma=prior_sigma)

    def task_a_optimality(idx):
        return np.trace(cov_wrapper(idx=idx) @ sum_jac_jac, axis1=-2, axis2=-1)

    return task_a_optimality


def test_task_a_optimality_simplification():

    n_test = 1000
    n_calset = 100
    n_cal = 10
    n = 12
    n_y = 10
    n_p = 20
    prior_sigma = 0.17

    jac_test = np.random.random((n_test, n_y, n_p))
    jac_calset = np.random.random((n_calset, n_y, n_p))
    idx = random_subset(n=n_calset, k=n_cal, m=n)

    def task_a_optimality_explicit(_idx):
        """use explicit formula"""
        cov_wrapper = covariance_wrapper(jac=jac_calset, prior_sigma=prior_sigma)
        cov = cov_wrapper(idx=idx)
        b = 0
        for jt in jac_test:
            b += np.trace(jt @ cov @ jt.T, axis1=-2, axis2=-1)
        b /= len(jac_test)
        return b

    task_a_optimality_simplified = task_a_optimality_wrapper(jac_calset=jac_calset, jac_test=jac_test,
                                                             prior_sigma=prior_sigma)

    aa = task_a_optimality_explicit(idx)
    bb = task_a_optimality_simplified(idx)
    assert np.allclose(aa, bb)


def d_optimality_wrapper(jac):

    def d_optimality(idx):
        jac_i = jac[idx]
        # opt_d = (1 / np.linalg.det(np.swapaxes(jac_i, -1, -2) @ jac_i)).mean(axis=-1)

        s = jac_i.shape
        jac2 = jac_i.reshape(s[:-3] + (s[-3] * s[-2], s[-1]))
        opt_d = 1 / np.linalg.det((np.swapaxes(jac2, -1, -2) @ jac2))

        # opt_d = np.linalg.det(covariance(jac_i))
        # print("Attention check out the difference here!")
        return opt_d

    return d_optimality


def detmax(fun, x0=None, n=100, k=30, excursion=10, method='add->remove', max_loop=3,
           verbose=0):
    """
    method:  'add->remove' TODO citation
             'remove->add'
    """

    improvement_threshold = 1e-2
    if x0 is None:
        # x0 = greedy(n=n, k=k, verbose=0)
        x0 = random_subset(n=n, k=k, m=1, dtype=np.int16)[0]

    def __add(x, nn):
        x = idx_times_all(idx=x, n=nn)
        oo = fun(x)
        oo[x[0, :-1]] = np.inf
        idx_min = np.argmin(oo)
        # idx_min = np.random.choice(np.argsort(oo)[:nn//10])
        oo = oo[idx_min]
        x = x[idx_min]
        return x, oo

    def remove(x, exc):
        oo = None
        for _ in range(1, exc+1):
            x = np.repeat([x], repeats=len(x), axis=0)
            x = x[np.logical_not(np.eye(len(x), dtype=bool))].reshape(len(x), len(x)-1)

            oo = fun(x)
            idx_min = np.argmin(oo)
            # idx_min = np.random.choice(np.argsort(oo)[:100//10])

            oo = oo[idx_min]
            x = x[idx_min]

        return np.sort(x), oo

    def add(x, nn, exc):
        oo = None
        for _ in range(1, exc+1):
            x, oo = __add(x=x, nn=nn)

        return np.sort(x), oo

    def addremove(x, nn, exc):
        x = np.repeat([x], repeats=len(x), axis=0)
        x = x[np.logical_not(np.eye(len(x), dtype=bool))].reshape(len(x), len(x) - 1)

        x, oo = __add(x=x, nn=nn)
        return np.sort(x), oo

    o = np.inf
    for q in range(1, excursion+1):

        i = 0
        for i in range(max_loop):
            o_old = o
            if method == 'add->remove':
                x0, o = add(x=x0, nn=n, exc=q)
                x0, o = remove(x=x0, exc=q)
            elif method == 'remove->add':
                x0, o = remove(x=x0, exc=q)
                x0, o = add(x=x0, nn=n, exc=q)
            elif method == 'both':
                raise NotImplementedError()
            else:
                raise ValueError('Unknown method, see doc string for more information')

            if o_old - o < improvement_threshold:
                break

        if verbose >= 2:
            print("Depth: {} | Loop {} | Objective: {:.4} | Configuration: {} ".format(q, i+1, o, x0))

    if verbose >= 1:
        print(" Objective: {:.4} | Configuration: {}".format(o, x0))
    return x0, o


def idx_times_all(idx, n):
    idx = np.atleast_2d(idx)
    idx2 = idx.repeat(n, axis=0)
    idx2 = np.hstack((idx2, np.tile(np.arange(n), reps=idx.shape[0])[:, np.newaxis])).astype(int)

    return idx2


def greedy(n, k, fun, verbose=0):

    best = []
    for i in range(k):
        idx_i = idx_times_all(idx=best, n=n)
        opt = fun(idx_i)
        opt[best] = np.inf
        best.append(np.argmin(opt))

    best = np.sort(best)
    obj = fun(best)
    if verbose > 0:
        print(f"{repr(best)} | {obj}")
    return best, obj


def random(n, k, m, fun, chunk=1000,
           n_processes=10,
           dtype=np.uint8, verbose=0):

    def fun2(_m):
        _idx = random_subset(n=n, k=k, m=_m, dtype=dtype)
        _o = fun(_idx)
        return _idx, _o

    idx, o = mp_wrapper(m, fun=fun2, n_processes=n_processes, max_chunk_size=chunk)

    if verbose > 1:
        fig, ax = new_fig()
        ax.hist(o, bins=100)

    i_sorted = np.argsort(o)
    o = o[i_sorted]
    idx = idx[i_sorted].astype(int)

    return idx, o


def ga(n, k, m, fun, verbose, **kwargs):
    from wzk.ga.kofn import kofn

    best, ancestors = kofn(n=n, k=k, fitness_fun=fun,  pop_size=m, verbose=verbose, **kwargs)

    print(repr(best))
    return best, ancestors


# Visualization
def configuration_histogram(idx_list, label_list, color_list, title='', **kwargs):
    n = 100
    offset = 0.0005

    offset = [offset*n, offset]
    fig, ax = new_fig(scale=2, title='Configuration Histogram' + title)
    ax.set_xlim(0, n-1)
    ax.set_xlabel('Configuration Index')
    ax.set_ylabel('Count of Configuration')

    for i, (idx, label, color) in enumerate(zip(idx_list, label_list, color_list)):
        ax.step(offset[0]*i + np.arange(n),
                offset[1]*i + np.bincount(idx.ravel(), minlength=n) / len(idx),
                where='mid', color=color, label=label, **kwargs)

    ax.legend(loc='upper right')
    return ax


def plot_parameter_distribution(par_list, ax=None, bins=100, **kwargs):

    n, m = par_list.shape
    if isinstance(bins, int):
        bins = [bins] * m
    elif isinstance(bins, list):
        if len(bins) == 1:
            bins *= m
        else:
            assert len(bins) == m, f" {len(bins)} != {m} "
    else:
        raise ValueError
    if ax is None:
        ax = subplot_grid(n=m)

    bins2 = []
    for i, xi in enumerate(par_list.T):
        bins_i = ax[np.unravel_index(i, ax.shape)].hist(xi, density=True, bins=bins[i], **kwargs)[1]
        bins2.append(bins_i)
    return ax, bins2


def ancestor_gif(ancestors, n):
    from wzk import new_fig
    import matplotlib.pyplot as plt
    n_gens, pop_size, k = ancestors.shape

    counts = np.zeros((n_gens, n))

    for i, a in enumerate(ancestors):
        u, c = np.unique(a, return_counts=True)
        counts[i, u] = c

    fig, ax = new_fig()
    ax.hlines(pop_size*k/n, xmin=0, xmax=n)
    h = ax.step(np.arange(n), counts[0], where='mid')[0]

    i = 0
    while True:
        h.set_ydata(counts[i])
        i += 1
        i %= n_gens-1
        ax.set_xlabel(f"Generation: {i}")
        plt.pause(0.1)
