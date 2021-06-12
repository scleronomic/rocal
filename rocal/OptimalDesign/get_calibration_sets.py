import numpy as np

from wzk import random_subset, change_tuple_order, get_timestamp

from rocal.OptimalDesign.oed import random, detmax, greedy
from rocal.Robots import Justin19Cal
# from rocal.parameter import get_parameter_identifier

from rocal.definitions import ICHR20_CALIBRATION


robot = Justin19Cal()

# setup = None


def load_sjj(directory):
    pass

    # try:
    #     sjj_test = np.load(f"{directory}/{cal_rob.id}/sjj_{cal_rob.target_mode}__{name_test}__test.npy")
    # except FileNotFoundError:
    #     sjj_test = (jac_test.transpose((0, 2, 1)) @ jac_test).mean(axis=0)
    #     np.save(f"{directory}/{cal_rob.id}/sjj_{cal_rob.target_mode}__{name_test}__test.npy", sjj_test)
    #
    # assert jac.shape[-1] == sjj_test.shape[-1], f"{jac.shape[-1]} != {sjj_test.shape[-1]}"
    #
    # ta_fun = task_a_optimality_wrapper(jac_calset=jac, sum_jac_jac=sjj_test, prior=prior)
    # return ta_fun


def create_random_stairs():
    n = 10000
    m = 100
    k_list = np.arange(1, 51)
    idx = [random_subset(n=n, k=k, m=m) for k in k_list]
    idx = np.array(idx + [[]], dtype=object)

    return idx


def random3():
    n = 10000
    m = 100
    # k_list = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] + np.arange(1, 51).tolist())
    k_list = np.array([50, 40, 30])
    u, i = np.unique(k_list, return_index=True)
    k_list = k_list[np.sort(i)]

    opt_a, *_ = setup(model='j', cal_set='dummy', test_set='100000', verbose=3)
    idx = []
    obj = []
    for k in k_list:
        # i, o = random(n=n, k=k, matrix=matrix, fun=opt_a, dtype=np.uint16, verbose=0)
        print(k)
        if k == 1:
            i, o = np.atleast_2d(*greedy(n=n, k=1, fun=opt_a))
        else:
            i, o = change_tuple_order(detmax(fun=opt_a, n=n, k=k, excursion=min(5, k), max_loop=10, verbose=0)
                                      for _ in range(int(m*1.1)))
        idx.append(np.array(i)[:m])
        obj.append(np.array(o)[:m])

        idx2 = np.array(idx + [[]], dtype=object)
        obj2 = np.array(obj).ravel()
        np.save(ICHR20_CALIBRATION + 'idx_DetmaxBest_150.npy', (idx2, obj2))


def random2(c, directory, n, k, m):
    opt_a, *_ = setup(directory=directory, c=c)
    idx, obj = random(n=n, k=k, m=m, fun=opt_a, verbose=0, n_processes=10, dtype=np.uint16)


def greedy2():
    pass


def detmax2(c, directory,
            n, m, k, factor=1.1,
            save=False,
            verbose=1):

    opt_a = setup(c=c, directory=directory)

    i, o = change_tuple_order([detmax(fun=opt_a, n=n, k=k, excursion=min(k, 5), max_loop=10, verbose=3)
                               for _ in range(int(factor*m))])
    i, j = np.unique(i, axis=0, return_index=True)
    i = i[:m]
    o = np.array(o)[j[:m]]

    if verbose > 0:
        from wzk import print_stats

        print_stats(o, names='Task A-Optimality (Detmax)')
        print(f"Task A-Optimality Min={np.min(o)} Median={np.median(o)}")
        print(f"idx: {i.shape} , obj: {o.shape}")

    save_idx_wrapper(c, directory, name, save=False)

    return i, o


def save_idx_wrapper(c, directory, name, save=False):
    if save:
        yon = input("Should this Detmax run be saved? (y or n)")
        if yon == 'y':
            name2 = get_parameter_identifier(c, )
            np.save(directory + f'idx__{name}__{k}over{n}_{m}__{name2}__{get_timestamp()}', (i, o, []))


if __name__ == '__main__':
    # i, o = detmax2(c=c, directory=ICHR20_CALIBRATION, n=10000, k=50, m=10, save=True)
    pass
