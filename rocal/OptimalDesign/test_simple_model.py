import numpy as np

from wzk.mpl import new_fig, save_fig, correlation_plot, hist_vlines
from wzk import euclidean_norm, numeric_derivative, random_subset, change_tuple_order
from scipy.optimize import minimize as minimize_scipy

from Plots.old.util import true_best_with_noises

from rocal.OptimalDesign.oed import (task_a_optimality_wrapper, d_optimality_wrapper, greedy,
                                     detmax, configuration_histogram)


# Finding
#   A) The optimal Task-A Criterion is all values = 1 -> maximal, but if all points are the same
#   the result is most likely false, because the redundancies will not be found
#   -> the criterion is good but in addition the values should be far apart / different
#   B) The measurement noise really influences the performance,
#   No consistent result for the usefulness of a-optimality emerges
#   because this noise makes everything gaussian
#   C) Measuring at positions where the cameras work well and have little noise is
#   a good thing, because the optimal calibration poses are those with only little
#   measurement error
#   BUT this seems only to be partly true, as for the Nonlinear model the results look
#   not so consistent again
#   D) One Idea is to use larger models with more measurement points because than
#   the influence of 'measurement luck should be reduced'
#   E) over 1000 runs of the linear model the ota greedy design was on average (median) 0.9 times the median
#   of 1000 sample calibration designs -> is this worth it?
#   but when comparing against the worst ota and an average ota calibration set
#   the optimum performs far better on average. -> make this 3 plots
#   was made just with different noise -> test with completely random setting
#   F) In general, the relative D-efficiency of two designs is defined as the ratio of the two determinants
#   raised to the power 1/p, where p is the number of unknown model parameters.


class Model:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

        self.p_true = None
        self.p_nominal = None
        self.p_error = None

        self.measurement_noise = None
        self.max_parameter_error = None

    def initialize_parameters(self, max_parameter_error):

        if max_parameter_error is not None:
            self.max_parameter_error = max_parameter_error

        self.p_nominal = np.random.random((self.ny, self.nx))
        self.p_error = (np.random.random((self.ny, self.nx)) * 2 - 1) * self.max_parameter_error
        self.p_true = self.p_nominal + self.p_error

    def forward(self, x, p_true):
        raise NotImplementedError

    def forward2(self, x, p_error):
        p_error2 = p_error.reshape(self.p_nominal.shape)
        p_true2 = self.p_nominal + p_error2
        return self.forward(x=x, p_true=p_true2)

    def forward2_wrapper(self, x):
        def __forward2(p_error):
            return self.forward2(x=x, p_error=p_error)
        return __forward2

    def create_dummy_data(self, measurement_noise=None, n=None, x=None):
        if x is None:
            x = np.random.random((n, self.nx))

        if measurement_noise is None:
            measurement_noise = self.measurement_noise

        y = self.forward(x=x, p_true=self.p_true)
        noise = np.random.normal(loc=0, scale=measurement_noise, size=np.shape(y))
        y += noise

        return x, y, noise


class Linear(Model):

    def __init__(self, nx, ny):
        super(Linear, self).__init__(nx=nx, ny=ny)

    def initialize_parameters(self, max_parameter_error, mode='random'):
        self.p_nominal = np.zeros((self.ny, self.nx))
        if mode == 'random':
            self.p_error = (np.random.random((self.ny, self.nx)) * 2 - 1) * max_parameter_error
        else:
            self.p_error = np.full((self.ny, self.nx), max_parameter_error)
        self.p_true = self.p_nominal + self.p_error

    def forward(self, x, p_true):
        return (p_true[np.newaxis, :, :] @ x[..., np.newaxis])[..., 0]


class Diagonal(Linear):

    def __init__(self, nx, ny):
        assert nx == ny
        super(Diagonal, self).__init__(nx=nx, ny=ny)

    def initialize_parameters(self, max_parameter_error, mode=None):
        self.p_nominal = np.eye(self.nx) * np.random.random(self.nx)
        self.p_error = np.eye(self.nx) * (np.random.random((self.ny, self.nx)) * 2 - 1) * max_parameter_error
        self.p_true = self.p_nominal + self.p_error

    def forward2(self, x, p_error):
        p_error2 = np.zeros_like(self.p_nominal)
        p_error2[range(p_error2.shape[0]), range(p_error2.shape[1])] = p_error
        p_true2 = self.p_nominal + p_error2
        return self.forward(x=x, p_true=p_true2)


class NonLinear(Model):

    def __init__(self, nx, ny):
        super(NonLinear, self).__init__(nx=nx, ny=ny)

    def forward(self, x, p_true):
        return np.exp(np.cos((p_true[np.newaxis, :, :] @ x[..., np.newaxis])[..., 0]))


class Arm2d(Model):

    def __init__(self, nx, ny):
        super(Arm2d, self).__init__(nx, ny)

    def initialize_parameters(self, max_parameter_error):
        """
        max_parameter_error: in [rad] for the joint offsets and [matrix] for the link length respectively"""
        self.p_nominal = np.zeros((self.ny, self.nx))
        self.p_nominal[1, :] = 1
        self.p_error = (np.random.random((self.ny, self.nx)) * 2 - 1) * max_parameter_error
        self.p_true = self.p_nominal + self.p_error

    @staticmethod
    def arm2d(q, length):
        q_cs = np.cumsum(q, axis=-1)
        tcp = np.stack([(np.cos(q_cs) * length).sum(axis=-1), (np.sin(q_cs) * length).sum(axis=-1)]).T
        return tcp

    def forward(self, x, p_true):
        return self.arm2d(q=p_true[0, :] + x, length=p_true[1, :])


def mse_prediction(model, p, x, y):
    y2 = model.forward2(p_error=p, x=x)
    res = euclidean_norm(y2 - y, axis=-1, squared=True).mean()
    return res


def calibrate(model, x_cal, y_cal, x0=None, method='L-BFGS-B', verbose=0):
    def fun(p):
        return mse_prediction(model=model, p=p, x=x_cal, y=y_cal)

    x0 = np.zeros(np.size(model.p_error)) if x0 is None else x0
    res = minimize_scipy(x0=x0, fun=fun, method=method, options=dict(disp=verbose > 0))

    return res.x, res.fun


def calibrate_wrapper(model, x_calset, y_calset, x_test, y_test, x0=None, method='linear_lstsq'):
    def __calibrate(idx):
        if method == 'linear_lstsq':
            p = np.linalg.lstsq(a=x_calset[idx], b=y_calset[idx], rcond=None)[0].ravel()
        else:
            p, obj_train = calibrate(model=model, x_cal=x_calset[idx], y_cal=y_calset[idx], x0=x0,
                                     method=method)

        obj_test = mse_prediction(model=model, p=p, x=x_test, y=y_test)
        return p, obj_test

    return __calibrate


def get_mse_parameter(p_true, p):
    return euclidean_norm(p - p_true, axis=-1, squared=True)


def main():
    # For ncal = 1, greedy min and greedy max work really well
    save = False
    n = 1000

    prior_sigma = 0.01

    n_test = 1000
    n_cal = 6
    n_calset = 1000
    n_optval = 10000

    ny = 1
    nx = 2
    measurement_noise = 0.01
    max_parameter_error = 0.1
    # np.random.seed(0)
    x0 = np.zeros(ny * nx)

    model = Linear(nx=nx, ny=ny)

    model.initialize_parameters(max_parameter_error=max_parameter_error)  # mode='random')

    x_test, y_test, noise_test = model.create_dummy_data(measurement_noise=measurement_noise, n=n_test)
    x_optval, y_optval, _ = model.create_dummy_data(measurement_noise=measurement_noise, n=n_optval)
    x_calset, y_calset, noise_calset = model.create_dummy_data(measurement_noise=measurement_noise, n=n_calset)

    # Replace last n_cal values with pure ones -> should have the best signal / noise ratio -> A-Opt
    # x_calset2 = np.linspace(0.1, 1, int(n_calset**(1/nx)))
    # x_calset2 = np.array(np.meshgrid(*[x_calset2]*nx)).reshape(nx, -1).T
    # # x = np.random.uniform(low=1, high=1, shape=(n_cal, nx))
    # np.random.seed(0)
    # x_calset2 = np.random.random((n_calset, nx))
    # np.random.seed()
    # x_calset2, y_calset2, noise_calset2 = model.create_dummy_data(measurement_noise=measurement_noise, x=x_calset2)
    # x_calset[-len(x_calset2):] = x_calset2
    # y_calset[-len(y_calset2):] = y_calset2
    # noise_calset[-len(noise_calset2):] = noise_calset2

    # Calculate Optimality Criteria
    jac_calset = numeric_derivative(fun=model.forward2_wrapper(x=x_calset), x=x0)
    jac_optval = numeric_derivative(fun=model.forward2_wrapper(x=x_optval), x=x0)

    a_task_opt_fun = task_a_optimality_wrapper(jac_calset=jac_calset, jac_test=jac_optval, prior_sigma=prior_sigma)
    d_opt_fun = d_optimality_wrapper(jac=jac_calset)
    idx_greedy_ota, greedy_ota = greedy(n=n_calset, k=n_cal, fun=a_task_opt_fun)
    idx_greedy_nota, greedy_nota = greedy(n=n_calset, k=n_cal, fun=lambda _idx: -a_task_opt_fun(_idx))
    idx_greedy_od, greedy_od = greedy(n=n_calset, k=n_cal, fun=d_opt_fun)
    idx_greedy_nod, greedy_nod = greedy(n=n_calset, k=n_cal, fun=lambda _idx: -d_opt_fun(_idx))

    idx_det, det_ota = change_tuple_order([detmax(fun=a_task_opt_fun, n=n_calset, k=n_cal, excursion=2, max_loop=2)
                                           for _ in range(1000)])
    idx_greedy_ota = idx_det[np.argmin(det_ota)]
    idx_det = np.unique(idx_det, axis=0)

    idx = random_subset(n=n_calset, k=n_cal, m=n)
    idx = np.concatenate((idx, idx_det, [idx_greedy_ota, idx_greedy_nota, idx_greedy_od, idx_greedy_nod]))

    ota = a_task_opt_fun(idx=idx)

    # Perform Calibration
    mse_test_noise = []
    mse_parameter, mse_test = None, None
    for i in range(100):
        x_calset, y_calset, noise_calset = model.create_dummy_data(measurement_noise=measurement_noise, x=x_calset)

        cal_fun = calibrate_wrapper(model=model, x0=x0,
                                    x_calset=x_calset, y_calset=y_calset,
                                    x_test=x_test, y_test=y_test)

        p_cal, mse_test = change_tuple_order([cal_fun(idx=i) for i in idx])
        p_cal, mse_test = np.array(p_cal), np.array(mse_test)
        mse_parameter = get_mse_parameter(p_true=model.p_error.ravel(), p=p_cal)
        mse_test_noise.append(mse_test)

    mse_test_noise = np.array(mse_test_noise)

    true_best_with_noises(err_r=mse_test_noise[:, :n], obj_r=ota[:n],
                          err_b=mse_test_noise[:, n:n+len(idx_det)], obj_b=ota[n:n+len(idx_det)])

    i = -4
    perc_ota_greedy_min = np.sum(mse_test[i] > mse_test[:n]) / n
    # rel_med = mse_test[i] / np.median(mse_test[:n])
    # print(f"{perc_ota_greedy_min} / {len(mse_test)} | {perc_nota_greedy_max} / {len(mse_test)}")
    # return perc_ota_greedy_min, perc_nota_greedy_max
    relative_perf = (mse_test[i] / min(mse_test[:n]) - 1)
    print("{:.2} \t|  {:.2}  \t|  {:.3}  \t|  {:.3}".format(relative_perf, perc_ota_greedy_min, mse_parameter[i],
                                                            np.abs(noise_calset[idx[i]]).mean()))

    # return relative_perf, perc_ota_greedy_min, mse_parameter[i], np.abs(noise_calset[idx[i]]).mean(), rel_med
    ax = correlation_plot(a=mse_parameter, b=mse_test, name_a='MSE Parameter', name_b='MSE Prediction',
                          colors=['g', 'm'], lower_perc=0.1, upper_perc=99.9)

    ax = correlation_plot(a=np.abs(noise_calset[idx]).mean(axis=(-1, -2)), b=mse_test,
                          name_a='Noise', name_b='MSE Prediction')
    save_fig(fig=ax.get_figure(), save=save)
    ax, percs = hist_vlines(x=mse_test, bins=100, name='MSE Prediction',
                            hl_idx=[-4, -3, -2, -1], hl_color=['g', 'r', 'm', 'y'],
                            hl_name=['Greedy A-Min', 'Greedy A-Max', 'Greedy D-Min', 'Greedy D-Max'],
                            lower_perc=0.1, upper_perc=99.9)
    save_fig(fig=ax.get_figure(), save=save)

    # perc_ota_greedy_min, perc_nota_greedy_max, perc_od_greedy_min, perc_nod_greedy_max = percs
    ax, percs = hist_vlines(x=mse_parameter, bins=100, name='MSE Parameter',
                            hl_idx=[-4, -3, -2, -1], hl_color=['g', 'r', 'm', 'y'],
                            hl_name=['Greedy A-Min', 'Greedy A-Max', 'Greedy D-Min', 'Greedy D-Max'],
                            lower_perc=0.1, upper_perc=99.9)
    save_fig(fig=ax.get_figure(), save=save)

    ax = correlation_plot(a=ota, b=mse_test, name_a='Task A-Optimality', name_b='MSE Prediction',
                          colors=['g', 'm'],
                          lower_perc=0.1, upper_perc=99.9)
    save_fig(fig=ax.get_figure(), save=save)

    ax = correlation_plot(a=ota, b=mse_parameter, name_a='Task A-Optimality', name_b='MSE Parameter',
                          colors=['g', 'm'],
                          lower_perc=0.1, upper_perc=99.9)
    save_fig(fig=ax.get_figure(), save=save)

    configuration_histogram(idx_list=[idx[:n], idx_det, idx[np.argsort(mse_test)[:n]]],
                            label_list=['random', 'detmax', 'best'],
                            color_list=['k', 'b', 'g'])

    x_self_diff = np.linalg.norm(x_calset[idx][:, :, np.newaxis, :] -
                                 x_calset[idx][:, np.newaxis, :, :], axis=-1).mean(axis=(-2, -1))
    x_norm = np.linalg.norm(x_calset[idx], axis=-1).mean(axis=-1)

    ax = correlation_plot(a=ota, b=x_norm, name_a='Task A-Optimality', name_b='X Norm',
                          colors=['g', 'm'],
                          lower_perc=0.1, upper_perc=99.9)
    save_fig(fig=ax.get_figure(), save=save)

    ax = correlation_plot(a=x_norm, b=mse_test, name_a='X Norm', name_b='MSE Prediction',
                          colors=['g', 'm'],
                          lower_perc=0.1, upper_perc=99.9)
    save_fig(fig=ax.get_figure(), save=save)

    ax = correlation_plot(a=ota, b=x_self_diff, name_a='Task A-Optimality', name_b='X Self Diff',
                          colors=['g', 'm'],
                          lower_perc=0.1, upper_perc=99.9)
    save_fig(fig=ax.get_figure(), save=save)

    ax = correlation_plot(a=x_self_diff, b=mse_test, name_a='X Self Diff', name_b='MSE Prediction',
                          colors=['g', 'm'],
                          lower_perc=0.1, upper_perc=99.9)
    save_fig(fig=ax.get_figure(), save=save)

    fig, ax = new_fig(title='Tcp Error Histogram')
    ax.hist(mse_test[n:-4], bins=100, density=True, range=(np.percentile(mse_test, 0), np.percentile(mse_test, 99)),
            color='b', alpha=0.5, label='detmax')
    ax.hist(mse_test[:n], bins=100, density=True, range=(np.percentile(mse_test, 0), np.percentile(mse_test, 99)),
            color='r', alpha=0.5, label='random')
    ax.set_xlabel('TCP Error')
    ax.legend()

    save_fig(fig=ax.get_figure(), save=save)


main()
#
# for i in range(100):
#     main()
# a, b, c, d, e = change_tuple_order([main() for _ in range(10000)])

# a, b, c, d, e, [] = np.load(PROJECT_ROOT + '10000_nx2_ny1_ncal4/abcde.npy', allow_pickle=True)
# fig, ax = new_fig(scale=2, title='Histogram: MSE relative to Min')
# ax.hist(a, bins=100, range=(0, 5))
# save_fig(fig=fig, formats='pdf')
#
# fig, ax = new_fig(scale=2, title='Histogram: Perc')
# ax.hist(b, bins=100)
# save_fig(fig=fig, formats='pdf')
#
# fig, ax = new_fig(scale=2, title='Histogram: MSE Error')
# ax.hist(c, bins=100)
# save_fig(fig=fig, formats='pdf')
#
# fig, ax = new_fig(scale=2, title='Histogram: Rel Median')
# ax.hist(e, bins=100, range=(0, 5))
# save_fig(fig=fig, formats='pdf')
#
# correlation_plot(c, b, name_a='MSE Error', name_b='Perc')
# save_fig(fig=fig, formats='pdf')
#
# correlation_plot(a, b, name_a='MSE relative to Min', name_b='Perc')
# save_fig(fig=fig, formats='pdf')
#
#
# correlation_plot(c, d, name_a='MSE Error', name_b='Noise')
# save_fig(fig=fig, formats='pdf')
#
# correlation_plot(b, d, name_a='Perc', name_b='Noise')
# save_fig(fig=fig, formats='pdf')

# np.save(PROJECT_ROOT + 'abcde.npy', (a, b, c, d, e, []))
