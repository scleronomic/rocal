import numpy as np


def tsne_embedding():
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    n = 20000

    np.random.seed(0)
    a = sample_q(par=par, n_samples=n, valid=False)[:, 0, :]
    b = np.array(
        [np.load(f'/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRight/{a}/random_poses_1000.npy')[1:-1] for a in
         'ABCDE']).reshape(-1, 19)

    ab = np.vstack((a, b))
    ab_tsne = TSNE(2, verbose=1, n_jobs=10).fit_transform(ab)
    ab_pca = PCA(2).fit_transform(ab)

    fig, ax = new_fig(aspect=1)
    ax.scatter(ab_tsne[:n, 0], ab_tsne[:n, 1], color='b', alpha=0.5, label='Random Configurations')
    ax.scatter(ab_tsne[n:, 0], ab_tsne[n:, 1], color='r', alpha=0.5, label='Valid for Calibration')
    ax.legend()
    ax.set_xlabel('$t_1$')
    ax.set_ylabel('$t_2$')
    save_fig(file='TSNE RIGHT ARM', formats=('png', 'pdf'))

    fig, ax = new_fig(aspect=1)
    ax.scatter(ab_pca[:n, 0], ab_pca[:n, 1], color='b', alpha=0.5)
    ax.scatter(ab_pca[n:, 0], ab_pca[n:, 1], color='r', alpha=0.5)


def plot_correlation():
    save = True
    # directory = DLR_USERSTORE_PAPER_2020_CALIB + 'Figures/10000_j/'
    # directory = DLR_USERSTORE_PAPER_2020_CALIB + 'Figures/100_c/'
    # file_rand1000 = DLR_USERSTORE_PAPER_2020_CALIB + '10over100__TCPError__right__joint_offset__1000.npy'

    # file_rand10000 = DLR_USERSTORE_PAPER_2020_CALIB + '30over100__TCPError__right__custom__10000.npy'
    # file_detmax1000 = DLR_USERSTORE_PAPER_2020_CALIB + '30over100__TCPError__right__custom__1000detmaxgood_w_std001.npy'
    # file_best1000 = DLR_USERSTORE_PAPER_2020_CALIB + '30over100__TCPError__right__custom__1000bestrand.npy'
    # file_worst1000 = DLR_USERSTORE_PAPER_2020_CALIB + '30over100__TCPError__right__custom__927worstrand.npy'


    # file_worst1000 = DLR_USERSTORE_PAPER_2020_CALIB + '30over100__TCPError__right__custom__1000.npy'

    # file_rand10000 = DLR_USERSTORE_PAPER_2020_CALIB + '10over100__TCPError__right__joint_offset__10000.npy'
    # file_best1000 = DLR_USERSTORE_PAPER_2020_CALIB + '10over100__TCPError__right__joint_offset__1000bestrand.npy'
    # file_worst1000 = DLR_USERSTORE_PAPER_2020_CALIB + '10over100__TCPError__right__joint_offset__1000worstrand.npy'
    # # file_detmax1000 = DLR_USERSTORE_PAPER_2020_CALIB + '10over100__TCPError__right__joint_offset__1000bestdetmaxC.npy'
    # # file_detmax1000 = DLR_USERSTORE_PAPER_2020_CALIB + '10over100__TCPError__1000detmaxbest_50_std.npy'
    # file_detmax1000 = DLR_USERSTORE_PAPER_2020_CALIB + '10over100__TCPError__1000detmaxbest_full_std.npy'

    idx_r, par_r, err_mean_r, err_max_r = load_error_stats(file=file_rand10000)
    idx_b, par_b, err_mean_b, err_max_b = load_error_stats(file=file_best1000)
    idx_w, par_w, err_mean_w, err_max_w = load_error_stats(file=file_worst1000)
    # idx_d, par_d, err_mean_d, err_max_d = load_error_stats(file=file_detmax1000)

    # opt_ta, _, opt_d, _ = setup(model='j', cal_set='dummy_j', test_set='100000')
    opt_ta, _, opt_d, _ = setup(model='c', cal_set='100test', test_set='100000')

    ota_r = opt_ta(idx_r)
    ota_b = opt_ta(idx_b)
    ota_w = opt_ta(idx_w)
    # ota_d = opt_ta(idx_d)

    # od_r = opt_d(idx_r)
    # od_b = opt_d(idx_b)
    # od_w = opt_d(idx_w)
    # od_d = opt_d(idx_d)

    i_mine10 = np.argsort(err_mean_r)[:10]
    i_mine100 = np.argsort(err_mean_r)[:100]
    i_mine1000 = np.argsort(err_mean_r)[:1000]

    # i_mina10 = np.argsort(ota_d)[:10]
    # i_mina100 = np.argsort(ota_d)[:100]

    fig, ax = new_fig(title='Bins')
    ota_bins = ota_r.reshape(100, 100).mean(axis=1)
    err_mean_bins = err_mean_r.reshape(100, 100).mean(axis=1)
    iso = np.argsort(ota_bins)
    ax.plot(ota_bins[iso], err_mean_bins[iso], '-b')
    ax.set_xlabel('Task A-Optimality')
    ax.set_ylabel('Mean TCP Error [matrix]')

    fig, ax = new_fig(title='Bins2')
    from wzk import digitize_group
    bins = np.linspace(np.percentile(ota_r, 1), np.percentile(ota_r, 99), 101)
    g = digitize_group(ota_r, bins=bins)
    r = [err_mean_r[gg].mean() for gg in g]
    ax.plot(bins[:], r[:-1], '-b')
    ax.set_xlabel('Task A-Optimality')
    ax.set_ylabel('Mean TCP Error [matrix]')
    # x = np.digitize(ota_r, bins=bins)

    # parameter distribution
    bins = 100
    par_idx = slice(18, None)
    ax, bins = plot_parameter_distribution(par_list=par_r[:, par_idx], color='b', alpha=0.5, bins=bins)
    plot_parameter_distribution(par_list=par_r[i_mine100][:, par_idx], ax=ax, color='r', alpha=0.5, bins=bins)
    # plot_parameter_distribution(par_list=par_d[:, par_idx], ax=ax, color='matrix', alpha=0.5, bins=bins)
    plot_parameter_distribution(par_list=par_b[:, par_idx], ax=ax, color='matrix', alpha=0.5, bins=bins)
    ax.flat[0].figure.suptitle('Parameter Distribution')

    # Correlation Plot
    fig, ax = new_fig(title='Correlation Plot: A-Optimality vs. Max TCP Error')
    ax.plot(ota_r, err_max_r, alpha=0.5, ls='', marker='x', c='k', label='10000 Random Configurations')
    ax.plot(ota_b, err_max_b, alpha=0.5, ls='', marker='o', c='g', label='1000 Good Configurations (Random Search)')
    ax.plot(ota_w, err_max_w, alpha=0.5, ls='', marker='o', c='r', label='1000 Bad Configurations (Random Search)')
    # ax.plot(ota_d, err_max_d, alpha=0.5, ls='', marker='o', c='y', label='1000 Good Configurations (Detmax)')
    ax.set_xlabel('A-Optimality')
    ax.set_ylabel('Max TCP Error [matrix]')
    ax.legend()

    fig, ax = new_fig(title='Correlation Plot: A-Optimality vs. Mean TCP Error', scale=2)
    ax.plot(ota_r, err_mean_r, alpha=0.5, ls='', marker='x', c='k', label='10000 Random Configurations')
    ax.plot(ota_b, err_mean_b, alpha=0.5, ls='', marker='o', c='g', label='1000 Good Configurations (Random Search)')
    ax.plot(ota_w, err_mean_w, alpha=0.5, ls='', marker='o', c='r', label='1000 Bad Configurations (Random Search)')
    # ax.plot(ota_d, err_mean_d, alpha=0.5, ls='', marker='o', c='y', label='1000 Good Configurations (Detmax)')
    ax.set_xlabel('Task A-Optimality')
    ax.set_ylabel('Mean TCP Error [matrix]')

    slope, intercept, r, prob, sterrest = linregress(ota_bins, err_mean_bins)
    xx = np.linspace(ota_r.min()*0.95, ota_r.max()*1.05, 100)
    ax.plot(xx, xx*slope + intercept, 'b', label='Regression Line')

    ax.legend()

    fig, ax = new_fig(title='TCP Error Histogram')
    # hist_bins = np.linspace(0.01, 0.04, 100)
    hist_bins = np.linspace(0.0025, 0.005, 100)
    # ax.hist(err_mean_d, bins=hist_bins, density=True, color='b', alpha=0.5, label='Detmax Best 1000')
    ax.hist(err_mean_b, bins=hist_bins, density=True, color='r', alpha=0.5, label='Best 1000')
    ax.hist(err_mean_w, bins=hist_bins, density=True, color='r', alpha=0.5, label='Worst 1000')
    ax.hist(err_mean_r, bins=hist_bins, density=True, color='r', alpha=0.5, label='All 10000')
    ax.legend()

    # ax[0].hist(idx_r.ravel(), bins=100, range=(-0.5, 99.5))
    n = 100
    k = 10

    # ax = configuration_histogram(title='| All',
    #                              idx_list=[idx_r, idx_d, idx_r[i_mine10], idx_r[i_mine100], idx_r[i_mine1000]],
    #                              label_list=['all', 'good detmax', '10 best mse', '100 best mse',  '100 best mse'],
    #                              color_list=['b', 'g', 'r', 'matrix', 'orange'], alpha=0.5, lw=5)
    # save_fig(fig=ax.get_figure(), formats=('png', 'pdf'), save=save)
    # ax = configuration_histogram(title='| Best vs. Worst',
    #                              idx_list=[idx_d[i_mina10], idx_r[i_mine10]],
    #                              label_list=['10 best ota', '10 best mse'],
    #                              color_list=['b', 'r'],  alpha=0.5, lw=5)
    # save_fig(fig=ax.get_figure(), formats=('png', 'pdf'), save=save)
    # idx = idx_d[i_mina10]
    # bc = np.bincount(idx.ravel(), minlength=n) / len(idx)
    # bc_mine = np.bincount(idx_r[i_mine100].ravel(), minlength=n) / len(idx_r[i_mine100])
    # idx_most_common_best = np.argsort(bc_mine)[-30:]
    # opt_ta(idx_most_common_best)
    # opt_ta(idx_d[i_mina10])

    # a = 1
    # fig, ax = new_fig()
    # ax.imshow(mat.T, cmap='gray_r')
    # ol_all = get_element_overlap(arr1=idx_r[:100], arr2=idx_r[-100:])ÓÓÓ
    # ol.mean()
    # ol_all.mean()
    # a = 1
    # idx_det50test, *_ = np.load(DLR_USERSTORE_PAPER_2020_CALIB + 'detmax__best1000_50test.npy', allow_pickle=True)
    # idx_detfull, *_ = np.load(DLR_USERSTORE_PAPER_2020_CALIB + 'detmax__best1000_full.npy', allow_pickle=True)
    #
    # ax = configuration_histogram(idx_list=[idx_detfull, idx_det50test, idx_r[i_mine100], idx_r[i_mine1000]],
    #                              label_list=['detmax full', 'detmax 50', '100 best mse', '1000 best mse'],
    #                              color_list=['b', 'g', 'r', 'matrix'])
    # save_fig(fig=ax.get_figure(), formats=('png', 'pdf'))


    # nn = 10
    # a = np.sort(np.argsort(np.bincount(idx_det50test.ravel(), minlength=100))[-nn:])
    # b = np.sort(np.argsort(np.bincount(idx_r[i_mine1000].ravel(), minlength=100))[-nn:])
    # b = np.sort(np.argsort(np.bincount(idx_detfull.ravel(), minlength=100))[-nn:])
    # print(a, b)
    # count = 0
    # for aa in a:
    #     if aa in b:
    #         count += 1
    # print(count)
    # fig, ax = new_fig(title="D-Optimality_vs_MeanError", scale=2)
    # ax.plot(od_r, err_mean_r, ls='', marker='x', c='k')
    # ax.plot(od_b, err_mean_b, ls='', marker='o', c='g')
    # ax.plot(od_w, err_mean_w, ls='', marker='o', c='r')
    # ax.set_xlabel('D-Optimality')
    # ax.set_ylabel('Mean TCP Error [matrix]')
    #
    save_all(directory=directory, formats=('png', 'pdf'), close=False, save=save)


def plot_correlation_exhaustive():
    file = DLR_USERSTORE_PAPER_2020_CALIB + '2over100__TCPError__right__joint_offset__exhaustive.npy'
    # file = DLR_USERSTORE_PAPER_2020_CALIB + '10over100__TCPError__right__joint_offset__4845.npy'
    file = DLR_USERSTORE_PAPER_2020_CALIB + '10over100__TCPError__right__joint_offset__10000.npy'

    q_cal, _ = calibration.load_measurements_right_head(torso_right.file_100)
    idx, err_mean, err_max = load_error_stats(file=file)

    q_cal2 = q_cal[idx]

    diff_q = np.linalg.norm(q_cal2[:, :, np.newaxis, :] - q_cal2[:, np.newaxis, :, :], axis=-1).mean(axis=(-2, -1))

    opt_a, _, opt_d, _ = setup(mode='j')
    ota = opt_a(idx=idx)
    ax = correlation_plot(ota, err_mean, name_a='Task A-Optimality', name_b='MSE TCP')
    # save_fig(fig=ax.get_figure())
    ax = correlation_plot(ota, diff_q, name_a='Task A-Optimality', name_b='Norm Q')
    # save_fig(fig=ax.get_figure())
    ax = correlation_plot(err_mean, diff_q, name_a='MSE TCP', name_b='Norm Q')
    # save_fig(fig=ax.get_figure())


def check_optimality_variance():

    n_test_list_max = 1000000
    n_test_list = [1, 10, 20, 30, 40, 50,
                   100, 200, 300, 400, 500,
                   1000, 2000, 3000, 4000, 5000,
                   10000, 20000, 30000, 40000, 50000]

    n_calib = 100
    k = 10
    n_idx = 100
    n_runs = 100

    q = sample_q(robot=robot, n_samples=n_test_list_max)[:, 0, :]

    x0, x_dict = torso_right.get_q0(mode='f')
    x_wrapper = torso_right.create_x_unwrapper(**x_dict)
    calc_targets = calibration.create_wrapper_kinematic(x_wrapper=x_wrapper, kinematic=torso_right.calc_targets, q=q)
    jac = numeric_derivative(fun=calc_targets, x=x0)
    jac = jac[:, 0, :3, -1, 18:]

    print(jac.shape)
    np.save(DLR_USERSTORE_PAPER_2020_CALIB + f'jac_f{jac.shape[-1]}_100000.npy', jac)
    # jac = np.load(dfn.DLR_USERSTORE_PAPER_2020_CALIB + 'jac_f11_100000.npy')
    return
    idx_list = random_subset(n=n_calib, k=k, m=n_idx)
    jac_cal = jac[np.random.choice(n_test_list_max, n_calib, replace=False)]

    res = []
    for i in range(n_runs):
        print(i)
        res1 = []
        for n_test in n_test_list:
            jac_test = jac[np.random.choice(n_test_list_max, n_test, replace=False)]
            opt_a = task_a_optimality_wrapper(jac_calset=jac_cal, jac_test=jac_test)
            res1.append([opt_a(idx=idx) for idx in idx_list])
        res.append(res1)

    res = np.array(res)
    x = res.std(axis=0).mean(-1)
    fig, ax = new_fig(title='STD of A-Optimality')
    ax.loglog(n_test_list, x)
    ax.set_xlabel('Size of Test-Set')
    ax.set_ylabel('mean std of A-Opt)')
    save_fig(dfn.PROJECT_ROOT + 'Variance_A-Opt', fig=fig)

