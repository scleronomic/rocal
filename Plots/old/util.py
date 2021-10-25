import numpy as np
from wzk import new_fig, save_fig


def true_best_with_noises(err_r, err_b, obj_r, obj_b,
                          sqrt_obj=True,
                          save_dir=None):
    rm = err_r.mean(axis=0)
    bm = err_b.mean(axis=0)

    rstd = err_r.std(axis=0)
    bstd = err_b.std(axis=0)

    print("Random, Mean={:.4}, Std={:.4}".format(rm.mean(), rstd.mean()))
    print("Detmax, Mean={:.4}, Std={:.4}".format(bm.mean(), bstd.mean()))

    if sqrt_obj:
        obj_r, obj_b = np.sqrt(obj_r), np.sqrt(obj_b)

    fig, ax = new_fig()
    ax.plot(obj_r, rm, ls='', marker='o', color='r', alpha=0.5, label='Random')
    ax.plot(obj_b, bm, ls='', marker='o', color='b', alpha=0.5, label='Detmax')

    ax.plot(np.vstack((obj_r,) * 2), np.vstack((rm - rstd, rm + rstd)), color='r', alpha=0.3)
    ax.plot(np.vstack((obj_b,) * 2), np.vstack((bm - bstd, bm + bstd)), color='b', alpha=0.3)

    # Mean Cross
    c, ls, a, z = 'k', '-', 0.8, 10
    ax.hlines(rm.mean(), xmin=obj_r.mean()-obj_r.std(), xmax=obj_r.mean()+obj_r.std(),
              color=c, ls=ls, alpha=a, zorder=z)
    ax.hlines(bm.mean(), xmin=obj_b.mean()-obj_b.std(), xmax=obj_b.mean()+obj_b.std(),
              color=c, ls=ls, alpha=a, zorder=z)
    ax.vlines(obj_r.mean(), ymin=rm.mean()-rm.std(), ymax=rm.mean()+rm.std(),
              color=c, ls=ls, alpha=a, zorder=z)
    ax.vlines(obj_b.mean(), ymin=bm.mean()-bm.std(), ymax=bm.mean()+bm.std(),
              color=c, ls=ls, alpha=a, zorder=z)

    # Mark optimum
    i_r = np.argmin(rm)
    print(i_r)
    ax.plot(obj_r[i_r], rm[i_r], ls='', marker='o', color='k', alpha=1, label='Minimum')
    ax.plot(np.vstack((obj_r[i_r],)*2), np.vstack((rm[i_r] - rstd[i_r], rm[i_r] + rstd[i_r])), color='k', alpha=0.8)

    i_b = np.argmin(bm)
    ax.plot(obj_b[i_b], bm[i_b], ls='', marker='o', color='k', alpha=1)
    ax.plot(np.vstack((obj_b[i_b],) * 2), np.vstack((bm[i_b] - bstd[i_b], bm[i_b] + bstd[i_b])), color='k', alpha=0.8)

    ax.set_xlabel('Task A-Optimality')
    ax.set_ylabel('Mean TCP Error')
    ax.legend()

    if save_dir is not None:
        save_fig(fig=fig, filename=save_dir + 'true_best_with_noises', bbox=None, formats=('png', 'pdf'))
