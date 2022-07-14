import numpy as np
from rocal.definitions import ICHR22_AUTOCALIBRATION, ICHR22_AUTOCALIBRATION_FIGS

from wzk.mpl import new_fig, save_fig, set_style, set_borders
arr = np.load(f"{ICHR22_AUTOCALIBRATION}/Results/vicon-old__cc0c0__difference_right_left.npy", allow_pickle=True).item()

t0 = arr['t0']
t2 = arr['t2']
i_delete = np.array([ 68,  79,  83, 141, 194, 224, 280, 313, 321, 332, 488, 491, 492, 528])
t0, t2 = np.delete(t0, i_delete, axis=0), np.delete(t2, i_delete, axis=0)

t0, t2 = t0[..., :-1, -1], t2[..., :-1, -1]

dr_02 = np.linalg.norm(t0[:, 0] - t2[:, 0], axis=1) * 1000
dl_02 = np.linalg.norm(t0[:, 1] - t2[:, 1], axis=1) * 1000


d0_lr = (t0[:, 1] - t0[:, 0])
d2_lr = (t2[:, 1] - t2[:, 0])
dn2_lr = np.linalg.norm(d2_lr, axis=-1)
dn0_lr = np.linalg.norm(d0_lr, axis=-1)
drl_02 = np.abs(dn2_lr - dn0_lr) * 1000

set_borders(left=0.13, right=0.995, bottom=0.24, top=0.99)
set_style(('ieee',))
fig, ax = new_fig(width='ieee1c', height=1.3)
style = dict(alpha=0.5, range=(0, 10), bins=50, density=True)
ax.hist(dr_02, color='blue', label='absolut right', **style)
ax.hist(dl_02, color='red', label='absolut left', **style)
ax.hist(drl_02, color='cyan', label='relative right-left', **style)
ax.set_xlabel('Error [mm]', labelpad=0)
ax.set_ylabel('Density')
ax.legend()

save_fig(fig=fig, file=f"{ICHR22_AUTOCALIBRATION_FIGS}/difference_right_left.pdf", bbox=None)
