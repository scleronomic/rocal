from wzk.mpl import (set_style)

from A_Plots.Calibration2020.util import true_best_with_noises
from rocal.definitions import ICHR20_CALIBRATION

set_style(s=('ieee',))

from Justin.Calibration import setup
from Justin.Calibration import load_error_stats

fig_width_inch = 10
dir_files = ICHR20_CALIBRATION + 'Dummy_j_001/'
dir_figures = ICHR20_CALIBRATION + 'Figures/Dummy_j_001/'


opt_ta, _, opt_d, _ = setup(model='j', cal_set='dummy', test_set='100000')

file_rand_n = dir_files + 'error_Random_100_100noises.npy'
file_det_b_n = dir_files + 'error_DetmaxBest_100_100noises.npy'


idx_rand_n, par_rand_n, err_mean_rand_n, err_max_rand_n = load_error_stats(file=file_rand_n)
idx_det_b_n, par_det_b_n, err_mean_det_b_n, err_max_det_b_n = load_error_stats(file=file_det_b_n)

ota_rand_n = opt_ta(idx_rand_n[0])
ota_det_b_n = opt_ta(idx_det_b_n[0])


true_best_with_noises(err_r=err_mean_rand_n, err_b=err_mean_det_b_n, obj_r=ota_rand_n, obj_b=ota_det_b_n,
                      save_dir=dir_figures)

