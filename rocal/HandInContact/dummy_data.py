import numpy as np
from itertools import combinations
from rocal.Robots import JustinHand12Cal
from wzk.spatial import sample_frame_noise


def create_dummy_data():
    cal_rob = JustinHand12Cal(dkmc='000f')
    n = 60
    pairs = np.array(list(combinations(cal_rob.cm_f_idx, 2)))
    pairs = np.repeat(pairs, n/6, axis=0)
    q = cal_rob.sample_q(n)
    f = sample_frame_noise(shape=4, trans=0.05, rot=0.2)

    return pairs, q, f
