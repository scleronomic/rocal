import numpy as np
import scipy.io

# from rocal.Robots import JustinHand12Cal
# _cal_rob = JustinHand12Cal()
# pairs = _cal_rob.cm_f_idx[pairs]


def load_data(file):
    mat = scipy.io.loadmat(file)['data'][0, 0]
    q = np.concatenate((mat[1], mat[2], mat[3], mat[4], mat[5], mat[6]))
    n = len(q)
    n_i = np.array([len(mat[i]) for i in range(1, 7)])

    # t_f t_m t_r f_m f_r m_r
    pairs = np.array([[3, 2],
                      [3, 1],
                      [3, 0],
                      [2, 1],
                      [2, 0],
                      [1, 0]])
    pairs = np.concatenate([pp[np.newaxis].repeat(nn, axis=0) for pp, nn, in zip(pairs, n_i)])
    d = np.zeros(n)
    return q, d, pairs


# Bad example:
# q[10] = np.array([0.3027871, -0.3161505, 0.6642020,
#                   -0.0171715, 0.4152261, 0.8547665,
#                   -0.4747111, 0.3777183, 0.1215826,
#                   -0.0106201, -0.1294758, 0.9147583])
