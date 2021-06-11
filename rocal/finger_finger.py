import numpy as np

from mopla.Kinematic.forward import frames2pos

from rocal.calibration import calibrate
from rocal.Robots import JustinHand12Cal

from wzk.spatial import trans_rotvec2frame
from wzk.geometry import capsule_capsule_pairs



def pairwise(f, robot, ij, Ti, Tj, mode):
    i, j = ij
    f_i = f[..., i, :, :]
    f_j = f[..., j, :, :]

    f_i = Ti @ f_i
    f_j = Tj @ f_j

    ii = np.nonzero(robot.capsules_f_idx == i)[0].item()
    jj = np.nonzero(robot.capsules_f_idx == j)[0].item()

    x_i = (f_i[..., np.newaxis, :, :] * robot.capsules_pos[ii][:, np.newaxis, :]).sum(axis=-1)[..., :-1]
    x_j = (f_j[..., np.newaxis, :, :] * robot.capsules_pos[jj][:, np.newaxis, :]).sum(axis=-1)[..., :-1]

    lines = np.concatenate((x_i[..., np.newaxis, :, :], x_j[..., np.newaxis, :, :]), axis=-3)
    xa, xb = capsule_capsule_pairs(lines=lines, pairs=np.arange(2)[np.newaxis, :], radii=robot.capsules_rad[[i, j]])


if __name__ == '__main__':
    pass
    # test_seed2()
    # iterative_mass_compliance()
    # evaluate_different_effects©()
    # leave_one_out_analysis_joints()
    # test_static_equilibrium_truncation()

    cal_rob = JustinHand12Cal(dcmf='000f')

    from wzk.spatial import sample_frames, invert

    a, b, c = sample_frames(shape=3)
    d1 = a @ b @ c
    d2 = a @ b @ invert(a) @ a @ c
    assert np.allclose(d1, d2)

    robot = cal_rob
    f = cal_rob.get_frames(q=cal_rob.sample_q(10))
    ij = 5, 11
    Ti = np.eye(4)
    Tj = np.eye(4)

    # directory = ICHR20_CALIBRATION + '/Measurements/600'
    # (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=-1, seed=75)
    #
    #
    # x, stats = calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=1,
    #                      cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)