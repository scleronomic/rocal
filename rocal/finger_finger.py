import numpy as np
from itertools import combinations
from wzk.spatial import sample_frame_noise, trans_rotvec2frame, frame2trans_rotvec, frame_difference

from rocal.calibration import calibrate
from rocal.Robots import JustinHand12Cal
from rocal.parameter import Parameter
from rocal.measurment_functions import _cal_touch

if __name__ == '__main__':
    pass
    cal_par = Parameter()
    cal_rob = JustinHand12Cal(dkmc='000f')

    n = 60
    pairs = np.array(list(combinations(cal_rob.cm_f_idx, 2)))
    pairs = np.repeat(pairs, n/6, axis=0)

    q = cal_rob.sample_q(n)
    f = cal_rob.get_frames(q)
    cm = sample_frame_noise(shape=4, trans=0.05, rot=0.2)
    cm0 = sample_frame_noise(shape=4, trans=0., rot=0.0)

    d = _cal_touch(f=f, pairs=pairs, cal_rob=cal_rob, cm=cm)
    d0 = _cal_touch(f=f, pairs=pairs, cal_rob=cal_rob, cm=cm0)

    x, stats = calibrate(q_cal=q, t_cal=(pairs, d), q_test=q, t_test=(pairs, d), verbose=1,
                         obj_fun='touch',
                         cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)

    print('max error 0:', np.abs(d - d0).max()*1000, 'mm')
    print('max error  :', stats['max']*1000, 'mm')
    x = np.reshape(x, (4, 6))
    trans2, rot2 = x[:, :3], x[:, 3:]
    cm2 = trans_rotvec2frame(trans=trans2, rotvec=rot2)

    trans, rot = frame2trans_rotvec(f=cm)
    d_trans, d_rot = frame_difference(cm, cm2)
    print(trans - trans2)  # TODO there is redundancy!
    print('tans')
    for i in range(4):
        print(np.round(trans[i], 4))
        print(np.round(trans2[i], 4))
        print(np.round(d_trans[i], 4))

    print(trans2-trans)
    print('rot')
    for i in range(4):
        print(np.round(rot[i], 4))
        print(np.round(rot2[i], 4))
        print(np.round(d_rot[i], 4))
