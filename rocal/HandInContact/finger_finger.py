import numpy as np
from wzk import tic, toc
from wzk.spatial import sample_frame_noise, trans_rotvec2frame, frame2trans_rotvec, frame_difference, invert

from rocal.calibration import calibrate, obj_wrapper
from rocal.Robots import JustinHand12Cal, JustinHand12CalCal
from rocal.parameter import Parameter
from rocal.measurment_functions import _cal_touch

from rocal.HandInContact.load import load_data
from rocal.HandInContact.plots import finger_before_after, finger_hist


if __name__ == '__main__':
    pass
    cal_par = Parameter()
    cal_rob = JustinHand12Cal(dkmc='f000')
    # cal_rob = JustinHand12CalCal(dkmc='000f')

    cm0 = np.eye(4)[np.newaxis].repeat(4, axis=0)

    file1 = '/Users/jote/Documents/Code/Python/DLR/rocal/export-1.mat'
    file3 = '/Users/jote/Documents/Code/Python/DLR/rocal/export-3.mat'

    q, d, pairs = load_data(file3)
    f = cal_rob.get_frames(q)
    d0 = _cal_touch(f=f, pairs=pairs, cal_rob=cal_rob, cm=cm0)

    tic()
    x, d1 = calibrate(q_cal=q, t_cal=(pairs, d), q_test=q, t_test=(pairs, d), verbose=1,
                      obj_fun='touch',
                      cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
    toc('CALIBRATION')

    # convert to mm
    d0, d1 = d0 * 1000, d1 * 1000
    print('max error before:', np.abs(d - d0).max(), 'mm')
    print('max error after :', np.abs(d - d1).max(), 'mm')

    cm1 = np.reshape(x[-4 * 6:], (4, 6)).copy()
    trans2, rot2 = cm1[:, :3], cm1[:, 3:]
    cm1 = trans_rotvec2frame(trans=trans2, rotvec=rot2)
    # # TODO do i need to normalize the rotation
    # cm2 = invert(cm1[2]) @ cm1  # normalize on fore finger [2]
    #
    # d2 = _cal_touch(f=f, cm=cm2, pairs=pairs, cal_rob=cal_rob) * 1000
    # print(d0.mean(), d0.max())
    # print(d1.mean(), d1.max())
    # print(d2.mean(), d2.max())
    # f0 = cal_rob.f_static[1::2]
    # f2 = cm2 @ f0

    finger_before_after(d0=d0, d1=d1, pairs=pairs)
    finger_hist(d0=d0, d1=d1, pairs=pairs)

    q_test, d_test, pairs = load_data(file1)
    # f = cal_rob.get_frames(q)
    # d20 = _cal_touch(f=f, pairs=pairs, cal_rob=cal_rob, cm=cm0) * 1000
    # d21 = _cal_touch(f=f, pairs=pairs, cal_rob=cal_rob, cm=cm1) * 1000

    obj_fun = obj_wrapper(cal_rob=cal_rob, cal_par=cal_par, obj_fun='touch')
    obj_fun_test = obj_fun(q=q_test, t=(pairs, d_test))
    d_test0 = obj_fun_test(x=x*0, verbose=1)[0]
    d_test1 = obj_fun_test(x=x, verbose=1)[0]
    d_test0, d_test1 = d_test0 * 1000, d_test1 * 1000

    finger_before_after(d0=d_test0, d1=d_test1, pairs=pairs)
    finger_hist(d0=d_test0, d1=d_test1, pairs=pairs)
