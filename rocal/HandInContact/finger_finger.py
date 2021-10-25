import numpy as np
from wzk import tic, toc
from wzk.spatial import sample_frame_noise, trans_rotvec2frame, frame2trans_rotvec, frame_difference, invert  # noqa

from rocal.calibration import calibrate, obj_wrapper
from rocal.Robots import JustinHand12Cal, JustinHand12CalCal  # noqa
from rocal.parameter import Parameter
from rocal.measurment_functions import _cal_touch

from rocal.HandInContact.load import load_data
from rocal.HandInContact.plots import finger_before_after, finger_hist, print_result


if __name__ == '__main__':
    pass
    cal_par = Parameter()
    cal_rob = JustinHand12Cal(dkmc='j000')
    # cal_rob = JustinHand12CalCal(dkmc='j000')

    cm0 = np.eye(4)[np.newaxis].repeat(4, axis=0)

    file1 = '/Users/jote/Documents/Code/Python/DLR/rocal/Data/JustinHand12/export-1.mat'  # 14.06
    file2 = '/Users/jote/Documents/Code/Python/DLR/rocal/Data/JustinHand12/export-2.mat'  # 15.06
    file3 = '/Users/jote/Documents/Code/Python/DLR/rocal/Data/JustinHand12/export-3.mat'  # 05.07
    file4 = '/Users/jote/Documents/Code/Python/DLR/rocal/Data/JustinHand12/export-4.mat'  # 28.07

    q, d, pairs = load_data(file4)
    f = cal_rob.get_frames(q)
    d0 = _cal_touch(f=f, pairs=pairs, cal_rob=cal_rob, cm=cm0)
    print(len(d0))
    tic()
    x, d1 = calibrate(q_cal=q, t_cal=(pairs, d), q_test=q, t_test=(pairs, d), verbose=1,
                      obj_fun='touch',
                      cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
    toc('CALIBRATION')

    # convert to mm
    d0, d1 = d0 * 1000, d1 * 1000
    print_result(d0=d0, d1=d1, d=0)

    # cm1 = np.reshape(x[-4 * 6:], (4, 6)).copy()
    # trans2, rot2 = cm1[:, :3], cm1[:, 3:]
    # cm1 = trans_rotvec2frame(trans=trans2, rotvec=rot2)
    # # # TODO do i need to normalize the rotation
    # cm2 = invert(cm1[2]) @ cm1  # normalize on fore finger [2]
    # f0 = cal_rob.f_static[1::2]
    # f2 = cm2 @ f0
    # print(repr(f2))

    finger_before_after(d0=d0, d1=d1, pairs=pairs)
    finger_hist(d0=d0, d1=d1, pairs=pairs)
    #
    # # Test on different dataset
    # q_test, d_test, pairs = load_data(file3)
    # obj_fun = obj_wrapper(cal_rob=cal_rob, cal_par=cal_par, obj_fun='touch')
    # obj_fun_test = obj_fun(q=q_test, t=(pairs, d_test))
    # d_test0 = obj_fun_test(x=x*0, verbose=1)[0]
    # d_test1 = obj_fun_test(x=x, verbose=1)[0]
    # d_test0, d_test1 = d_test0 * 1000, d_test1 * 1000
    # print(np.abs(x).mean())
    # print_result(d0=d_test0, d1=d_test1, d=0)
    # finger_before_after(d0=d_test0, d1=d_test1, pairs=pairs)
    # finger_hist(d0=d_test0, d1=d_test1, pairs=pairs)

# print(repr(x.reshape(4, 4, 4)))
print(repr(x.reshape(4, 4)))
# np.abs(x).mean()
# 0.0033887093880088804