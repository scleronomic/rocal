import numpy as np

from rocal.calibration import calibrate
from rocal.Robots import Justin19CalKinect
from rocal.Vis.plotting import print_stats2
from rocal.parameter import Parameter, unwrap_x

from rocal.definitions import ICHR22_AUTOCALIBRATION

from rocal.Measurements.test_io3 import get_qus

cal_par = Parameter()

file_pole = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-pole-1657128676-measurements.npy", allow_pickle=True)
file_right = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_70_kinect-right-measurements.npy", allow_pickle=True)
file_left = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_70_kinect-left-measurements.npy", allow_pickle=True)

q_pole, t_pole = get_qus(d=file_pole)
q_right, t_right = get_qus(d=file_right)
q_left, t_left = get_qus(d=file_left)

n = 200
q_pole, t_pole = q_pole[:n], t_pole[:n]
q_right, t_right = q_right[:n], t_right[:n]
q_left, t_left = q_left[:n], t_left[:n]

# if n == 200:
# i_delete_pole = np.array([28,  92,  93,  94,  95,  96,  98, 100], dtype=int)
# i_delete_right = np.array([284, 285, 286, 288, 289, 290, 306, 326, 332], dtype=int) - 200
# i_delete_left = np.array([523], dtype=int) - 400

i_delete_pole = np.array([ 28,  85,  92,  93,  94,  95,  96,  98, 100, 183], dtype=int)
i_delete_right = np.array([262, 282, 284, 285, 286, 288, 289, 290, 306, 309, 313, 321, 326, 332, 350, 351], dtype=int) - 200
i_delete_left = np.array([401, 405, 442, 462, 468, 472, 476, 479, 480, 492, 497, 500, 501, 518, 523, 527, 546, 554, 558, 563, 568, 589], dtype=int) - 400

# i0 = np.arange(600)
# a = np.array([28,  92,  93,  94,  95,  96,  98, 100, 284, 285, 286, 288, 289, 290, 306, 326, 332, 523], dtype=int)
# i1 = np.delete(i0, a)
# b = np.array([84, 175, 254, 274, 294, 298, 306, 333, 334, 384, 388, 425, 445,
#               451, 455, 459, 462, 463, 475, 480, 483, 484, 501, 509, 528, 536,
#               540, 545, 550, 571])
# i2 = np.delete(i1, b)
# j = np.delete(i0, i2)

q_pole, t_pole = np.delete(q_pole, i_delete_pole, axis=0), np.delete(t_pole, i_delete_pole, axis=0)
q_right, t_right = np.delete(q_right, i_delete_right, axis=0), np.delete(t_right, i_delete_right, axis=0)
q_left, t_left = np.delete(q_left, i_delete_left, axis=0), np.delete(t_left, i_delete_left, axis=0)


q = np.concatenate((q_pole, q_right, q_left), axis=0)
l = np.cumsum([0, len(q_pole), len(q_right), len(q_left)])

t = np.zeros((len(q), 3, 2))
t[l[0]:l[1], 0, ...] = t_pole
t[l[1]:l[2], 1, ...] = t_right
t[l[2]:l[3], 2, ...] = t_left


cal_rob = Justin19CalKinect(dkmca='ccfcf', add_nominal_offsets=True, use_imu=False, el_loop=1)


x, stats = calibrate(q_cal=q, t_cal=t, q_test=None, t_test=None, verbose=1, obj_fun='marker_image',
                     cal_par=cal_par, cal_rob=cal_rob, x0_noise=0, )


x = unwrap_x(x=x, cal_rob=cal_rob, add_nominal_offset=True)


from wzk import new_fig
fig, ax = new_fig(aspect=1)
ax.plot(*stats[l[0]:l[1]].T, color='red', label='pole', marker='o', ls='')
ax.plot(*stats[l[1]:l[2]].T, color='magenta', label='right', marker='x', ls='')
ax.plot(*stats[l[2]:l[3]].T, color='blue', label='left', marker='s', ls='')
ax.legend()


mse = np.linalg.norm(stats, axis=-1)
fig, ax = new_fig(title='Uncalibrated')
ax.hist(mse[l[0]:l[1]], color='red', range=(0, 15), bins=50, alpha=0.5, label='pole')
ax.hist(mse[l[1]:l[2]], color='magenta', range=(0, 15), bins=50, alpha=0.5, label='right')
ax.hist(mse[l[2]:l[3]], color='blue', range=(0, 15), bins=50, alpha=0.5, label='left')
ax.legend()
ax.set_xlabel('|pixel difference|')
i_delete = np.nonzero(mse > np.percentile(mse, 95))[0]

print('pole: ', np.round(mse[l[0]:l[1]].mean(), 3))
print('right:', np.round(mse[l[1]:l[2]].mean(), 3))
print('left: ', np.round(mse[l[2]:l[3]].mean(), 3))


# pass


# Full without mass
# pole:  0.831
# right: 0.686
# left:  0.525


# Fuller
# pole:  0.582
# right: 0.605
# left:  0.593


# print('Results:')
# cm = (x['cm'])
# print(cal_rob.cm[3] - cm[3])

# print('Marker Pole:')
# print(cm[0])
#
# print('Marker Right:')
# print(cm[1])
#
# print('Marker Left:')
# print(cm[2])
#
# print('Kinect:')
# print(cm[3])


# print(x)