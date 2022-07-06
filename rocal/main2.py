import numpy as np

from rocal.calibration import calibrate
from rocal.Robots.Justin19_Kinect import Justin19CalKinect
from rocal.Vis.plotting import print_stats2
from rocal.parameter import Parameter, unwrap_x

from rocal.definitions import ICHR22_AUTOCALIBRATION

from rocal.Measurements.test_io3 import get_qus

cal_par = Parameter()

file_pole = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_10_kinect-pole-1657033930-measurements.npy"
file_right = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_10_kinect-right_mirror-1657042612-measurements.npy"
file_left = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_10_kinect-left-1657027137-measurements.npy"
# file_right = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_10_kinect-left-1657027137-measurements.npy"

q_pole, t_pole = get_qus(d=file_pole)
q_right, t_right = get_qus(d=file_right)
q_left, t_left = get_qus(d=file_left)

q = np.concatenate((q_pole, q_right, q_left), axis=0)

l = np.cumsum([0, len(q_pole), len(q_right), len(q_left)])

t = np.zeros((len(q), 3, 2))
t[l[0]:l[1], 0, ...] = t_pole
t[l[1]:l[2], 1, ...] = t_right
t[l[2]:l[3], 2, ...] = t_left


cal_rob = Justin19CalKinect(dkmc='000c', add_nominal_offsets=True, use_imu=False, el_loop=1)


x, stats = calibrate(q_cal=q, t_cal=t, q_test=None, t_test=None, verbose=1, obj_fun='marker_image',
                     cal_par=cal_par, cal_rob=cal_rob, x0_noise=0, )


x = unwrap_x(x=x, cal_rob=cal_rob, add_nominal_offset=True)


print('Results:')
cm = (x['cm'])
# print(cal_rob.cm[3] - cm[3])

print('Marker Pole:')
print(cm[0])

print('Marker Right:')
print(cm[1])

print('Marker Left:')
print(cm[2])

print('Kinect:')
print(cm[3])
