import numpy as np

from rokin.Robots import Justin19
from rokin.Vis.robot_3d import animate_path, pv

from mopla.Parameter import get_par_justin19
from mopla.World.real_obstacles import add_tube_img

from rocal.definitions import ICHR22_AUTOCALIBRATION, ICHR22_AUTOCALIBRATION_FIGS


robot = Justin19()

q = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/q10000_random_kinect-pole.npy")

limits = np.array([[-1.5, +1.5],  # 3.0
                   [-1.5, +1.5],  # 3.0
                   [-0.5, +2.5]])  # 3.0

img = np.zeros((256, 256, 256), dtype=bool)
add_tube_img(img=img, x=np.array([1.3, 0., 0.4]), length=0.8, radius=0.025, limits=limits)


pl = pv.Plotter()

d = 0.04
z = 0.86
y = -0.005
y1 = pv.Cube((1.3, y+d/2, z), x_length=d, y_length=d, z_length=d)
y2 = pv.Cube((1.3, y-d/2, z-d), x_length=d, y_length=d, z_length=d)

b1 = pv.Cube((1.3, y-d/2, z), x_length=d, y_length=d, z_length=d)
b2 = pv.Cube((1.3, y+d/2, z-d), x_length=d, y_length=d, z_length=d)

pl.add_mesh(y1, color='yellow', opacity=1)
pl.add_mesh(y2, color='yellow', opacity=1)

pl.add_mesh(b1, color='black', opacity=1)
pl.add_mesh(b2, color='black', opacity=1)

animate_path(pl=pl, q=q, robot=robot, kwargs_world=dict(img=img, limits=limits, box=False))
pl.show()
