import sympy
import numpy as np

from Justin.Calibration.calibration import create_wrapper_kinematic, unwrap_x
from Justin.Calibration.justin import Justin19Calib
from wzk import numeric_derivative

cal_rob = Justin19Calib(dcmf='0f00')

q = cal_rob.sample_q(100)


kin_fun = create_wrapper_kinematic(cal_rob=cal_rob, q=q)
x = np.random.random(18*3)


def fun(x):
    return kin_fun(x)[..., :3, -1].ravel()


# def fun(dh):
#     f = robot.get_frames_dh(q=q, dh=dh.reshape(-1, 4))
#     return f[..., [13, 22], :3, -1].ravel()


jac = numeric_derivative(fun=fun, x=x)
jac = np.round(jac, 3)


_, idx = sympy.Matrix(jac).rref()  # to check the rows you need to transpose!

b = np.zeros_like(x, dtype=bool)
idx = np.array(idx)
b[idx] = True
print(b.reshape(18, -1))
print(jac.shape, idx.shape)

from wzk.mpl import new_fig
fig, ax = new_fig()
ax.plot(idx)