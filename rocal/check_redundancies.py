import sympy
import numpy as np

from wzk.mpl import new_fig
from wzk import numeric_derivative

from rocal.calibration import create_wrapper_kinematic
from rocal.Robots import JustinFinger03Cal
from rocal.parameter import unwrap_x

# cal_rob = Justin19Cal(dkmc='f000')
cal_rob = JustinFinger03Cal(dkmc='f000')

q = cal_rob.sample_q(100)


kin_fun = create_wrapper_kinematic(cal_rob=cal_rob, q=q)
x = np.random.uniform(size=4*4)
# x = np.random.uniform(size=cal_rob.n_dof*4)

idx_f = [-1]


def fun_x(_x):
    f, cm = kin_fun(_x)
    return f[:, idx_f, :3, -1].ravel()


def fun_dh(dh):
    f = cal_rob.get_frames_dh(q=q, dh=dh.reshape(-1, 4))
    return f[:, idx_f, :3, -1].ravel()


fun = fun_x(x)
jac = numeric_derivative(fun=fun_x, x=x)
jac = np.round(jac, 3)


_, idx = sympy.Matrix(jac).rref()  # to check the rows you need to transpose!

b = np.zeros_like(x, dtype=bool)
idx = np.array(idx)
b[idx] = True
print(b.reshape(18, -1))
print(jac.shape, idx.shape)

fig, ax = new_fig()
ax.plot(idx)
