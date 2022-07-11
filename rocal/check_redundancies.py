import sympy
import numpy as np

from wzk.mpl import new_fig
from wzk import numeric_derivative, print_dict

from rocal import calibration, parameter
from rocal.Robots import Justin19CalKinect, Justin19CalVicon
from rocal.main_kinect import get_qt

dkmca = 'fffff'
obj_fun_str = 'marker_image'
cal_rob = Justin19CalKinect(dkmca=dkmca, el_loop=1)
cal_par = parameter.Parameter()


q, t, l = get_qt(n=200, mode='commanded')


def test():

    q = np.random.random(100)
    x = np.random.random(5)

    def fun_x(_x):
        return q * (_x[0] + _x[1] + _x[2]) + _x[3] * q**3 + _x[4] * q**4

    jac = numeric_derivative(fun=fun_x, x=x)
    jac = np.round(jac, 3)
    _, idx = sympy.Matrix(jac).rref()  # to check the rows you need to transpose!
    print(idx)


def method_b():
    n, x_bool_dict = parameter.get_x_bool_dict(cal_rob=cal_rob)

    x = np.random.uniform(low=-0.1, high=+0.1, size=n)

    # Pre
    obj_fun = calibration.obj_wrapper(cal_rob=cal_rob, cal_par=cal_par, obj_fun=obj_fun_str)
    obj_fun = obj_fun(q=q, t=t)

    def fun_x(_x):
        d, o = obj_fun(_x, verbose=1)
        return np.linalg.norm(d, axis=-1)

    jac = numeric_derivative(fun=fun_x, x=x)
    jac = np.round(jac, 3)
    a, idx = sympy.Matrix(jac).rref()  # to check the rows you need to transpose!

    print_redundancies(idx=idx, n=n)


def method_a():
    kin_fun = calibration.create_wrapper_kinematic(cal_rob=cal_rob, q=q)
    x = np.random.uniform(size=cal_rob.n_dh*4)

    # x = np.random.uniform(size=cal_rob.n_dof*4)

    idx_f = [13, 22, 26]

    def fun_x(_x):
        f = kin_fun(_x)[0]
        return f[:, idx_f, :3, -1].ravel()


    def fun_dh(dh):
        f = cal_rob.get_frames_dh(q=q, dh=dh.reshape(-1, 4))
        return f[:, idx_f, :3, -1].ravel()


    jac = numeric_derivative(fun=fun_x, x=x)
    jac = np.round(jac, 3)


    _, idx = sympy.Matrix(jac).rref()  # to check the rows you need to transpose!


    print(b.reshape(cal_rob.n_dh, -1))
    print(jac.shape, idx.shape)

    fig, ax = new_fig()
    ax.plot(idx)


def print_redundancies(idx, n):
    b = np.zeros(n, dtype=bool)
    b[np.array(idx, dtype=int)] = True
    b = parameter.unwrap_x(x=b, cal_rob=cal_rob)
    print_dict(b)


if __name__ == '__main__':
    method_b()
