import numpy as np
from wzk import new_fig

gravity = 9.81
stiffness = 3.456
mass = 1.23
kappa = 0.1  # 1 / stiffness
f_gravity = mass * gravity
x0 = 1


def f_elasticity(x):
    return stiffness * (x - x0)


n = 100
xi = x0 + np.random.normal()
x_list = np.zeros(n)
for i in range(n):
    print(xi)
    x_list[i] = xi
    xi = xi + kappa * (- f_elasticity(xi))

x_true = mass * gravity / stiffness + x0
print('Truth:    ', x_true)
print('Iteration:', xi)


fig, ax = new_fig()
ax.semilogy(np.abs(np.diff(x_list)))
