import numpy as np
from wzk import new_fig

gravity = 1
compliance = 1.8
mass = 1
length = 1
x0 = 1

theta0 = 0


def f(theta):
    return mass*gravity * length*np.cos(theta)


def g(tau):
    return theta0 + compliance*tau


def h(theta):
    return g(f(theta))


x = np.random.normal()
n = 100
x_list = np.zeros(n)
for i in range(n):
    x_list[i] = x
    x = h(x)
    print(x)

fig, ax = new_fig()
ax.plot(x_list)
print('contraction factor:', mass*gravity*length*compliance)
