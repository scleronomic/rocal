import numpy as np

alpha = 0.1
x0 = 1


def h(x):
    return x0 + alpha*x


x = np.random.normal()
n = 100
for i in range(n):
    x = h(x)

print(x)


