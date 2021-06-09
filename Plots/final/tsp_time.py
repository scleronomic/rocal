import numpy as np

from wzk import tsp

from Kinematic.Robots import Justin19

robot = Justin19()

time_limit = 20
n = 1000
np.random.seed(1)
q = robot.sample_q(n)

dist_mat = np.abs(q[:, np.newaxis, :] - q[np.newaxis, :, :]).max(axis=-1)
route = tsp.solve_tsp(points=q, dist_mat=dist_mat, time_limit=time_limit, verbose=1)

q2 = q[route]
dist0 = np.abs(np.diff(q, axis=0)).max(axis=-1).sum(axis=0)
dist2 = np.abs(np.diff(q2, axis=0)).max(axis=-1).sum(axis=0)
print(dist0)
print(dist2)

# Finding, using TSP, one can reduce the time by factor 2
