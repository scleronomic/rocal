import pandas as pd

from rocal.Measurements.io2 import get_q
from rocal.Robots.Justin19 import Justin19Cal

cal_rob = Justin19Cal(dkmc='000c', ma0=True, fr0=True, use_imu=False, el_loop=1)
(q0_cal, q_cal, t_cal), (_, _, _) = get_q(cal_rob=cal_rob, split=-1, seed=75)

df = pd.DataFrame([(q, t) for q, t in zip(q0_cal, t_cal[:, :, :3, -1])], columns=('joints', 'markers'))
print(df.shape)
