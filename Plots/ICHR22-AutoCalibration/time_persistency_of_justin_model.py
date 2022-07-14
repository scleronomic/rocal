import numpy as np

from wzk.mpl import new_fig

from rocal.Measurements import io2, from_ardx_packets
from rocal.definitions import ICHR22_AUTOCALIBRATION

file_old = f"{ICHR22_AUTOCALIBRATION}/Vicon/2/random_poses_smooth_100-1605712000.measurements"
# file_old_b = f"{ICHR22_AUTOCALIBRATION}/Vicon/2/random_poses_smooth_100-1605017698.measurements"
file_old_b = f"{ICHR22_AUTOCALIBRATION}/Vicon/2/random_poses_smooth_100-1604597912.measurements"
# file_old = f"{ICHR22_AUTOCALIBRATION}/Vicon/random_poses_smooth_100-1637256337.measurements"  # here we only had the right marker not mounted on the robot
file_new = f"{ICHR22_AUTOCALIBRATION}/Vicon/2/random_poses_smooth_100-1657536656-measurements.npy"

q_old, t_old, _ = io2.load_measurements_right_left_head(file_old_b)
t_old = t_old[:, :2, :, :]  # ignore the head

q_new, t_new = from_ardx_packets.get_qt_vicon(file=file_new, mode='commanded')
# q_new, t_new, _ = io2.load_measurements_right_left_head(file_old_b)
# t_new = t_new[:, :2, :, :]  # ignore the head


# difference in joints
dqnm_old_new = np.linalg.norm(q_new - q_old, axis=-1).mean()
print(f"mean |q_new - q_old|   = {np.round(np.rad2deg(dqnm_old_new), 3)} deg")  # TODO are both really commanded? probably not, they were always in a separate file


d_lr_old = t_old[:, 1, :-1, -1] - t_old[:, 0, :-1, -1]
d_lr_new = t_new[:, 1, :-1, -1] - t_new[:, 0, :-1, -1]


dn_lr_old = np.linalg.norm(d_lr_old, axis=-1)
dn_lr_new = np.linalg.norm(d_lr_new, axis=-1)

# fig, ax = new_fig(n_rows=2)
ax[0].plot(1000 * (dn_lr_old-dn_lr_new), marker='o', color='blue')
ax[0].set_ylabel('|dn_lr_old_a - dn_lr_new| [mm]')
ax[1].hist(1000 * (dn_lr_old-dn_lr_new), color='red', density=True)
ax[1].set_xlabel('|dn_lr_old_a - dn_lr_new| [mm]')

# Finding
#   After two years there is a max difference of 0.5 mm in the norm between the left and right markers.
#   comparing only the old measurements in itself, the max difference is 0.2 mm
