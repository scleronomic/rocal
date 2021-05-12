from wzk import rename_dict_keys

file_results = '/volume/USERSTORE/tenh_jo/0_Data/Calibration/Results/Parameters/selected_dh_torques_xyz.npy'
x, x_dict = load_results(file=file_results)


def update_dict(x_dict, x):
    bx = np.ones_like(x, dtype=bool)
    bx[-26:-20] = False
    x = x[bx]
    rename_dict_keys(x_dict, {'dh_bool_arm': 'dh_bool_arm',
                              'cp_bool_arm': 'cp_bool_arm'})
    x_dict.pop('compliance_bool_base')
    save_results(x=x, x_dict=x_dict, file=file_results)


x[:18] = [1.12872064, 1.45718515, 0.06700128, -0.01409389, 0.00415192, 0.07664874,
          0.03542709, -0.01508473, 0.28121661, 0.01368702, - 0.2085893, 0.22174258,
          -0.1110068, -0.01912222, 0.08283752, -2.46331742, -0.14503255, 2.02023273]



directory = '/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRight/Validation/Measurements/'
file = directory + 'm_fixed0_paths.msgpk'
file_cal = directory + 'm_fixed0_cal_paths.msgpk'



q_meas, t_meas = load_measurements_right_head(file=file)
q_meas_cal, t_meas_cal = load_measurements_right_head(file=file_cal)
q_meas_cal, t_meas_cal = load_measurements_right_head(file=file_cal)


calc_targets2 = create_wrapper_kinematic(create_x_wrapper=create_x_wrapper, kinematic=calc_targets,
                                         x_dict=x_dict, x=x)

t_sim = calc_targets2(q_meas)
t_sim_cal = calc_targets2(q_meas_cal)

d_meas = t_meas[1:-1, 0, :3, -1] - t_meas_cal[1:-1, 0, :3, -1]
d_sim = t_sim[1:-1, 0, :3, -1] - t_sim_cal[1:-1, 0, :3, -1]
print(d_meas - d_sim)


f_world_base = trans_rotvec2frame(trans=x[0:3], rotvec=x[3:6])
f_head_target = trans_rotvec2frame(trans=x[6:9], rotvec=x[9:12])
f_right_target = trans_rotvec2frame(trans=x[12:15], rotvec=x[15:18])

# Two arms
file_two_arms = '/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/Measurements/torso-left-right_10.msgpk'
q_meas_2arms, t_meas_2arms = load_measurements_right_head(file=file_two_arms)
t_sim_2arms = calc_targets2(q_meas_2arms)

d_2arms = t_meas_2arms[1:-1, 0, :3, -1] - t_sim_2arms[1:-1, 0, :3, -1]
print(d_2arms)
#                      mean       std    median       min       max
# Translation [matrix]   0.00346   0.00215   0.00282   0.00043   0.01530
#  Rotation [deg]   0.32060   0.26432   0.27045   0.03264   1.72985
# Right [ 64  36  25  87  63  38  32 101  29  50]
#
# Head
#                      mean       std    median       min       max
# Translation [matrix]   0.00214   0.00105   0.00201   0.00020   0.00549
#  Rotation [deg]   0.31429   0.16901   0.27731   0.07055   0.91046
#
# Head [ 39  27  60  25 101  32  97  56  33  12]


