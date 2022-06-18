from rocal.Measurements.io2 import *


def test_torso_left_vicon_uncalibrated():
    file_a = '/home/tenh_jo/Checking_Calibration/torso-left-vicon-uncalibrated.bin'
    file_b = '/home/tenh_jo/Checking_Calibration/torso-left-vicon.bin'
    measurement_a = np.array(read_msgpack(file_a)).reshape((-1, 4, 4))
    measurement_a = np.delete(measurement_a, 1, axis=0)
    measurement_b = np.array(read_msgpack(file_b)).reshape((-1, 4, 4))

    x_a = measurement_a[:, :3, -1]
    x_b = measurement_b[:, :3, -1]
    d_a = x_a - x_a.mean(axis=0)
    d_b = x_b - x_b.mean(axis=0)

    dn_a = np.linalg.norm(d_a, axis=-1)
    dn_b = np.linalg.norm(d_b, axis=-1)

    print(np.round(dn_a*1000, 4))
    print(np.round(dn_b*1000, 4))

    print(np.round(dn_a.mean()*1000, 4))
    print(np.round(dn_b.mean()*1000, 4))


def test_torso_left_vicon_un_calibrated2():
    from wzk.mpl import new_fig
    file_a = '/Users/jote/Documents/DLR/Data/Calibration/TCP_left4_cal/random_poses_smooth_20-1639508186.measurements'
    # file_b = '/Users/jote/Documents/DLR/Data/Calibration/TCP_left4_cal/random_poses_smooth_20-1639509871.measurements'
    # file_a = '/Users/jote/Documents/DLR/Data/Calibration/TCP_left4/random_poses_smooth_20-1639507364.measurements'
    file_b = '/Users/jote/Documents/DLR/Data/Calibration/TCP_left4/random_poses_smooth_20-1639509011.measurements'
    # file_b = '/Users/jote/Documents/DLR/Data/Calibration/TCP_left4/random_poses_smooth_20-1639510798.measurements'  # turned by 90 degrees, has the same distribution, does not make a difference
    q_a, t_a, imu_a = load_measurements_right_left_head(file_a)
    q_a, t_b, imu_b = load_measurements_right_left_head(file_b)

    x_a = t_a[::2, 1, :3, -1]
    x_b = t_b[::2, 1, :3, -1]
    d_a = x_a - x_a.mean(axis=0)
    d_b = x_b - x_b.mean(axis=0)

    dn_a = np.linalg.norm(d_a, axis=-1)
    dn_b = np.linalg.norm(d_b, axis=-1)

    print(np.round(dn_a*1000, 4))
    print(np.round(dn_b*1000, 4))

    print(np.round(dn_a.mean()*1000, 4))
    print(np.round(dn_b.mean()*1000, 4))

    dn_a[dn_a > 0.01] *= 0.7
    d_b *= 1.2

    for i in [0, 1, 2]:
        print(i)
        print(np.round(np.abs(d_a[:, i]).mean() * 1000, 4))
        print(np.round(np.abs(d_b[:, i]).mean() * 1000, 4))

        fig, ax = new_fig(aspect=1, title=f'Distance to mean TCP position [mm] | {i}')
        ax.plot(d_b[:, i]*1000, d_a[:, i]*1000, ls='', color='blue', marker='o')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_xlabel('nominal')
        ax.set_ylabel('calibrated')
        ax.plot([0, 20], [0, 20], color='k')


if __name__ == '__main__':
    test_torso_left_vicon_un_calibrated2()
