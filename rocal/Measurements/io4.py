import numpy as np

from wzk.numpy2 import object2numeric_array
from wzk import print_dict
from mopla.Planner.ardx2 import ardx, pkt2dict

from rocal.definitions import ICHR22_AUTOCALIBRATION


file = "/volume/USERSTORE/tenh_jo/Data/Calibration/TorsoRightLeft/2/random_poses_smooth_100-1657536656-measurements"


def pkt_list2dict_list(file):
    ardx.require("monitor.torso-monitor-packets")
    ardx.require("vicon-to-ardx.vicon-packets")

    torso = ardx.read_recorder_file(file, "torso-monitor", "torso_monitor_packet")
    vicon = ardx.read_recorder_file(file, "vicon", "vicon_tracker_packet")
    assert len(torso) == len(vicon)

    # name = vicon[0]
    # print(ardx.to_string(vicon[2].targets[1].name))
    # print(ardx.to_string(vicon[2].targets[1].f_global_target))
    # dir(vicon[2].targets[1])
    #
    # v = ardx.numpy_view(vicon[2].targets[1].name)
    # ''.join(chr(vv) for vv in v[v != 0])
    #
    # dir(vicon[2])
    d = []
    for i in range(len(torso)):
        d.append(dict(vicon=pkt2dict(vicon[i]),
                      torso=pkt2dict(torso[i])))
        print_dict(d[-1])

    np.save(file, arr=d)


if __name__ == '__main__':
    pass

    # copy_marker_txt()
    # pkt_list2dict_list_all()
    pkt_list2dict_list(file=file)
    print('done')
    # from matplotlib import pyplot as plt
    #
    # from wzk.mpl import new_fig, imshow, save_fig
    # from rokin.Robots import Justin19
    # from rocal.Tools import KINECT, MARKER_LEFT, MARKER_RIGHT, MARKER_POLE
    #
    # # file = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_10_kinect-left-1657027137-measurements.npy"
    # # file = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_10_kinect-right_mirror-1657042612-measurements.npy"
    # file = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_10_kinect-pole-1657033930-measurements.npy"
    #
    # d = np.load(file, allow_pickle=True)
    #
    # q = get_qs(d)
    # robot = Justin19()
    # u0 = KINECT.project_marker2image(robot=robot, marker=MARKER_POLE, q=q, distort=False)
    # u0[:, :] = u0[:, ::-1]
    #
    # plot_all_images(d=d)


# from wzk.mpl import new_fig, imshow
# fig, ax = new_fig(aspect=1)
# imshow(ax=ax, img=reshape_img(img=img))
#
# plt.imshow(img, origin='lower')
# plt.plot(*get_marker_u(marker), marker='x', color='red', markersize=20)