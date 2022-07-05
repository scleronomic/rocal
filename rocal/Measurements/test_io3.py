import numpy as np
from mopla.Planner.ardx2 import ardx, pkt2dict

from rocal.definitions import ICHR22_AUTOCALIBRATION
#ardx.require("bcatch.imu-to-ard.imu-raw-packets")

filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Left/paths_10_kinect-left-1657027137-measurements"
# filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Right/paths_10_kinect-right_mirror-1657042612-measurements"
#filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Pole/paths_10_kinect-pole-1657033930-measurements"


def pkt_list2dict_list(file):
    ardx.require("autocalib.detect-marker-ard.marker-detector-result-packets")
    ardx.require("robotfusion.kinect-to-ardx.kinect-packets")
    ardx.require("monitor.torso-monitor-packets")

    rgb = ardx.read_recorder_file(file, "rgb-kinect", "kinect_rgb_packet")
    marker = ardx.read_recorder_file(file, "marker-rgb-kinect", "MarkerDetectionResultPacket")
    torso = ardx.read_recorder_file(file, "torso-monitor", "torso_monitor_packet")
    base = ardx.read_recorder_file(file, "torso-monitor", "torso_monitor_packet")
    assert len(rgb) == len(marker) == len(torso) == len(base)

    d = []
    for i in range(len(rgb)):
        d.append(dict(rgb=pkt2dict(rgb[i]),
                      marker=pkt2dict(marker[i]),
                      torso=pkt2dict(torso[i]),
                      base=pkt2dict(base[i])))

    np.save(file, arr=d)


pkt_list2dict_list(file=filepath)

from wzk.mpl import new_fig, imshow, save_fig

file = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_10_kinect-right_mirror-1657042612-measurements.npy"

# d = np.load(file, allow_pickle=True)
#
# i = 3
# img = d[i]['rgb']['img']
# marker = d[i]['marker']
#
# # for j in range(10):
# #     print(j, d[i]['marker']['num'], d[i]['marker']['detections'])
#
#
# def get_marker_u(marker):
#     if marker['num'] != 1:
#         return None
#     else:
#         return np.array([marker['detections'][0]['y'], marker['detections'][0]['x']])
#
#
# def reshape_img(img):
#     img = img.reshape(480, 640, 3)
#     img = np.swapaxes(img, 0, 1)
#     return img
#
#
# def plot_all_images():
#     for i in range(len(d)):
#         img = d[i]['rgb']['img']
#         img = reshape_img(img=img)
#         xy = get_marker_u(d[i]['marker'])
#         fig, ax = new_fig(aspect=1)
#         plt.imshow(img, origin='lower')
#         if xy is not None:
#             plt.plot(*xy, marker='x', color='red', markersize=30, lw=5)
#         save_fig(fig=fig, file=f"{ICHR22_AUTOCALIBRATION}/{i}_marker", formats='png')
#
# img = reshape_img(img=img)
#
# fig, ax = new_fig(aspect=1)
# # imshow(ax=ax, img=img)
#
#
#
#
# from matplotlib import pyplot as plt
#
#
# plot_all_images()


# from wzk.mpl import new_fig, imshow
# fig, ax = new_fig(aspect=1)
# imshow(ax=ax, img=img)

# plt.imshow(img, origin='lower')
# plt.plot(*get_marker_u(marker), marker='x', color='red', markersize=20)