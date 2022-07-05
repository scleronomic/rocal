import numpy as np
from mopla.Planner.ardx2 import ardx, pkt2dict

from rocal.definitions import ICHR22_AUTOCALIBRATION
#ardx.require("bcatch.imu-to-ard.imu-raw-packets")

#filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Left/paths_10_kinect-left-1657027137-measurements"
filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Right/paths_10_kinect-right_mirror-1657042612-measurements"
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

#
# file = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_10_kinect-right_mirror-1657042612-measurements.npy"
#
# d = np.load(file, allow_pickle=True)
#
#
# for i in range(40):
#     img = d[i]['rgb']['img']
#     marker_detections = d[i]['marker']['detections']
#     print(d[i]['marker'])
#     print(marker_detections)
#
# def reshape_img(img):
#     img = img.reshape(480, 640, 3)
#     img = np.swapaxes(img, 0, 1)
#     return img
#
# img = reshape_img(img=img)
#
#
# from matplotlib import pyplot as plt
#
# # from wzk.mpl import new_fig, imshow
#
# # fig, ax = new_fig(aspect=1)
# # imshow(ax=ax, img=img)
#
# plt.imshow(img, origin='lower')