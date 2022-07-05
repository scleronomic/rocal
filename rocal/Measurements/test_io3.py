import numpy as np
from mopla.Planner.ardx2 import ardx, pkt2dict

ardx.require("autocalib.detect-marker-ard.marker-detector-result-packets")
ardx.require("robotfusion.kinect-to-ardx.kinect-packets")
ardx.require("monitor.torso-monitor-packets")
#ardx.require("bcatch.imu-to-ard.imu-raw-packets")

#filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Left/paths_10_kinect-left-1657027137-measurements"
filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Right/paths_10_kinect-right_mirror-1657042612-measurements"
#filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Pole/paths_10_kinect-pole-1657033930-measurements"


def pkt_list2dict_list(file):
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
