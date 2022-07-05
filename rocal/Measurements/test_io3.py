import ardx.ardx as ardx
ardx.require("autocalib.detect-marker-ard.marker-detector-result-packets")
ardx.require("robotfusion.kinect-to-ardx.kinect-packets")
ardx.require("monitor.torso-monitor-packets")
#ardx.require("bcatch.imu-to-ard.imu-raw-packets")

#filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Left/paths_10_kinect-left-1657027137-measurements"
filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Right/paths_10_kinect-right_mirror-1657042612-measurements"
#filepath = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Pole/paths_10_kinect-pole-1657033930-measurements"

rgb = ardx.read_recorder_file(filepath, "rgb-kinect", "kinect_rgb_packet")
marker = ardx.read_recorder_file(filepath, "marker-rgb-kinect", "MarkerDetectionResultPacket")
torso = ardx.read_recorder_file(filepath, "torso-monitor", "torso_monitor_packet")
#imu_base = ardx.read_recorder_file(filepath, "imu-base", "IMURDataPacket")


def ardx2dict(a):
    d = {}
    for b in dir(a):
        if b.startswith('__'):
            continue

        bb = getattr(a, b)

        if isinstance(bb, (int, float)):
            d[b] = bb

        else:
            try:
                d[b] = ardx.numpy_view(bb)
            except TypeError:
                d[b] = ardx2dict(bb)

    return d


print(ardx2dict(a=rgb[0]))
print(ardx2dict(a=marker[0]))
print(ardx2dict(a=torso[0]))


