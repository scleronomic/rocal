import os
import numpy as np

from wzk import object2numeric_array

from mopla.Planner import ardx2


def load_wrapper(d):
    if isinstance(d, str):
        d = np.load(d, allow_pickle=True)
    return d


def get_q(di, mode='commanded'):
    mode_dict = dict(commanded='q_poti',
                     measured='q')
    mode = mode_dict[mode]
    # q - measured joint positions | q_poti - commanded joint positions

    return np.array([di['torso'][mode]])[0]


def get_q_list(d, mode='commanded'):
    d = load_wrapper(d)
    return np.array([get_q(di, mode=mode) for di in d])


# Kinect
def get_marker(di):
    if di['marker']['num'] != 1:
        return False
    else:
        return np.array([di['marker']['detections'][0]['y'], di['marker']['detections'][0]['x']])


def get_marker_list(d):
    d = load_wrapper(d)
    return np.array([get_marker(di) for di in d], dtype=object)


def get_img(di):
    img = di['rgb']['img']
    img = img.reshape(480, 640, 3)
    img = np.swapaxes(img, 0, 1)
    return img


def get_img_list(d):
    d = load_wrapper(d)
    return np.array([get_img(di) for di in d])


# Vicon
def get_targets(di):
    assert di['vicon']['num_targets'] >= 2

    target_dict = dict()
    for i in range(di['vicon']['num_targets']):
        target_dict[di['vicon']['targets'][i]['name']] = di['vicon']['targets'][i]['f_global_target']

    t_left = target_dict[b'Agile Justin Head']
    t_right = target_dict[b'BiggestObject']

    t = np.stack((t_right, t_left), axis=0)
    return t


def get_targets_list(d):
    d = load_wrapper(d)
    return np.array([get_targets(di) for di in d])


def get_qt_vicon(file, mode='commanded'):
    q = get_q_list(file, mode=mode)
    t = get_targets_list(file)

    return q, t


def get_qt_kinect(d, mode='commanded'):
    d = load_wrapper(d)
    q = get_q_list(d, mode=mode)
    t = get_marker_list(d)
    b = np.array([False if ti is False else True for ti in t])

    q, t = q[b], t[b]
    t = object2numeric_array(t)
    return q, t


def pkt_list2dict_list_vicon(file):
    ardx2.ardx.require("monitor.torso-monitor-packets")
    ardx2.ardx.require("vicon-to-ardx.vicon-packets")

    torso = ardx2.ardx.read_recorder_file(file, "torso-monitor", "torso_monitor_packet")
    vicon = ardx2.ardx.read_recorder_file(file, "vicon", "vicon_tracker_packet")
    assert len(torso) == len(vicon)

    d = []
    for i in range(len(torso)):
        d.append(dict(vicon=ardx2.pkt2dict(vicon[i]),
                      torso=ardx2.pkt2dict(torso[i])))

    np.save(file, arr=d)


def pkt_list2dict_list_kinect(file):
    ardx2.ardx.require("autocalib.detect-marker-ard.marker-detector-result-packets")
    ardx2.ardx.require("robotfusion.kinect-to-ardx.kinect-packets")
    ardx2.ardx.require("monitor.torso-monitor-packets")

    rgb = ardx2.ardx.read_recorder_file(file, "rgb-kinect", "kinect_rgb_packet")
    torso = ardx2.ardx.read_recorder_file(file, "torso-monitor", "torso_monitor_packet")
    base = ardx2.ardx.read_recorder_file(file, "torso-monitor", "torso_monitor_packet")
    marker = ardx2.ardx.read_recorder_file(file, "marker-rgb-kinect", "MarkerDetectionResultPacket")
    marker_corrected = get_corrected_marker_from_txt(file=file, torso=torso)
    assert len(rgb) == len(marker) == len(torso) == len(base)

    d = []
    for i in range(len(rgb)):
        d.append(dict(rgb=ardx2.pkt2dict(rgb[i]),
                      marker=ardx2.pkt2dict(marker[i]),
                      marker_corrected=ardx2.pkt2dict(marker_corrected[i]),
                      torso=ardx2.pkt2dict(torso[i]),
                      base=ardx2.pkt2dict(base[i])))

    np.save(file, arr=d)


def get_corrected_marker_from_txt(file, torso):
    if not os.path.exists(f"{file}/marker.txt"):
        return [None] * len(torso)

    from collections import defaultdict
    from collections import namedtuple

    ardx2.ardx.require("monitor.torso-monitor-packets")

    Point = namedtuple('Point', 'x y score')
    ImgInfo = namedtuple('ImgInfo', 'm_time')
    Marker = namedtuple('Marker', 'num detections img_info')

    with open(f"{file}/marker.txt", "r") as f:
        points_per_image = defaultdict(lambda: [])
        for line in f.readlines():
            data = line.split()
            points_per_image[int(data[0])].append(data)

    marker = []
    for i in range(len(torso)):
        detections = []
        for points in points_per_image[i]:
            detections.append(Point(x=float(points[1]), y=float(points[2]), score=float(points[3])))
        marker.append(Marker(num=len(detections), detections=detections, img_info=ImgInfo(m_time=torso[i].m_time)))

    return marker
