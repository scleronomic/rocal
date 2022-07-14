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
def get_marker(di, mode='normal'):
    mode_dict = dict(corrected='marker_corrected',
                     normal='marker')
    marker_mode = mode_dict[mode]
    if di[marker_mode]['num'] != 1:
        return False
    else:
        return np.array([di[marker_mode]['detections'][0]['y'], di[marker_mode]['detections'][0]['x']])


def get_marker_list(d, mode='normal'):
    d = load_wrapper(d)
    return np.array([get_marker(di=di, mode=mode) for di in d], dtype=object)


def get_img(di):
    img = di['rgb']['img']
    print(img)
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


def get_qt_kinect(d, q_mode='commanded', m_mode='corrected'):
    d = load_wrapper(d)
    q = get_q_list(d, mode=q_mode)
    t = get_marker_list(d, mode=m_mode)
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

# Marker(num=1, detections=[Point(x=487.765045, y=169.120102, score=0.972401)], img_info=ImgInfo(m_time=1657126401367480320))
# Marker(num=1, detections=[Point(x=93.940529, y=161.911331, score=0.992288)], img_info=ImgInfo(m_time=1657126404481551104))
# Marker(num=1, detections=[Point(x=309.0242, y=286.675323, score=0.961257)], img_info=ImgInfo(m_time=1657126411512713216))
# Marker(num=1, detections=[Point(x=90.209, y=444.70636, score=0.985144)], img_info=ImgInfo(m_time=1657126414249547008))
# Marker(num=1, detections=[Point(x=487.34906, y=458.922302, score=0.966917)], img_info=ImgInfo(m_time=1657126417394576896))
# Marker(num=1, detections=[Point(x=479.589691, y=153.902359, score=0.956294)], img_info=ImgInfo(m_time=1657126420257785088))
# Marker(num=1, detections=[Point(x=75.835579, y=160.319916, score=0.951467)], img_info=ImgInfo(m_time=1657126423444574720))
# Marker(num=1, detections=[Point(x=286.619141, y=270.243317, score=0.96582)], img_info=ImgInfo(m_time=1657126430341669888))
# Marker(num=1, detections=[Point(x=73.391167, y=426.022186, score=0.957798)], img_info=ImgInfo(m_time=1657126433110641664))
# Marker(num=1, detections=[Point(x=472.661255, y=437.210205, score=0.945652)], img_info=ImgInfo(m_time=1657126436152844800))
# Marker(num=1, detections=[Point(x=458.375793, y=136.013199, score=0.952761)], img_info=ImgInfo(m_time=1657126438994677504))
# Marker(num=1, detections=[Point(x=78.344513, y=109.539467, score=0.966252)], img_info=ImgInfo(m_time=1657126442166906368))
# Marker(num=1, detections=[Point(x=294.533142, y=254.614868, score=0.918927)], img_info=ImgInfo(m_time=1657126445360722944))
# Marker(num=1, detections=[Point(x=251.2556, y=415.785645, score=0.935257)], img_info=ImgInfo(m_time=1657126448082727936))
# Marker(num=1, detections=[Point(x=500.17337, y=408.693512, score=0.931204)], img_info=ImgInfo(m_time=1657126451014751232))
# Marker(num=1, detections=[Point(x=501.03772, y=112.317909, score=0.916162)], img_info=ImgInfo(m_time=1657126453956978944))
# Marker(num=1, detections=[Point(x=252.808136, y=103.517288, score=0.924101)], img_info=ImgInfo(m_time=1657126456828786176))
# Marker(num=1, detections=[Point(x=298.269867, y=266.722168, score=0.960588)], img_info=ImgInfo(m_time=1657126462185837568))
# Marker(num=1, detections=[Point(x=88.864296, y=421.361572, score=0.958139)], img_info=ImgInfo(m_time=1657126465060845056))
# Marker(num=1, detections=[Point(x=482.035156, y=427.536957, score=0.974011)], img_info=ImgInfo(m_time=1657126468391873536))
# Marker(num=1, detections=[Point(x=516.794373, y=229.356674, score=0.973853)], img_info=ImgInfo(m_time=1657126471308860928))
# Marker(num=1, detections=[Point(x=56.139339, y=226.134659, score=0.962592)], img_info=ImgInfo(m_time=1657126474650893056))
# Marker(num=0, detections=[], img_info=ImgInfo(m_time=1657126485542064896))
def pkt_list2dict_list_kinect(file):
    ardx2.ardx.require("autocalib.detect-marker-ard.marker-detector-result-packets")
    ardx2.ardx.require("robotfusion.kinect-to-ardx.kinect-packets")
    ardx2.ardx.require("monitor.torso-monitor-packets")

    rgb = ardx2.ardx.read_recorder_file(file, "rgb-kinect", "kinect_rgb_packet")
    torso = ardx2.ardx.read_recorder_file(file, "torso-monitor", "torso_monitor_packet")
    base = ardx2.ardx.read_recorder_file(file, "torso-monitor", "torso_monitor_packet")
    marker = ardx2.ardx.read_recorder_file(file, "marker-rgb-kinect", "MarkerDetectionResultPacket")
    marker_corrected = get_corrected_marker_from_txt(file=file, torso=torso)
    assert len(rgb) == len(marker) == len(torso) == len(base) == len(marker_corrected)

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

    from wzk import print_dict
    marker = []
    for i in range(len(torso)):
        detections = []
        for points in points_per_image[i]:
            detections.append(Point(x=float(points[1]), y=float(points[2]), score=float(points[3])))
        marker.append(Marker(num=len(detections), detections=detections, img_info=ImgInfo(m_time=torso[i].m_time)))
        print(marker[-1])
    return marker
