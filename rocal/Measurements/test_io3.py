import numpy as np

from wzk.numpy2 import object2numeric_array

from mopla.Planner.ardx2 import ardx, pkt2dict

from rocal.definitions import ICHR22_AUTOCALIBRATION
#ardx.require("bcatch.imu-to-ard.imu-raw-packets")

file_pole = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Pole/paths_10_kinect-pole-1657117796-measurements"
file_right = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Right/paths_20_kinect-right-1657119419-measurements"
file_left = "/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/Left/paths_20_kinect-left-1657118908-measurements"


def combine_measurements():
    file_pole20 = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_20_kinect-pole-1657122658-measurements.npy"
    file_pole50 = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-pole-1657128676-measurements.npy"

    file_right20 = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_20_kinect-right-1657119419-measurements.npy"
    file_right50 = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-right-1657126485-measurements.npy"

    file_left20 = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_20_kinect-left-1657118908-measurements.npy"
    file_left50 = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-left-1657121972-measurements.npy"
    file_right = f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_10_kinect-left-1657027137-measurements.npy"

    a = np.load(file_left20, allow_pickle=True)
    b = np.load(file_left50, allow_pickle=True)
    c = np.concatenate((a[:-1], b[:-1]), axis=0)
    np.save(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_70_kinect-left-measurements.npy", c)


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


def load_wrapper(d):
    if isinstance(d, str):
        d = np.load(d, allow_pickle=True)
    return d


def get_q(di):
    # q - measured joint positions | q_poti - commanded joint positions
    return np.array([di['torso']['q']])[0]


def get_qs(d):
    d = load_wrapper(d)
    return np.array([get_q(di) for di in d])


def get_marker(di):
    if di['marker']['num'] != 1:
        return False
    else:
        return np.array([di['marker']['detections'][0]['y'], di['marker']['detections'][0]['x']])


def get_markers(d):
    d = load_wrapper(d)
    return np.array([get_marker(di) for di in d], dtype=object)


def get_img(di):
    img = di['rgb']['img']
    img = img.reshape(480, 640, 3)
    img = np.swapaxes(img, 0, 1)
    return img


def get_imgs(d):
    return np.array([get_img(di) for di in d])


def plot_all_images(d):
    for i in range(len(d)):
        img = get_img(di=d[i])
        xy = get_marker(di=d[i])

        fig, ax = new_fig(aspect=1)
        plt.imshow(img, origin='lower')
        if np.any(xy):
            plt.plot(*xy, marker='x', color='red', markersize=30, lw=5)
        plt.plot(*u0[i], marker='x', color='blue', markersize=30, lw=5)
        save_fig(fig=fig, file=f"{ICHR22_AUTOCALIBRATION}/{i}_marker", formats='png')


def get_qus(d):
    d = load_wrapper(d)
    q = get_qs(d)
    u = get_markers(d)
    b = np.array([False if ui is False else True for ui in u])

    q, u = q[b], u[b]
    u = object2numeric_array(u)
    return q, u


def pkt_list2dict_list_all():
    from wzk.files import list_directories
    main_directory = '/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect'

    sub_directories = ['Pole', 'Right', 'Left']

    for sub_directory in sub_directories:
        directory = f'{main_directory}/{sub_directory}'
        for file in list_directories(directory):
            print(f'{directory}/{file}')
            pkt_list2dict_list(f'{directory}/{file}')



# pass
def copy_marker_txt():

    from wzk import list_directories, cp
    directory_wink_do = '/home/wink_do/public/autocalib/marker_detections/'
    directory_tenh_jo = '/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/{mode}'
    dirs = list_directories(directory_wink_do)

    print(dirs)
    for d in dirs:

        if 'pole' in d:
            mode = 'Right'
        elif 'right' in d:
            mode = 'Left'
        elif 'left' in d:
            mode = 'Pole'
        else:
            raise ValueError('mode not found')

        file_a = f"{directory_wink_do}/{d}/marker.txt"
        file_b = f"{directory_tenh_jo.format(mode=mode)}/{d}/marker.txt"

        cp(src=file_a, dst=file_b)


if __name__ == '__main__':
    copy_marker_txt()
    # pkt_list2dict_list_all()

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