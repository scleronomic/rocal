import numpy as np

from wzk.numpy2 import object2numeric_array
from wzk import print_dict
from mopla.Planner.ardx2 import ardx, pkt2dict

from rocal.definitions import ICHR22_AUTOCALIBRATION


file = "/volume/USERSTORE/tenh_jo/Data/Calibration/TorsoRightLeft/0/random_poses_smooth_3-1657533295-measurements"


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


def get_qs(d, mode='commanded'):
    d = load_wrapper(d)
    return np.array([get_q(di, mode=mode) for di in d])


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


def get_qus(d, mode='commanded'):
    d = load_wrapper(d)
    q = get_qs(d, mode=mode)
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


def copy_marker_txt():

    from wzk import list_directories, cp
    directory_wink_do = '/home/wink_do/public/autocalib/marker_detections/'
    directory_tenh_jo = '/volume/USERSTORE/tenh_jo/Data/Calibration/Kinect/{mode}'
    dirs = list_directories(directory_wink_do)

    print(dirs)
    for d in dirs:

        if 'pole' in d:
            mode = 'Pole'
        elif 'right' in d:
            mode = 'Right'
        elif 'left' in d:
            mode = 'Left'
        else:
            raise ValueError('mode not found')

        file_a = f"{directory_wink_do}/{d}/marker.txt"
        file_b = f"{directory_tenh_jo.format(mode=mode)}/{d}/marker.txt"

        cp(src=file_a, dst=file_b)


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