import numpy as np


from rocal.definitions import ICHR22_AUTOCALIBRATION
from rocal.Measurements.from_ardx_packets import get_img, get_marker, pkt_list2dict_list

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


def plot_all_images(d):
    from wzk.mpl import new_fig, save_fig, plt
    for i in range(len(d)):
        img = get_img(di=d[i])
        xy = get_marker(di=d[i])

        fig, ax = new_fig(aspect=1)
        plt.imshow(img, origin='lower')
        if np.any(xy):
            plt.plot(*xy, marker='x', color='red', markersize=30, lw=5)
        plt.plot(*u0[i], marker='x', color='blue', markersize=30, lw=5)
        save_fig(fig=fig, file=f"{ICHR22_AUTOCALIBRATION}/{i}_marker", formats='png')


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
    pkt_list2dict_list_all()

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