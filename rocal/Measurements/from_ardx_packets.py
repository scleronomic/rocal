import numpy as np


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

