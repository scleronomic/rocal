import numpy as np
import copy
from wzk import spatial


class Parameter:
    def __init__(self, x_weighting=0., t_weighting=1.):
        # Justin, two marker
        self.lambda_trans, self.lambda_rot = 1000, 0  # was 100
        self.t_weighting = t_weighting

        self.x_weighting = x_weighting
        self.x_nominal = 0

        self.prior_sigma = 0.01  # 0.01  # was  0.01 for dummy
        self.mu_sigma = 0  #

        self.method = 'PyOpt - SLSQP'  # way faster
        self.options = {'maxiter': 200,
                        'disp': True,
                        'ftol': 1e-6,
                        'sens_step': 1e-6,
                        'sens_type': 'fd',  # 'fd' or 'cs
                        'pll_type': None}  # 'POA' or None  | I measured no speed difference


def get_active_parameters(cal_rob):

    """
    dh: DH parameters (rho)
        '0' - zero   ( turn all dh parameters off )
        'j' - joint  ( joint offset )
        'f' - full   ( all 4 DH parameters per joint )
        'c' - custom ( subsection of n x 4 dh parameters based on prior studies )

    el: Elasticity (kappa)
        '0' - zero   ( turn all elasticity off )
        'j' - joint  ( joint elasticity )
        'f' - full   ( joint el + both lateral compliances n x 3 )
        'c' - custom ( subsection of n x 3 compliances based on prior studies )

    ma: Mass (nu)
        '0' - zero
        'm' - mass
        'p' - position
        'f' - full
        'c' - custom

    cm: close measurement loop
        '0' - zero
        'p' - position
        'o' - orientation
        'f' - full
        'c' - custom

    ad: additional parameters (user has to handle those inside the measurement function)
        '0' - zero
        'f' - full
        'c' - custom
    """

    dh, el, ma, cm, ad = cal_rob.dkmca
    n_dh, n_el, n_cm, n_ma, n_ad = cal_rob.n_dh, cal_rob.n_el, cal_rob.n_cm, cal_rob.n_ma, cal_rob.n_ad
    m_dh, m_el, m_ma, m_cm, m_ad = 4, 3, 4, 6, 1

    __error_string = ("Unknown {}-mode '{}'. Use one of the following: "
                      " ['j' (joint offsets), 'f' (full), 'c' (custom)]")

    # DH
    if dh == '0':
        dh_bool = np.zeros((n_dh, m_dh), dtype=bool)
    elif dh == 'j':
        dh_bool = np.zeros((n_dh, m_dh), dtype=bool)
        dh_bool[:, 1] = True
    elif dh == 'f':
        dh_bool = np.ones((n_dh, m_dh), dtype=bool)
    elif dh == 'c':
        dh_bool = cal_rob.dh_bool_c
        assert dh_bool.shape == (n_dh, m_dh)

    else:
        raise ValueError(__error_string.format('dh', dh))

    # Elasticity
    if el == '0':
        el_bool = np.zeros((n_el, m_el), dtype=bool)

    elif el == 'g':  # gravity  # hack
        el_bool = np.zeros((n_el, m_el), dtype=bool)
        el_bool[:3, 1] = True

    elif el == 'j':
        el_bool = np.zeros((n_el, m_el), dtype=bool)
        el_bool[:, -1] = True
    elif el == 'f':
        el_bool = np.ones((n_el, m_el), dtype=bool)
    elif el == 'c':
        el_bool = cal_rob.el_bool_c
        assert el_bool.shape == (n_el, m_el)

    else:
        raise ValueError(__error_string.format('el', el))

    # Mass
    if ma == '0':
        ma_bool = np.zeros((n_ma, m_ma), dtype=bool)
    elif ma == 'm':
        ma_bool = np.zeros((n_ma, m_ma), dtype=bool)
        ma_bool[:, -1] = True
    elif ma == 'p':
        ma_bool = np.zeros((n_ma, m_ma), dtype=bool)
        ma_bool[:, :3] = True
    elif ma == 'f':
        ma_bool = np.ones((n_ma, m_ma), dtype=bool)
    elif ma == 'c':
        ma_bool = cal_rob.ma_bool_c
        assert ma_bool.shape == (n_ma, m_ma)
    else:
        raise ValueError(__error_string.format('ma', ma))

    # Close measurement
    if cm == '0':
        cm_bool = np.zeros((n_cm, m_cm), dtype=bool)
    elif cm == 'p':
        cm_bool = np.zeros((n_cm, m_cm), dtype=bool)
        cm_bool[:, :3] = True
    elif cm == 'o':
        cm_bool = np.zeros((n_cm, m_cm), dtype=bool)
        cm_bool[:, 3:] = True
    elif cm == 'f':
        cm_bool = np.ones((n_cm, m_cm), dtype=bool)
    elif cm == 'c':
        cm_bool = cal_rob.cm_bool_c
        assert cm_bool.shape == (n_cm, m_cm)
    else:
        raise ValueError(__error_string.format('cm', cm))

    # Additional parameters
    if ad == '0':
        ad_bool = np.zeros((n_ad, m_ad), dtype=bool)
    elif ad == 'f':
        ad_bool = np.ones((n_ad, m_ad), dtype=bool)
    elif ad == 'c':
        ad_bool = cal_rob.ad_bool_c
        assert ad_bool.shape == (n_ad, m_ad)
    else:
        raise ValueError(__error_string.format('ad', ad))

    # excluded_joints = cal_rob.frame_frame_influence[:, cal_rob.cm_f_idx].sum(axis=-1) == 0
    # excluded_joints = excluded_joints[cal_rob.joint_f_idx_dh]
    # dh_bool[excluded_joints] = False
    # el_bool[excluded_joints] = False
    return dh_bool, el_bool, ma_bool, cm_bool, ad_bool


def get_x_bool_dict(cal_rob):
    dh_bool, el_bool, ma_bool, cm_bool, ad_bool = get_active_parameters(cal_rob)
    n = dh_bool.sum() + el_bool.sum() + ma_bool.sum() + cm_bool.sum() + ad_bool.sum()
    x_bool_dict = dict(dh_bool=dh_bool, el_bool=el_bool, ma_bool=ma_bool, cm_bool=cm_bool, ad_bool=ad_bool)
    return n, x_bool_dict


def __calibration_bool2number(cal_bool, idx0, x):
    cal = np.zeros_like(cal_bool, dtype=float)
    idx1 = idx0 + cal_bool.sum()
    cal[cal_bool] = x[idx0:idx1]
    return cal, idx1


def set_bool_dict_false(x_dict, x=None):
    x_dict0 = copy.deepcopy(x_dict)
    c = 0
    for key in x_dict0:
        c += int(np.sum(x_dict0[key][:]))
        x_dict0[key][:] = False

    if x is None:
        return x_dict0
    else:
        x0 = x[:len(x)-c]
        return x_dict0, x0


def update_dict_wrapper(d2):
    def update_dict(d):
        d.update(d2)
        return d

    return update_dict


def wrap_x(x, cal_rob):
    n, x_bool_dict = get_x_bool_dict(cal_rob=cal_rob)

    if isinstance(x, dict):
        dh, el, ma, cm, ad = tuple([x[key] for key in x])

    elif isinstance(x, (tuple, list)):
        dh, el, ma, cm, ad = tuple([xx for xx in x])

    else:
        raise ValueError

    # cm = np.concatenate(frame2trans_rotvec(f=cm), axis=-1)
    x = np.hstack((dh[x_bool_dict['dh_bool']].ravel(),
                   el[x_bool_dict['el_bool']].ravel(),
                   ma[x_bool_dict['ma_bool']].ravel(),
                   cm[x_bool_dict['cm_bool']].ravel(),
                   ad[x_bool_dict['ad_bool']].ravel()))
    return x


def unwrap_x(x, cal_rob, add_nominal_offset=False):
    n, x_bool_dict = get_x_bool_dict(cal_rob=cal_rob)
    x_unwrapper = create_x_unwrapper(**x_bool_dict)
    x = x_unwrapper(x)
    if add_nominal_offset:
        x = offset_nominal_parameters(cal_rob=cal_rob, **x)
        x = dict(dh=x[0], el=x[1], ma=x[2], cm=x[3], ad=x[4])
    return x


def create_x_unwrapper(cm_bool, dh_bool, el_bool, ma_bool, ad_bool,
                       update_dict=None):

    def x_unwrapper(x):
        dh, j = __calibration_bool2number(cal_bool=dh_bool, idx0=0, x=x)
        el, j = __calibration_bool2number(cal_bool=el_bool, idx0=j, x=x)
        ma, j = __calibration_bool2number(cal_bool=ma_bool, idx0=j, x=x)
        cm, j = __calibration_bool2number(cal_bool=cm_bool, idx0=j, x=x)
        ad, j = __calibration_bool2number(cal_bool=ad_bool, idx0=j, x=x)

        d = dict(dh=dh, el=el, ma=ma, cm=cm, ad=ad)
        if update_dict:
            d = update_dict(d)

        return d

    return x_unwrapper


def offset_nominal_parameters(cal_rob,
                              dh, el, ma, cm, ad):
    # Update nominal values of robot
    dh2 = cal_rob.dh + dh
    el2 = cal_rob.el + el
    ma2 = cal_rob.ma + ma
    cm2 = cal_rob.cm @ spatial.trans_rotvec2frame(trans=cm[:, :3], rotvec=cm[:, 3:])
    ad2 = ad

    return dh2, el2, ma2, cm2, ad2


def print_parameter(x):
    assert isinstance(x, dict)
    print(f'self.dh = np.{repr(x["dh"])}')
    print(f'self.el = np.{repr(x["el"])}')
    print(f'self.ma = np.{repr(x["ma"])}')