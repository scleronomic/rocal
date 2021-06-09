import os
import platform
from wzk.dlr import DLR_USERSTORE

PLATFORM_IS_LINUX = platform.system() == 'Linux'
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

if PLATFORM_IS_LINUX:
    ICHR20_CALIBRATION_DATA = DLR_USERSTORE + '0_Data/Calibration'
    ICHR20_CALIBRATION = DLR_USERSTORE + '/0_Data/Calibration/Results/Paper'
    ICHR20_CALIBRATION_FIGS = ICHR20_CALIBRATION + '/Plots/Final/'

else:
    ICHR20_CALIBRATION = '/Users/jote/Documents/DLR/Data/Calibration'
    ICHR20_CALIBRATION_DATA = '/Users/jote/Documents/DLR/Data/Calibration'
    ICHR20_CALIBRATION_FIGS = ICHR20_CALIBRATION + '/Plots'
