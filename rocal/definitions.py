import os
import platform
from wzk.dlr import DLR_USERSTORE, ICLOUD

PLATFORM_IS_LINUX = platform.system() == 'Linux'
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

if PLATFORM_IS_LINUX:
    ICHR20_CALIBRATION = DLR_USERSTORE + '/0_Data/Calibration/Results/Paper'
    ICHR22_AUTOCALIBRATION = f"{DLR_USERSTORE}/Paper/ICHR22_AutoCalibration/Data"

else:
    ICHR20_CALIBRATION = f'/Users/jote/{ICLOUD}/Paper/ICHR20_Calibration/Data'
    ICHR22_AUTOCALIBRATION = f'/Users/jote/{ICLOUD}/Paper/ICHR22_AutoCalibration/Data'

ICHR20_CALIBRATION_FIGS = ICHR20_CALIBRATION + '/Plots'
ICHR22_AUTOCALIBRATION_FIGS = ICHR22_AUTOCALIBRATION + '/Plots'
