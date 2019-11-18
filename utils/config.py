""" Configuration methods for rASR and Timeflux loading/testing
"""
from glob import glob
import os
import getpass

class Config():
    """Load user-dependent variable and path
    """
    username = getpass.getuser()
    data_path = []
    if username == "raph":
        data_path = '/Users/raph/OMIND_SERVER/DATA/rASR Data/'
    elif username == "louis":
        data_path = "/Users/louis/AlayaTec Dropbox/louis korczowski/rASR Data/"
    else:
        raise NameError(username + ": User path not defined")

    raw_files = glob(os.path.join(data_path, 'raw', '*xdf'))
    filtered_files = glob(os.path.join(data_path, 'filtered', '*set'))
    original_asr_out_files = glob(os.path.join(data_path, 'original ASR out', '*set'))
    riemannian_asr_out_files = glob(os.path.join(data_path, 'Riemannian ASR out', '*set'))
    calibration_files = glob(os.path.join(data_path, 'calibration data', '*set'))
