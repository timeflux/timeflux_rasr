""" Configuration methods for rASR and Timeflux loading/testing
"""
from glob import glob
import os
import getpass


class Config:
    """Load user-dependent variable and path"""

    def __init__(self, username=None):
        # generate attribute
        self.known_users = {
            "raph": "/Users/raph/OMIND_SERVER/DATA/rASR Data/",
            "louis": "/Users/louis/Dropbox/rASR Data/",
            "sylchev": "/Users/sylchev/Dropbox/rASR Data/",
        }

        if username is None:
            self.username = getpass.getuser()
        else:
            self.username = username

        if self.username in self.known_users.keys():
            self.data_path = self.known_users[self.username]
        else:
            raise KeyError(
                username + ": User path not defined, please add path in utils.config."
            )

        self.results_folder = os.path.join(self.data_path, "results")

        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        self.raw_files = sorted(glob(os.path.join(self.data_path, "raw", "*xdf")))
        self.filtered_files = sorted(
            glob(os.path.join(self.data_path, "filtered", "*set"))
        )
        self.original_asr_out_files = sorted(
            glob(os.path.join(self.data_path, "original ASR out", "*set"))
        )
        self.riemannian_asr_out_files = sorted(
            glob(os.path.join(self.data_path, "Riemannian ASR out", "*set"))
        )
        self.calibration_files = sorted(
            glob(os.path.join(self.data_path, "calibration data", "*set"))
        )
