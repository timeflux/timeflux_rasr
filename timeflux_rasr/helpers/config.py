"""Configuration methods for rASR and Timeflux loading/testing"""

from glob import glob
import os

#: Environment variable giving the folder of the rASR comparison data.
DATA_ENV = "TIMEFLUX_RASR_DATA"


class Config:
    """Paths of the rASR comparison data (raw, filtered, MATLAB outputs).

    Parameters
    ----------
    data_path : str | None
        Folder holding the ``raw``, ``filtered``, ``calibration data``,
        ``original ASR out`` and ``Riemannian ASR out`` subfolders. Defaults
        to the ``TIMEFLUX_RASR_DATA`` environment variable.
    """

    def __init__(self, data_path=None):
        if data_path is None:
            data_path = os.environ.get(DATA_ENV)
        if not data_path:
            raise ValueError(
                f"No data folder: pass data_path or set the {DATA_ENV} "
                "environment variable."
            )
        self.data_path = data_path

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
