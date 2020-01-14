"""Utils for visualization of data and comparison
"""
import matplotlib.pyplot as plt
import os


def plot_all_mne_data(raw, output_folder=None, title=None):
    """Standardized function to give save all comparison

    TODO: implement here qualitative examination of the data
    """

    title_raw = title+"_raw"
    raw.plot(title=title_raw).savefig(
        os.path.join(output_folder, title_raw),
        dpi=192, show=False)
    plt.close()

    title_psd = title+"_psd"

    raw.plot_psd().savefig(
        os.path.join(output_folder, title_psd),
        dpi=192, show=False)
    plt.close()
