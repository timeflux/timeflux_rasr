"""Utils for visualization of data and comparison
"""
import matplotlib.pyplot as plt
import os
import math
import numpy as np


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

    title_dist = title+"_dist"

    data = raw.get_data()
    data = np.log(((1e6 * data) ** 2))  # log-normal instantaneous power
    Ne = data.shape[0]
    nb_columns = 6
    nb_rows = math.ceil(Ne/nb_columns)
    fig, ax = plt.subplots(nb_rows, nb_columns, sharex=True, sharey=True,gridspec_kw={'hspace': 0, 'wspace': 0})

    for n in range(Ne):
        n_row = math.floor(n / nb_columns)
        n_column = n - (n_row * nb_columns)
        ax[n_row, n_column].hist(data[n, :], 40, histtype='stepfilled', color='gray')
        ax[n_row, n_column].set_xlim([-3, 10])
        ax[n_row, n_column].set_xlabel(r"log(µV$^2$)")
    plt.savefig(
        os.path.join(output_folder, title_dist),
        dpi=192, show=False)
    plt.close()

