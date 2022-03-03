import numpy as np
from struct import unpack
import matplotlib.pyplot as plt
import datetime
import matplotlib

def draw_3d_hist(files_path_prefix, sample_x, sample_y, time_start, time_end, postfix=''):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    n_bins = 15
    hist, xedges, yedges = np.histogram2d(sample_x, sample_y, bins=n_bins)
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 70 * np.ones_like(zpos)
    dz = hist.ravel()

    cmap = matplotlib.cm.get_cmap('jet').copy()
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=rgba, alpha=0.6)

    date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
    plt.title("Joint distribution of sensible and latent fluxes\n" +
              f"{date_start.strftime('%Y-%m-%d')} - {date_end.strftime('%Y-%m-%d')}")
    plt.xlabel("Sensible")
    plt.ylabel("Latent")
    plt.savefig(files_path_prefix + f'Distributions/sens-lat_histogram_{postfix}.png')
    return
