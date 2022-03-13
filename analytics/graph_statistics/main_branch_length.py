import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import h5py
import sys
import seaborn as sns

sns.set_context("paper")
sns.set_style('whitegrid')


def mainbranchlengthDMLJ(datapath, basename, snaplist):
    """ A function that walks all z=0 halos' main branches computing the main branch length for each.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param cutoff: The halo mass cutoff in number of particles. Halos under this mass threshold are skipped.

    :return: lengths: An array of main branch lengths for all halos above the cutoff.
    """

    # Open z=0 direct graph file
    hdf = h5py.File(datapath + basename + snaplist[-1] + ".hdf5", "r")
    size = len(hdf['halo_IDs'])

    # Initialise the lengths array
    lengths = np.zeros(size, dtype=int)

    # Get the z=0 halo ids and masses
    z0halos = hdf['halo_IDs'][:]
    z0nparts = hdf['nparts'][:]

    hdf.close()

    # Loop over the z=0 halos walking down the main branch
    for z0halo in z0halos:

        print(z0halo, "of", size, end="\r")

        # Initialise the length counter, nprog, halo pointer and snapshot
        length = 0
        nprog = 100
        halo = z0halo
        snapshot = str(int(snaplist[-1]))

        # Loop until a halo with no progenitors is found
        while nprog > 0:

            # Open current snapshot
            hdf = h5py.File(datapath + basename + snapshot.zfill(4) + ".hdf5", "r")

            # Extract number of progenitors and start index
            nprog = hdf["nProgs"][halo]
            if nprog > 0:
                prog_start = hdf["prog_start_index"][halo]

                # Extract the main progenitor
                prog = hdf["Prog_haloIDs"][prog_start]

                # Move halo pointer to progenitor
                halo = prog

                # Compute the progenitor snapshot ID
                snapshot = str(int(snapshot) - 1)

                # Increment the length counter
                length += 1

            hdf.close()

        # Assign the main branch length to the lengths array
        lengths[z0halo] = length

    return lengths, z0nparts


def mainBranchLengthCompPlot():
    """ A function which walks the main branches of any algorithms with data in the supplied directory with the
    correct format (during this project this was SMT comparison project algorithms) and the main branches produced
    by DMLumberJack, produces histograms in 3 mass bins (20<=M<100, 100<=M<1000, 100<=M) and produces
    a plot of all 3 bins comparing algorithms.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param SMTtreepath: The file path for the SMT algorithm data files.
    :param cutoff: The halo mass cutoff in number of particles. Halos under this mass threshold are skipped.

    :return: None
    """

    # Get commandline paths
    datapath = sys.argv[1]
    basename = sys.argv[2]
    snapfile = sys.argv[3]
    outpath = sys.argv[4]

    # Load the snapshot list
    snaplist = list(np.loadtxt(snapfile, dtype=str))

    # Create lists of lower and upper mass thresholds for histograms
    low_threshs = [0, 100, 1000]
    up_threshs = [100, 1000, np.inf]

    # Compute DMLumberJack's main branch lengths
    lengths, nparts = mainbranchlengthDMLJ(datapath, basename, snaplist)

    bin_edges = np.arange(0, int(snaplist[-1]), 1)

    lims = {}

    # =============== Plot the results ===============

    # Set up figure
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=3, ncols=1)
    gs.update(wspace=0.5, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    for ax, low, up in zip([ax3, ax2, ax1], low_threshs, up_threshs):

        okinds = np.logical_and(nparts >= low, nparts < up)
        ls = lengths[okinds]

        H, _ = np.histogram(ls, bins=bin_edges)

        ax.bar(bin_edges[:-1] + 0.5, H, width=1, alpha=0.8,
               color="r", edgecolor="r")

        lims[up] = H.max()

    # Label axes
    ax3.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$N$')

    # Annotate the mass bins
    ax1.text(0.05, 0.8, r'$M_{H}>1000$',
                 bbox=dict(boxstyle="round,pad=0.3", fc='w',
                           ec="k", lw=1, alpha=0.8),
                 transform=ax1.transAxes,
                 horizontalalignment='left')
    ax2.text(0.05, 0.8, r'$1000\geq M_{H}>100$',
                 bbox=dict(boxstyle="round,pad=0.3", fc='w',
                           ec="k", lw=1, alpha=0.8),
                 transform=ax2.transAxes,
                 horizontalalignment='left')
    ax3.text(0.05, 0.8, r'$100\geq M_{H}>20$',
                 bbox=dict(boxstyle="round,pad=0.3", fc='w',
                           ec="k", lw=1, alpha=0.8),
                 transform=ax3.transAxes,
                 horizontalalignment='left')

    # Remove x axis from upper subplots
    ax1.tick_params(axis='x', bottom=False, left=False)
    ax2.tick_params(axis='x', bottom=False, left=False)

    # Set y axis limits such that 0 is removed from the upper two subplots to avoid tick stacking
    ax1.set_ylim(0.5, None)
    ax2.set_ylim(0.5, None)

    # Save figure with a transparent background
    fig.savefig(outpath + 'mainbranchlengthcomp.png', bbox_inches="tight")


mainBranchLengthCompPlot()
