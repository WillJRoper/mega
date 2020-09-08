import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, 'core/')
import utilities
import h5py
import seaborn as sns


sns.set_style("whitegrid")


def directprogdeschist():
    """ A function that extracts the number of progenitors and descendants for all halos and
    produces two histograms, one for the number of progenitors and one for the number of
    descendants.

    :return: None
    """

    # Initialise the arrays to store the number of progenitors and descendants
    # *** NOTE: These arrays are initialised with considerably more entries than necessary (namely enough
    # entries for every particle to have a logarithmic mass growth), unused entries are removed after all values
    # have been computed.
    nprogs = []
    ndescs = []

    # Read the parameter file
    paramfile = sys.argv[1]
    inputs, flags, params = utilities.read_param(paramfile)

    # Load the snapshot list
    snaplist = list(np.loadtxt(inputs['snapList'], dtype=str))

    # Get the density rank
    density_rank = sys.argv[2]

    # Loop through Merger Graph data assigning each value to the relevant list
    for snap in snaplist:

        # Print the snapshot for progress tracking
        print(snap)

        # Create file to store this snapshots graph results
        if density_rank == 0:
            hdf = h5py.File(inputs['directgraphSavePath'] + 'Mgraph_' + snap + '.hdf5', 'r')
        else:
            hdf = h5py.File(inputs['directgraphSavePath'] + 'SubMgraph_' + snap + '.hdf5', 'r')

        # Get the number of progenitors and descendants
        nprogs.extend(hdf["nProgs"][...])
        ndescs.extend(hdf["nDescs"][...])

        hdf.close()

    # Histogram the results with the number of bins defined such that every number between 0 and
    # the max number of progenitors and descendants has a bin.
    prog_bins = int(np.max(nprogs))
    desc_bins = int(np.max(ndescs))
    Hprog, binedgesprog = np.histogram(nprogs, bins=prog_bins)
    Hdesc, binedgesdesc = np.histogram(ndescs, bins=desc_bins)

    # =============== Plot the results ===============

    # Set up figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the histograms
    ax.bar(binedgesprog[:-1] + 0.5, Hprog, color='r', width=1, label='Progenitor')
    ax.bar(binedgesdesc[:-1] + 0.5, Hdesc, color='r', width=1, label='Descendant')

    # Set y-axis scaling to logarithmic
    ax.set_yscale('log')

    # Label axes
    ax.set_xlabel(r'$N_{\mathrm{direct}}$')
    ax.set_ylabel(r'$N$')

    # Include legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    # Save the plot as a png
    plt.savefig('../plots/ProgDescNumberHist.png', dpi=fig.dpi)

    return


if __name__ == "__main__":

    directprogdeschist()
