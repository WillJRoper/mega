import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, 'core/')
import utilities
import h5py
import seaborn as sns


sns.set_style("whitegrid")


def directprogdeschist(threshold=10):
    """ A function that extracts the number of progenitors and descendants for all halos and
    produces two histograms, one for the number of progenitors and one for the number of
    descendants.

    :return: None
    """

    # Initialise the arrays to store the number of progenitors and descendants
    # *** NOTE: These arrays are initialised with considerably more entries than necessary (namely enough
    # entries for every particle to have a logarithmic mass growth), unused entries are removed after all values
    # have been computed.
    total_nprogs = []
    total_ndescs = []

    # Read the parameter file
    paramfile = sys.argv[1]
    inputs, flags, params, _ = utilities.read_param(paramfile)

    # Load the snapshot list
    snaplist = list(np.loadtxt(inputs['snapList'], dtype=str))

    # Get the density rank
    density_rank = int(sys.argv[2])

    # Loop through Merger Graph data assigning each value to the relevant list
    for snap in snaplist:

        print(snap)

        # Create file to store this snapshots graph results
        if density_rank == 0:
            hdf = h5py.File(inputs['directgraphSavePath'] + 'Mgraph_' + snap + '.hdf5', 'r')
        else:
            hdf = h5py.File(inputs['directgraphSavePath'] + 'SubMgraph_' + snap + '.hdf5', 'r')

        if threshold == 10:

            # Get the number of progenitors and descendants
            nprogs = hdf["nProgs"][...]
            ndescs = hdf["nDescs"][...]
            total_nprogs.extend(nprogs[nprogs >= 0])
            total_ndescs.extend(ndescs[ndescs >= 0])

        else:

            nparts = hdf["nparts"][...]
            prog_start_index = hdf["prog_start_index"][...]
            desc_start_index = hdf["desc_start_index"][...]
            prog_npart = hdf["Prog_nPart"][...]
            desc_npart = hdf["Desc_nPart"][...]
            nprogs = hdf["nProgs"][...]
            ndescs = hdf["nDescs"][...]
            reals = hdf["real_flag"][...]

            for ind in range(len(nprogs)):

                if nparts[ind] < threshold or not reals[ind]:
                    continue

                pstart = prog_start_index[ind]
                dstart = desc_start_index[ind]
                nprog = nprogs[ind]
                ndesc = ndescs[ind]
                pnpart = prog_npart[pstart: pstart + nprog]
                dnpart = desc_npart[dstart: dstart + ndesc]

                pnpart = pnpart[pnpart >= threshold]
                dnpart = dnpart[dnpart >= threshold]

                # Get the number of progenitors and descendants
                total_nprogs.append(pnpart.size)
                total_ndescs.append(dnpart.size)

        hdf.close()

    # Histogram the results with the number of bins defined such that every number between 0 and
    # the max number of progenitors and descendants has a bin.
    prog_bins = int(np.max(total_nprogs))
    desc_bins = int(np.max(total_ndescs))
    Hprog, binedgesprog = np.histogram(total_nprogs, bins=prog_bins)
    Hdesc, binedgesdesc = np.histogram(total_ndescs, bins=desc_bins)

    # =============== Plot the results ===============

    # Set up figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the histograms
    ax.plot(binedgesprog[:-1] + 0.5, Hprog, color='palegreen', label='Progenitor')
    ax.plot(binedgesdesc[:-1] + 0.5, Hdesc, color='deepskyblue', label='Descendant')

    # Set y-axis scaling to logarithmic
    ax.set_yscale('log')

    # Label axes
    ax.set_xlabel(r'$N_{\mathrm{direct}}$')
    ax.set_ylabel(r'$N$')

    # Include legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    # Save the plot as a png
    if density_rank == 0:
        plt.savefig('analytics/plots/ProgDescNumberHist.png', dpi=fig.dpi)
    else:
        plt.savefig('analytics/plots/subProgDescNumberHist.png', dpi=fig.dpi)

    return


if __name__ == "__main__":

    directprogdeschist(threshold=10)
