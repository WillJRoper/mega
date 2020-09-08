import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import h5py
import time
import pickle


def mainbranchlengthDMLJ(tree_data, cutoff=None):
    """ A function that walks all z=0 halos' main branches computing the main branch length for each.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param cutoff: The halo mass cutoff in number of particles. Halos under this mass threshold are skipped.

    :return: lengths: An array of main branch lengths for all halos above the cutoff.
    """

    # Initialise the loop count
    count = -1

    # Initialise the previous progress
    previous_progress = -1

    # Assign the number of halos to a variable for progress reporting and assigning the lengths array size
    size = len(tree_data['061'].keys())

    # Initialise the lengths array
    lengths = np.zeros(size, dtype=int)

    # Loop over the z=0 halos walking down the main branch
    for ind, z0halo in enumerate(tree_data['061'].keys()):

        # Increment the loop counter
        count += 1

        # Ignore z0 halos with mass less than the cutoff if one is provided
        if cutoff != None:
            if tree_data['061'][z0halo]['current_halo_nPart'] < cutoff: continue

        # Compute and print the progress
        progress = int(count / size * 100)
        if progress != previous_progress:  # only print if the integer progress differs from the last printed value
            print('{x}: '.format(x='DMLumberJack'), progress, '%')
        previous_progress = progress

        # Extract the necessary progenitor data for this halo
        nprog = tree_data['061'][z0halo]['nProg']  # number of progenitors
        prog_haloids = tree_data['061'][z0halo]['Prog_haloIDs']  # progenitor IDs

        # Assign the z0 snapshot ID to a variable
        snapshot = '061'

        # Initialise the length counter
        length = 0

        # Loop until a halo with no progenitors is found
        while nprog != 0:

            # Increment the length counter
            length += 1

            # Compute the progenitor snapshot ID
            if int(snapshot) > 10:
                snapshot = '0' + str(int(snapshot) - 1)
            else:
                snapshot = '00' + str(int(snapshot) - 1)

            # Find the next step in main branch
            main = prog_haloids[0]

            # Extract the necessary progenitor data for this next main branch halo
            nprog = tree_data[snapshot][str(main)]['nProg']
            prog_haloids = tree_data[snapshot][str(main)]['Prog_haloIDs']

        # Assign the main branch length to the lengths array
        lengths[ind] = length

    return lengths


def mainBranchLengthCompPlot(tree_data, SMTtreepath, cutoff=None):
    """ A function which walks the main branches of any algorithms with data in the supplied directory with the
    correct format (during this project this was SMT comparison project algorithms) and the main branches produced
    by DMLumberJack, produces histograms in 3 mass bins (20<=M<100, 100<=M<1000, 100<=M) and produces
    a plot of all 3 bins comparing algorithms.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param SMTtreepath: The file path for the SMT algorithm data files.
    :param cutoff: The halo mass cutoff in number of particles. Halos under this mass threshold are skipped.

    :return: None
    """

    # Create lists of lower and upper mass thresholds for histograms
    low_threshs = [0, 100, 1000]
    up_threshs = [100, 1000, np.inf]

    # Initialise results dictionary
    algorithm_hists = {}

    # Compute DMLumberJack's main branch lengths
    lengths = mainbranchlengthDMLJ(tree_data, cutoff)

    # Initialise this algorithm's dictionary
    algorithm_hists['DMLumberJack'] = {}

    # Initialise the halo masses array
    halo_masses = np.zeros(len(tree_data['061'].keys()), dtype=int)

    # Define bins for the histogram from snapshots
    bins = [snap for snap in range(1, 62)]

    # Loop over z0 halos
    for ind, halo in enumerate(tree_data['061'].keys()):
        # Extract each halo's mass and assign to the array
        halo_masses[ind] = tree_data['061'][halo]['current_halo_nPart']

    # Loop over each set of mass bin limits
    for low_thresh, up_thresh in zip(low_threshs, up_threshs):
        # Extract the main branch lengths that lie in this mass bin
        thresh_lengths = lengths[np.where(np.logical_and(halo_masses > low_thresh, halo_masses <= up_thresh))]

        # Initialise this thresholds histogram dictionary entry
        algorithm_hists['DMLumberJack'][up_thresh] = {}

        # Compute and assign the histogram data to the dictionary
        hist = np.histogram(thresh_lengths, bins=bins)
        algorithm_hists['DMLumberJack'][up_thresh]['H'] = hist[0]
        algorithm_hists['DMLumberJack'][up_thresh]['bin_edges'] = hist[1]

    # Find the filepath for each 'other' algorithm's data at provided destination
    treefiles = []
    for root, dirs, files in os.walk(SMTtreepath):
        for name in files:
            treefiles.append(os.path.join(root, name))

    # Compute main branch length and histogram for each SMT algorithm
    # Loop through the found filepaths
    for treefile in iter(treefiles):

        # Get the progenitor information from this algorithm if the filepath found above is valid (can
        # find hidden files that are not desired depending on directory location)
        try:
            algorithm, tot_nprog, halonprogs, halo_progs = opentree(treefile)
        except UnicodeDecodeError:
            continue

        # Consistent trees can't be included since it introduces halo IDs of its own and therefore the mass
        # of these halos cannot be extracted.
        if algorithm == 'ConsistentTrees': continue

        # Compute the main branch lengths for this algorithm
        lengths, halo_masses = mainbranchlengthSMT(algorithm, tot_nprog, halonprogs, halo_progs)

        # Initialise this algorithm's dictionary
        algorithm_hists[algorithm] = {}

        # Loop over each set of mass bin limits
        for low_thresh, up_thresh in zip(low_threshs, up_threshs):
            # Extract the main branch lengths that lie in this mass bin
            thresh_lengths = lengths[np.where(np.logical_and(halo_masses > low_thresh, halo_masses <= up_thresh))]

            # Initialise this thresholds histogram dictionary entry
            algorithm_hists[algorithm][up_thresh] = {}

            # Compute and assign the histogram data to the dictionary
            hist = np.histogram(thresh_lengths, bins=bins)
            algorithm_hists[algorithm][up_thresh]['H'] = hist[0]
            algorithm_hists[algorithm][up_thresh]['bin_edges'] = hist[1]

    # =============== Plot the results ===============

    # Set up figure
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.5, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, : 2])
    ax2 = fig.add_subplot(gs[1, : 2])
    ax3 = fig.add_subplot(gs[2, : 2])

    # Create a sacrificial subplot for the legend
    ax4 = fig.add_subplot(gs[1, 2])

    # Loop through algorithms plotting the results using a different color for each algorithm
    # All 'other' algorithms are plotted with lines and DMLJ is plotted as a bar chart
    for algorithm, color in zip(algorithm_hists.keys(),
                                plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(list(algorithm_hists.keys()))]):

        if algorithm == 'DMLumberJack':
            ax1.bar(algorithm_hists[algorithm][np.inf]['bin_edges'][:-1] + 0.5,
                    algorithm_hists[algorithm][np.inf]['H'] + 1,
                    label=algorithm, color=color, width=1, alpha=0.8)
            ax2.bar(algorithm_hists[algorithm][1000]['bin_edges'][:-1] + 0.5, algorithm_hists[algorithm][1000]['H'] + 1,
                    label=algorithm, color=color, width=1, alpha=0.8)
            ax3.bar(algorithm_hists[algorithm][100]['bin_edges'][:-1] + 0.5, algorithm_hists[algorithm][100]['H'] + 1,
                    label=algorithm, color=color, width=1, alpha=0.8)

        else:
            ax1.plot(algorithm_hists[algorithm][np.inf]['bin_edges'][:-1] + 0.5,
                     algorithm_hists[algorithm][np.inf]['H'] + 1,
                     label=algorithm, color=color)
            ax2.plot(algorithm_hists[algorithm][1000]['bin_edges'][:-1] + 0.5,
                     algorithm_hists[algorithm][1000]['H'] + 1,
                     label=algorithm, color=color)
            ax3.plot(algorithm_hists[algorithm][100]['bin_edges'][:-1] + 0.5, algorithm_hists[algorithm][100]['H'] + 1,
                     label=algorithm, color=color)

    # Label axes
    ax3.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$N+1$')

    # Annotate the mass bins
    ax1.annotate(r'$M_{H}>1000$', (1, algorithm_hists['DMLumberJack'][np.inf]['H'].max() * 0.85),
                 bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
    ax2.annotate(r'$1000>=M_{H}>100$', (1, algorithm_hists['DMLumberJack'][1000]['H'].max() * 0.85),
                 bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
    ax3.annotate(r'$100>=M_{H}>20$', (1, algorithm_hists['DMLumberJack'][100]['H'].max() * 0.85),
                 bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))

    # Include legend
    handles, labels = ax1.get_legend_handles_labels()
    ax4.legend(handles, labels)

    # Remove axis from ax4
    ax4.set_axis_off()

    # Remove x axis from upper subplots
    ax1.xaxis.set_visible(False)
    ax2.xaxis.set_visible(False)

    # Set y axis limits such that 0 is removed from the upper two subplots to avoid tick stacking
    ax1.set_ylim(0.5, None)
    ax2.set_ylim(0.5, None)

    # Invoke tight layout to further enforce next subplot stacking
    plt.tight_layout()

    # Save figure with a transparent background
    plt.savefig('Merger Graph Statistics/mainbranchlengthcomp.png', dpi=600, transparent=True)
