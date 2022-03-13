import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, "core/")
sys.path.insert(1, "../mega_global_phase/core/")
import utilities
from matplotlib.colors import LogNorm
import h5py
import seaborn as sns


sns.set_style("whitegrid")


def alpha_v_plot():
    """ A function that extracts the number of progenitors and descendants for all halos and
    produces two histograms, one for the number of progenitors and one for the number of
    descendants.

    :return: None
    """

    # Initialise the lists to store everything
    alpha_hosts = []
    alpha_subs = []
    mass_hosts = []
    mass_subs = []

    # Read the parameter file
    paramfile = sys.argv[1]
    inputs, flags, params, cosmology, simulation = utilities.read_param(paramfile)

    # Load the snapshot list
    snaplist = list(np.loadtxt(inputs["snapList"], dtype=str))
    
    halo_sub = int(sys.argv[2])

    # Define bins
    bins = np.linspace(params["min_alpha_v"], params["ini_alpha_v"], 25)
    bin_cents = (bins[:-1] + bins[1:]) / 2

    # Loop through Merger Graph data assigning each value to the relevant list
    for snap in snaplist:

        if len(sys.argv) > 3:
            if snap != snaplist[int(sys.argv[3])]:
                continue

        print(snap)

        # Create file to store this snapshots graph results
        hdf = h5py.File(inputs["haloSavePath"] + "halos_" + str(snap) + ".hdf5", "r")

        # Get the data
        mass_host = hdf["total_masses"][...]
        alpha_host = hdf['exit_vlcoeff'][...]
        mass_hosts.extend(mass_host)
        alpha_hosts.extend(alpha_host)

        if halo_sub:

            # Get the data
            mass_sub = hdf["Subhalos"]["total_masses"][...]
            alpha_sub = hdf["Subhalos"]['exit_vlcoeff'][...]
            mass_subs.extend(mass_sub)
            alpha_subs.extend(alpha_sub)

        else:
            mass_sub = None
            alpha_sub = None

        hdf.close()

        try:

            # Set up plot
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

            H, _ = np.histogram(alpha_host, bins=bins)
            print(H)
            print(np.unique(alpha_host).size)

            # Plot data
            ax.plot(bin_cents, H, label="Host", color="r")

            if halo_sub:

                H, _ = np.histogram(alpha_sub, bins=bins)

                # Plot data
                ax.plot(bin_cents, H, label="Subhalo", color="b")

            # Label axes
            ax.set_ylabel(r"$N$")
            ax.set_xlabel(r"$\alpha_{v}$")

            # Set scale
            ax.set_yscale("log")

            # Get and draw legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)

            # Save figure
            fig.savefig(inputs["analyticPlotPath"] + "alpha_v_exit_" + snap + ".png", bbox_inches="tight")

            plt.close(fig)

        except ValueError as e:
            print(snap, e)
            continue

    if len(sys.argv) == 3:

        total_alpha_host = np.array(alpha_hosts)
        total_alpha_sub = np.array(alpha_subs)
        mass_hosts = np.array(mass_hosts)

if __name__ == "__main__":

    alpha_v_plot()
