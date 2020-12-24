import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, "core/")
import utilities
from matplotlib.colors import LogNorm
import h5py
import seaborn as sns


sns.set_style("whitegrid")


def energyplot():
    """ A function that extracts the number of progenitors and descendants for all halos and
    produces two histograms, one for the number of progenitors and one for the number of
    descendants.

    :return: None
    """

    # Initialise the arrays to store the number of progenitors and descendants
    # *** NOTE: These arrays are initialised with considerably more entries than necessary (namely enough
    # entries for every particle to have a logarithmic mass growth), unused entries are removed after all values
    # have been computed.
    total_KE = []
    total_GE = []
    mass = []

    # Read the parameter file
    paramfile = sys.argv[1]
    inputs, flags, params, _ = utilities.read_param(paramfile)

    # Load the snapshot list
    snaplist = list(np.loadtxt(inputs["snapList"], dtype=str))
    
    halo_sub = int(sys.argv[2])
    
    pmass = 1

    # Loop through Merger Graph data assigning each value to the relevant list
    for snap in snaplist:

        print(snap)

        # Create file to store this snapshots graph results
        hdf = h5py.File(inputs["haloSavePath"] + "halos_" + str(snap) + ".hdf5", "r")
        
        pmass = hdf.attrs["part_mass"]

        if halo_sub == 0:

            # Get the number of progenitors and descendants
            KE = hdf["halo_kinetic_energies"][...]
            GE = hdf["halo_gravitational_energies"][...]
            m = hdf["nparts"][...] * pmass
            reals = hdf["real_flag"][...]
            total_KE.extend(KE[reals])
            total_GE.extend(GE[reals])
            mass.extend(m[reals])

        else:

            # Get the number of progenitors and descendants
            KE = hdf["Subhalos"]["halo_kinetic_energies"][...]
            GE = hdf["Subhalos"]["halo_gravitational_energies"][...]
            m = hdf["Subhalos"]["nparts"][...] * pmass
            reals = hdf["Subhalos"]["real_flag"][...]
            total_KE.extend(KE[reals])
            total_GE.extend(GE[reals])
            mass.extend(m[reals])

        hdf.close()

        try:

            # Set up plot
            fig = plt.figure(figsize=(8, 6))
            axtwin = fig.add_subplot(111)
            ax = axtwin.twiny()

            # Plot data
            axtwin.scatter(m, KE / GE, facecolors="none", edgecolors="none")
            cbar = ax.hexbin(m / pmass, KE / GE, gridsize=50, xscale="log", yscale="log", mincnt=1,
                             norm=LogNorm(), linewidths=0.2)
            ax.axhline(1.0, linestyle="--", color="k", label="$E=0$")
            ax.axhline(0.5, linestyle="--", color="g", label="$2KE=|GE|$")

            axtwin.grid(False)
            axtwin.grid(True)

            # Label axes
            ax.set_ylabel(r"$\mathrm{KE}/|\mathrm{GE}|$")
            axtwin.set_xlabel(r"$M_{h} / M_{\odot}$")
            ax.set_xlabel(r"$N_{p}$")

            # Set scale
            ax.set_xscale("log")
            axtwin.set_xscale("log")
            ax.set_yscale("log")

            # Set limits
            ax.set_ylim(10 ** -1, None)

            # Get and draw legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)

            fig.colorbar(cbar, ax=ax)

            # Save figure
            if halo_sub == 0:
                fig.savefig("analytics/energy_plots/halo_energies_" + snap + ".png", bbox_inches="tight")
            else:
                fig.savefig("analytics/energy_plots/subhalo_energies_" + snap + ".png", bbox_inches="tight")

            plt.close(fig)

        except ValueError:
            continue

    total_KE = np.array(total_KE)
    total_GE = np.array(total_GE)
    mass = np.array(mass)

    ratio = total_KE / total_GE
    print("There are", ratio[ratio > 1].size, "unbound halos")

    # Set up plot
    fig = plt.figure(figsize=(8, 6))
    axtwin = fig.add_subplot(111)
    ax = axtwin.twiny()

    # Plot data
    axtwin.scatter(mass, ratio, facecolors="none", edgecolors="none")
    cbar = ax.hexbin(mass / pmass, ratio, gridsize=50, xscale="log", yscale="log", mincnt=1,
                     norm=LogNorm(), linewidths=0.2)
    ax.axhline(1.0, linestyle="--", color="k", label="$E=0$")
    ax.axhline(0.5, linestyle="--", color="g", label="$2KE=|GE|$")

    axtwin.grid(False)
    axtwin.grid(True)

    # Label axes
    axtwin.set_ylabel(r"$\mathrm{KE}/|\mathrm{GE}|$")
    axtwin.set_xlabel(r"$M_{h} / M_{\odot}$")
    ax.set_xlabel(r"$N_{p}$")

    # Set scale
    ax.set_xscale("log")
    axtwin.set_xscale("log")
    ax.set_yscale("log")

    # Set limits
    ax.set_ylim(10 ** -1, None)

    # Get and draw legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.colorbar(cbar, ax=ax)

    # Save figure
    if halo_sub == 0:
        fig.savefig("analytics/energy_plots/halo_energies.png", bbox_inches="tight")
    else:
        fig.savefig("analytics/energy_plots/subhalo_energies.png", bbox_inches="tight")

    return


if __name__ == "__main__":

    energyplot()
