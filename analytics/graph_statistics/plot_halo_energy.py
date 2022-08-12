import numpy as np
import matplotlib.pyplot as plt
import sys
from mega.core.param_utils import read_param
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

    # Initialise the arrays to store the energies
    total_KE = []
    total_int_nrg = []
    total_GE = []
    mass = []
    npart = []

    # Read the parameter file
    paramfile = sys.argv[1]
    inputs, flags, params, cosmology, simulation = read_param(paramfile)

    # Load the snapshot list
    snaplist = list(np.loadtxt(inputs["snapList"], dtype=str))
    
    halo_sub = int(sys.argv[2])

    real = int(sys.argv[3])

    # Loop through Merger Graph data assigning each value to the relevant list
    for snap in snaplist:

        if len(sys.argv) > 3:
            if snap != snaplist[int(sys.argv[4])]:
                continue

        print(snap)

        # Create file to store this snapshots graph results
        hdf = h5py.File(inputs["haloSavePath"] + "halos_" + str(snap) + ".hdf5", "r")

        if halo_sub == 0:

            # Get the number of progenitors and descendants
            KE = 10 ** hdf["halo_kinetic_energies"][...]
            int_nrg = hdf["halo_thermal_energy"][...]
            GE = 10 ** hdf["halo_gravitational_energies"][...]
            m = hdf["masses"][...] * 10 ** 10
            n = hdf["nparts"][:, 1]
            if real:
                reals = hdf["real_flag"][...]
            else:
                reals = np.ones(m.size, dtype=bool)
            total_KE.extend(KE[reals])
            total_int_nrg.extend(int_nrg[reals])
            total_GE.extend(GE[reals])
            mass.extend(m[reals])
            npart.extend(n[reals])

        else:

            # Get the number of progenitors and descendants
            KE = 10 ** hdf["Subhalos"]["halo_kinetic_energies"][...]
            int_nrg = hdf["Subhalos"]["halo_thermal_energy"][...]
            GE = 10 ** hdf["Subhalos"]["halo_gravitational_energies"][...]
            m = hdf["Subhalos"]["masses"][...] * 10 ** 10
            n = hdf["Subhalos"]["nparts"][...]
            if real:
                reals = hdf["Subhalos"]["real_flag"][...]
            else:
                reals = np.ones(m.size, dtype=bool)
            total_KE.extend(KE[reals])
            total_int_nrg.extend(int_nrg[reals])
            total_GE.extend(GE[reals])
            mass.extend(m[reals])
            npart.extend(n[reals, 1])

        hdf.close()

        try:

            # Set up plot
            fig = plt.figure(figsize=(8, 6))
            axtwin = fig.add_subplot(111)
            ax = axtwin.twiny()

            # Plot data
            axtwin.scatter(m, (KE + int_nrg) / GE,
                           facecolors="none", edgecolors="none")
            cbar = ax.hexbin(n, (KE + int_nrg) / GE,
                             gridsize=50, xscale="log", yscale="log", mincnt=1,
                             norm=LogNorm(), linewidths=0.2)
            ax.axhline(1.0, linestyle="--", color="k", label="$E=0$")
            ax.axhline(0.5, linestyle="--", color="g", label="$2KE=|GE|$")

            axtwin.grid(False)
            axtwin.grid(True)

            # Label axes
            ax.set_ylabel(r"$(\mathrm{KE} + U)/|\mathrm{GE}|$")
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
                fig.savefig(inputs["profile_plot_path"] + "energy_plots/halo_energies_" + snap + ".png", bbox_inches="tight")
            else:
                fig.savefig(inputs["profile_plot_path"] + "energy_plots/subhalo_energies_" + snap + ".png", bbox_inches="tight")

            plt.close(fig)

            # Set up plot
            fig = plt.figure(figsize=(8, 6))
            axtwin = fig.add_subplot(111)
            ax = axtwin.twiny()

            # Plot data
            axtwin.scatter(m, KE / GE,
                           facecolors="none", edgecolors="none")
            cbar = ax.hexbin(n, KE / GE,
                             gridsize=50, xscale="log", yscale="log", mincnt=1,
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
                fig.savefig(inputs["profile_plot_path"] + "energy_plots/halo_virialenergies_" + snap + ".png", bbox_inches="tight")
            else:
                fig.savefig(inputs["profile_plot_path"] + "energy_plots/subhalo_virialenergies_" + snap + ".png", bbox_inches="tight")

            plt.close(fig)

        except ValueError as e:
            print(e)
            continue

    total_KE = np.array(total_KE)
    total_int_nrg = np.array(total_int_nrg)
    total_GE = np.array(total_GE)
    mass = np.array(mass)
    npart = np.array(npart)

    ratio = (total_KE + total_int_nrg) / total_GE
    print("There are", ratio[ratio > 1].size, "unbound halos")

    # Set up plot
    fig = plt.figure(figsize=(8, 6))
    axtwin = fig.add_subplot(111)
    ax = axtwin.twiny()

    # Plot data
    axtwin.scatter(mass, ratio, facecolors="none", edgecolors="none")
    cbar = ax.hexbin(npart, ratio, gridsize=50, xscale="log", yscale="log",
                     mincnt=1, norm=LogNorm(), linewidths=0.2)
    ax.axhline(1.0, linestyle="--", color="k", label="$E=0$")
    ax.axhline(0.5, linestyle="--", color="g", label="$2KE=|GE|$")

    axtwin.grid(False)
    axtwin.grid(True)

    # Label axes
    axtwin.set_ylabel(r"$(\mathrm{KE} + U)/|\mathrm{GE}|$")
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
        fig.savefig(inputs["profile_plot_path"] + "energy_plots/halo_energies.png", bbox_inches="tight")
    else:
        fig.savefig(inputs["profile_plot_path"] + "energy_plots/subhalo_energies.png", bbox_inches="tight")

    plt.close(fig)

    ratio = total_KE / total_GE

    # Set up plot
    fig = plt.figure(figsize=(8, 6))
    axtwin = fig.add_subplot(111)
    ax = axtwin.twiny()

    # Plot data
    axtwin.scatter(mass, ratio, facecolors="none", edgecolors="none")
    cbar = ax.hexbin(npart, ratio, gridsize=50, xscale="log", yscale="log",
                     mincnt=1, norm=LogNorm(), linewidths=0.2)
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
        fig.savefig(
            inputs["profile_plot_path"] + "energy_plots/halo_virialenergies.png",
            bbox_inches="tight")
    else:
        fig.savefig(
            inputs["profile_plot_path"] + "energy_plots/subhalo_virialenergies.png",
            bbox_inches="tight")

    plt.close(fig)

    return


if __name__ == "__main__":

    energyplot()
