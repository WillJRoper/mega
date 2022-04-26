import numpy as np
import matplotlib.pyplot as plt
import sys
from mega.core.param_utils import read_param
from matplotlib.colors import LogNorm
import h5py
import seaborn as sns


sns.set_style("whitegrid")


def mass_func_plot():
    """

    :return: None
    """

    # Read the parameter file
    paramfile = sys.argv[1]
    inputs, flags, params, cosmology, simulation = read_param(paramfile)

    # Load the snapshot list
    snaplist = list(np.loadtxt(inputs["snapList"], dtype=str))

    # Host or subhalo flag
    halo_sub = int(sys.argv[2])

    # Particle species flag
    part_type = int(sys.argv[3])

    # Apply realness flags flag
    real = bool(sys.argv[4])

    # Define mass bins
    bins = np.logspace(8, 15, 50)
    intervals = bins[1:] - bins[:-1]
    bin_cents = (bins[:-1] + bins[1:]) / 2

    # Loop through Merger Graph data assigning each value to the relevant list
    for snap in snaplist:

        print(snap)

        # Create file to store this snapshots graph results
        try:
            hdf = h5py.File(inputs["haloSavePath"] + "halos_" + str(snap)
                            + ".hdf5", "r")
        except OSError:
            continue

        # Get box volume
        boxsize = hdf.attrs["boxsize"]
        vol = np.product(boxsize)

        if halo_sub == 0:

            # Get halo mass
            m = hdf["part_type_masses"][:, part_type] * 10 ** 10
            if real:
                reals = hdf["real_flag"][...]
            else:
                reals = np.ones(len(m), dtype=bool)

            # Apply realness flags
            m = m[reals]

        else:

            # Get halo mass
            m = hdf["Subhalos"]["part_type_masses"][:, part_type] * 10 ** 10
            if real:
                reals = hdf["Subhalos"]["real_flag"][...]
            else:
                reals = np.ones(len(m), dtype=bool)

            # Apply realness flags
            m = m[reals]

        hdf.close()

        try:

            # Histogram masses
            H, _ = np.histogram(m, bins=bins)

            # Lets remove ott zeroed bins
            ind = 0
            count = 0
            while count < 1:
                low_lim = bins[ind]
                count = H[ind]
                ind += 1
            ind = len(H) - 1
            count = 0
            while count < 1:
                high_lim = bins[ind + 1]
                count = H[ind]
                ind -= 1

            # Set up plot
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # Plot data
            ax.plot(bin_cents, H / (vol * intervals))

            # Label axes
            ax.set_ylabel(r"$\phi / [\mathrm{Mpc}^{-3} M_\odot]$")
            ax.set_xlabel(r"$M_{%d}$" % part_type)

            # Set scale
            ax.set_xscale("log")
            ax.set_yscale("log")

            # Set limits
            ax.set_xlim(low_lim, high_lim)

            # Save figure
            if halo_sub == 0:
                fig.savefig(inputs["profile_plot_path"]
                            + "mass_function/host_mass_func_%s_PartType%d.png"
                            % (snap, part_type),
                            bbox_inches="tight")
            else:
                fig.savefig(inputs["profile_plot_path"]
                            + "mass_function/"
                              "subhalo_mass_func_%s_PartType%d.png"
                            % (snap, part_type),
                            bbox_inches="tight")

            plt.close(fig)

        except ValueError as e:
            print(e)
            continue

if __name__ == "__main__":

    mass_func_plot()
