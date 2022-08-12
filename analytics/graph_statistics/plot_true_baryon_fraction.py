import numpy as np
import matplotlib.pyplot as plt
import sys
from mega.core.param_utils import read_param
from matplotlib.colors import LogNorm
import h5py
import seaborn as sns


sns.set_style("whitegrid")


def bary_frac_plot():
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

    # Apply realness flags flag
    real = bool(sys.argv[3])

    # Define mass bins
    bins = np.logspace(-4, 2, 50)
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

            # Get halo masses
            m_dm = hdf["part_type_masses"][:, 1] * 10 ** 10
            m_gas = hdf["part_type_masses"][:, 0] * 10 ** 10
            m_star = hdf["part_type_masses"][:, 4] * 10 ** 10
            m_bh = hdf["part_type_masses"][:, 5] * 10 ** 10
            m_bary = m_gas + m_star + m_bh
            if real:
                reals = hdf["real_flag"][...]
            else:
                reals = np.ones(len(m_dm), dtype=bool)

            # Apply realness flags
            bary_frac = m_bary[reals] / (m_dm[reals] + m_gas[reals] 
                                         + m_star[reals] + m_bh[reals])

        else:
            
            # Get halo masses
            m_dm = hdf["Subhalos"]["part_type_masses"][:, 1] * 10 ** 10
            m_gas = hdf["Subhalos"]["part_type_masses"][:, 0] * 10 ** 10
            m_star = hdf["Subhalos"]["part_type_masses"][:, 4] * 10 ** 10
            m_bh = hdf["Subhalos"]["part_type_masses"][:, 5] * 10 ** 10
            m_bary = m_gas + m_star + m_bh
            if real:
                reals = hdf["Subhalos"]["real_flag"][...]
            else:
                reals = np.ones(len(m_dm), dtype=bool)

            # Apply realness flags
            bary_frac = m_bary[reals] / (m_dm[reals] + m_gas[reals] 
                                         + m_star[reals] + m_bh[reals])

        hdf.close()

        try:

            # Histogram masses
            H, _ = np.histogram(bary_frac, bins=bins)
            
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
            ax.plot(bin_cents, H)

            # Label axes
            ax.set_ylabel(r"$N$")
            ax.set_xlabel(r"$f_{\mathrm{bary}}$")

            # Set scale
            ax.set_xscale("log")
            ax.set_yscale("log")

            # Set limits
            ax.set_xlim(low_lim, high_lim)

            # Save figure
            if halo_sub == 0:
                fig.savefig(inputs["profile_plot_path"]
                            + "baryon_fraction/"
                              "host_bary_frac_%s_HaloType%d.png"
                            % (snap, halo_sub),
                            bbox_inches="tight")
            else:
                fig.savefig(inputs["profile_plot_path"]
                            + "baryon_fraction/"
                              "subhalo_bary_frac_%s_HaloType%d.png"
                            % (snap, halo_sub),
                            bbox_inches="tight")

            plt.close(fig)

        except ValueError as e:
            print(e)
            continue

if __name__ == "__main__":

    bary_frac_plot()
