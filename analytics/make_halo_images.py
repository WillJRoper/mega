import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo

from core.param_utils import read_param


def plot_halo(pos, prog_pos, desc_pos, ihalo, snap, boxsize):

    npart = pos.shape[0]

    # Set up plot
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax2.scatter(pos[:, 0] - boxsize / 2, pos[:, 1] - boxsize / 2)

    for p in prog_pos:
        if p == -2:
            ax1.scatter(prog_pos[p][:, 0] - boxsize / 2, prog_pos[p][:, 1] - boxsize / 2, color="k",
                        zorder=0, marker=".")
        else:
            ax1.scatter(prog_pos[p][:, 0] - boxsize / 2, prog_pos[p][:, 1] - boxsize / 2, marker=".")
    for d in desc_pos:
        if d == -2:
            ax3.scatter(desc_pos[d][:, 0] - boxsize / 2, desc_pos[d][:, 1] - boxsize / 2, color="k",
                        zorder=0, marker=".")
        else:
            ax3.scatter(desc_pos[d][:, 0] - boxsize / 2, desc_pos[d][:, 1] - boxsize / 2, marker=".")

    for ax in [ax1, ax2, ax3]:
        ax.set_aspect("equal")
        ax.set_xlim(-boxsize / 2, boxsize / 2)
        ax.set_ylim(-boxsize / 2, boxsize / 2)

    fig.savefig("analytics/plots/halo_pos_%d_%d_%s.png" % (ihalo, npart, snap),
                bbox_inches="tight")


def wrap(pos, boxsize):

    # Define the comparison particle as the maximum
    # position in the current dimension
    max_part_pos = pos.max(axis=0)

    # Compute all the halo particle separations from the maximum position
    sep = max_part_pos - pos

    # If any separations are greater than 50% the boxsize
    # (i.e. the halo is split over the boundary)
    # bring the particles at the lower boundary together
    # with the particles at the upper boundary
    # (ignores halos where constituent particles aren't
    # separated by at least 50% of the boxsize)
    # *** Note: fails if halo's extent is greater than 50%
    # of the boxsize in any dimension ***
    pos[np.where(sep > 0.5 * boxsize)] += boxsize

    return pos


def main_plot():

    # Read the parameter file
    paramfile = sys.argv[1]
    (inputs, flags, params, cosmology,
     simulation) = read_param(paramfile)

    # Load the snapshot list
    snaplist = list(np.loadtxt(inputs['snapList'], dtype=str))

    # Get the snapshot index
    snap_ind = int(sys.argv[2])

    # How many plots?
    nhalo = int(sys.argv[3])

    # Get the snapshot
    snap = snaplist[snap_ind]
    prog_snap = snaplist[snap_ind - 1]
    desc_snap = snaplist[snap_ind + 1]

    # Open sim hdf5 files
    hdf = h5py.File(inputs['data'] + inputs["basename"] + snap + ".hdf5", "r")
    hdf_prog = h5py.File(inputs['data'] + inputs["basename"] + prog_snap + ".hdf5",
                         "r")
    hdf_desc = h5py.File(inputs['data'] + inputs["basename"] + desc_snap + ".hdf5",
                         "r")

    # Get boxsize
    l = hdf["Header"].attrs["BoxSize"][0]

    z = hdf["Header"].attrs["Redshift"]
    z_prog = hdf_prog["Header"].attrs["Redshift"]
    z_desc = hdf_desc["Header"].attrs["Redshift"]

    # t = cosmo.age(z)
    # t_p = cosmo.age(z_prog)
    # t_d = cosmo.age(z_desc)

    # Get positions
    pos = hdf["PartType1"]["Coordinates"][...]
    pos_prog = hdf_prog["PartType1"]["Coordinates"][...]
    pos_desc = hdf_desc["PartType1"]["Coordinates"][...]
    
    # Get particle ids
    pids = hdf["PartType1"]["ParticleIDs"][...]
    pids_prog = hdf_prog["PartType1"]["ParticleIDs"][...]
    pids_desc = hdf_desc["PartType1"]["ParticleIDs"][...]

    hdf.close()
    hdf_prog.close()
    hdf_desc.close()

    # Open the halo files
    hdf = h5py.File(inputs['haloSavePath'] + 'halos_' + snap + '.hdf5',
                    'r')
    hdf_prog = h5py.File(inputs['haloSavePath'] + 'halos_' + prog_snap + '.hdf5',
                    'r')
    hdf_desc = h5py.File(inputs['haloSavePath'] + 'halos_' + desc_snap + '.hdf5',
                    'r')

    # Get sorting indices
    sinds = hdf["sort_inds"][...]
    sinds_prog = hdf_prog["sort_inds"][...]
    sinds_desc = hdf_desc["sort_inds"][...]

    # Get the halo nparts for halo selection
    nparts = hdf["nparts"][...]
    nparts_prog = hdf_prog["nparts"][...]
    nparts_desc = hdf_desc["nparts"][...]

    assert np.all(pids[sinds] == pids_prog[sinds_prog]), "pids are not equal!!"
    
    # # Sort positions
    # pos = pos[sinds, :]
    # pos_prog = pos_prog[sinds_prog, :]
    # pos_desc = pos_desc[sinds_desc, :]
    
    # Get sorted particle indices
    part_ids = hdf["part_ids"][...]
    part_ids_prog = hdf_prog["part_ids"][...]
    part_ids_desc = hdf_desc["part_ids"][...]
    part_sids = hdf["sorted_part_ids"][...]
    
    # Get start index pointers
    begin = hdf["start_index"][...]
    begin_prog = hdf_prog["start_index"][...]
    begin_desc = hdf_desc["start_index"][...]
    
    # Get halo sim particle ids
    halo_pids = hdf["sim_part_ids"][...]
    halo_pids_prog = hdf_prog["sim_part_ids"][...]
    halo_pids_desc = hdf_desc["sim_part_ids"][...]
    
    # Open particle halo ids
    part_haloids = hdf["particle_halo_IDs"][...]
    part_progids = hdf_prog["particle_halo_IDs"][...]
    part_descids = hdf_desc["particle_halo_IDs"][...]
    
    # Loop until we've made all plots
    count = 0
    while count < nhalo:

        print(count)
        
        # Get a halo to plot
        ihalo = np.argmax(nparts[:, 1])
        
        # Get start and end
        b, e = begin[ihalo], begin[ihalo] + nparts[ihalo, 1]
        
        # Get parts
        parts = part_ids[b: e]
        sinds = part_sids[b: e]
        sim_pids = halo_pids[b: e]
        
        # Get positions
        halo_pos = wrap(pos[parts, :], l)
        
        # Get progenitors and descendants
        progs, pcounts = np.unique(part_progids[np.in1d(pids_prog, sim_pids)],
                                   return_counts=True)
        descs, dcounts = np.unique(part_descids[np.in1d(pids_desc, sim_pids)],
                                   return_counts=True)

        # Remove improper links
        progs = progs[pcounts >= 10]
        descs = descs[dcounts >= 10]
        
        # Get prog data
        prog_poss = {}
        for prog in progs:

            if prog == -2:
                
                continue

                # # Get positions
                # prog_poss[prog] = wrap(pos_prog[parts, :], l)

            else:

                # Get start and end
                b_prog, e_prog = begin_prog[prog], \
                                 begin_prog[prog] + nparts_prog[prog, 1]

                # Get parts
                parts_prog = halo_pids_prog[b_prog: e_prog]

                # Get positions
                prog_poss[prog] = wrap(pos_prog[np.in1d(pids_prog, parts_prog), :], l)

        # Get desc data
        desc_poss = {}
        for desc in descs:

            if desc == -2:
                
                continue

                # # Get positions
                # desc_poss[desc] = wrap(pos_desc[parts, :], l)

            else:

                # Get start and end
                b_desc, e_desc = begin_desc[desc], \
                                 begin_desc[desc] + nparts_desc[desc, 1]

                # Get parts
                parts_desc = halo_pids_desc[b_desc: e_desc]

                # Get positions
                desc_poss[desc] = wrap(pos_desc[np.in1d(pids_desc, parts_desc), :], l)

        plot_halo(halo_pos, prog_poss, desc_poss, ihalo, snap, l)
        
        # We're done
        count += 1
        nparts[ihalo] = 0


if __name__ == "__main__":
    main_plot()


