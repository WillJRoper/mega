import pickle
import numpy as np
import astropy.units as u

import mega.core.serial_io as io
import mega.graph_core.prog_desc_find as pdfind
import mega.core.domain_decomp as dd
from mega.core.talking_utils import message, pad_print_middle

import mpi4py
from mpi4py import MPI

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


def graph_main(density_rank, meta):
    """

    :param meta:
    :return:
    """

    # Get the timer instance
    tictoc = meta.tictoc

    # ======================= Set up everything we need =======================

    # Count halos, progenitors and descendants
    nhalo, nprog, ndesc = io.count_halos(tictoc, meta, density_rank)

    # Decomp the halos, progenitors and descendants
    my_halos, my_progs, my_descs = dd.halo_cell_domain_decomp(tictoc, meta,
                                                              comm, nhalo,
                                                              nprog, ndesc,
                                                              density_rank)
    if meta.verbose:
        tictoc.report("Splitting halos, progenitors and descendants")

    # Lets read the initial data we need to hand off the work
    prog_objs = io.read_link_data(
        tictoc, meta, density_rank, meta.prog_snap, my_progs
    )
    if meta.verbose:
        tictoc.report("Reading progenitor data")
    halo_objs, nhalo = io.read_current_data(
        tictoc, meta, density_rank, my_halos
    )
    if meta.verbose:
        tictoc.report("Reading halo data")
    desc_objs = io.read_link_data(
        tictoc, meta, density_rank, meta.desc_snap, my_descs
    )
    if meta.verbose:
        tictoc.report("Reading descendant data")

    # And now construct the cell structure to house our halos
    halo_cells, prog_cells, desc_cells = dd.construct_cells(tictoc,
                                                            halo_objs,
                                                            prog_objs,
                                                            desc_objs,
                                                            meta)

    if meta.verbose:
        tictoc.report("Constructing cell structure")

    comm.Barrier()

    tictoc.start_func_time("Housekeeping")

    if meta.verbose and meta.rank == 0:
        message(meta.rank, "=" * meta.table_width)
        message(meta.rank,
                pad_print_middle("Redshift/Scale Factor:",
                                 str(meta.z) + "/" + str(meta.a),
                                 length=meta.table_width))
        message(rank, pad_print_middle("Snapshots:",
                                       "%s->%s->%s" % (meta.prog_snap,
                                                       meta.snap,
                                                       meta.desc_snap),
                                       length=meta.table_width))
        message(rank, pad_print_middle("nHalo:",
                                       "%d" % nhalo,
                                       length=meta.table_width))
        message(rank, pad_print_middle("nProgenitor:",
                                       "%d" % nprog,
                                       length=meta.table_width))
        message(rank, pad_print_middle("nDescendant:",
                                       "%d" % ndesc,
                                       length=meta.table_width))
        message(meta.rank, "=" * meta.table_width)

    tictoc.stop_func_time()

    tictoc.start_func_time("Linking")

    # Now loop over the cells containing halos at the current snapshot
    for ijk, chalos in halo_cells.items():

        # Extract individual coordinates
        i, j, k = ijk

        # Loop over cell's halos
        for halo in chalos:

            # Extract the velocity of this halo and get it's magnitude
            vel = np.sqrt(halo.mean_vel[0] ** 2
                          + halo.mean_vel[1] ** 2
                          + halo.mean_vel[2] ** 2) * meta.U_v_conv

            # Work out how far we have to walk (minimum 2 neighbours)
            prog_d = int(np.ceil((vel * meta.prog_delta_t).value
                                 / meta.cell_width)) + 1
            desc_d = int(np.ceil((vel * meta.desc_delta_t).value
                                 / meta.cell_width)) + 1
            d = np.max((prog_d, desc_d, 2))

            # Loop over all surrounding cells and this one searching for
            # progenitors and descendants
            for ii in range(-d, d + 1, 1):
                iii = i + ii
                if (not meta.periodic) and (iii < 0 or iii > meta.cdim):
                    continue
                iii %= meta.cdim
                for jj in range(-d, d + 1, 1):
                    jjj = j + jj
                    if (not meta.periodic) and (jjj < 0 or jjj > meta.cdim):
                        continue
                    jjj %= meta.cdim
                    for kk in range(-d, d + 1, 1):
                        kkk = k + kk
                        if (not meta.periodic) and (kkk < 0 or kkk > meta.cdim):
                            continue
                        kkk %= meta.cdim

                        # Define cell key
                        key = (iii, jjj, kkk)

                        # Check for progenitors
                        if not meta.isfirst:

                            # Loop over the progenitors testing each one
                            for prog in prog_cells[key]:

                                # Compare this halo and progenitor
                                halo.compare_prog(prog, meta)

                        # Check for descendants
                        if not meta.isfinal:

                            # Loop over the descendants testing each one
                            for desc in desc_cells[key]:

                                # Compare this halo and progenitor
                                halo.compare_desc(desc, meta)

    comm.Barrier()

    tictoc.stop_func_time()

    if meta.verbose:
        tictoc.report("Linking progenitors and descendants")

    # Clean up results removing halos that don't meet the linking criteria
    results = pdfind.clean_halos(tictoc, meta, halo_cells)

    if meta.verbose:
        tictoc.report("Cleaning up progenitors and descendants")

    comm.Barrier()

    # Collect results
    tictoc.start_func_time("Collecting")
    all_results = comm.gather(results, root=0)
    tictoc.stop_func_time()

    if meta.verbose:
        tictoc.report("Collecting results")

    # Write out output
    if meta.rank == 0:

        # Write the data and get the reals arrays
        io.write_dgraph_data(tictoc, meta, all_results,
                             density_rank)

        if meta.verbose:
            tictoc.report("Writing")
