import pickle

import mega.core.serial_io as io
import mega.graph_core.prog_desc_find as pdfind
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

    # Lets read the initial data we need to hand off the work
    prog_objs, prog_min_pids, prog_max_pids = io.read_link_data(
        tictoc, meta, density_rank, meta.prog_snap
    )
    if meta.verbose:
        tictoc.report("Reading progenitor data")
    halos, min_pids, max_pids, nhalo = io.read_current_data(
        tictoc, meta, density_rank
    )
    if meta.verbose:
        tictoc.report("Reading and splitting halo data")
    desc_objs, desc_min_pids, desc_max_pids = io.read_link_data(
        tictoc, meta, density_rank, meta.desc_snap
    )
    if meta.verbose:
        tictoc.report("Reading descendant data")

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
        message(meta.rank, "=" * meta.table_width)

    tictoc.stop_func_time()

    tictoc.start_func_time("Linking")

    # Now loop over the halos we have on this rank and compare them to
    # all progenitors and descendants
    for halo in halos:

        # Loop over the progenitors testing each one
        for prog in prog_objs:

            # Lets do the early skip if we can
            if prog.min_pid > halo.max_pid or prog.max_pid < halo.min_pid:
                continue

            # Compare this halo and progenitor
            halo.compare_prog(prog, meta)

        # Loop over the descendants testing each one
        for desc in desc_objs:

            # Lets do the early skip if we can
            if desc.min_pid > halo.max_pid or desc.max_pid < halo.min_pid:
                continue

            # Compare this halo and progenitor
            halo.compare_desc(desc, meta)

    comm.Barrier()

    tictoc.stop_func_time()

    if meta.verbose:
        tictoc.report("Linking progenitors and descendants")

    # Clean up results removing halos that don't meet the linking criteria
    results = pdfind.clean_halos(tictoc, meta, halos)

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
