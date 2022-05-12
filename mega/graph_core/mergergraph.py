import pickle

import mega.core.serial_io as io
import mega.graph_core.prog_desc_find as pdfind
from mega.core.domain_decomp import graph_halo_decomp
from mega.core.talking_utils import message, pad_print_middle
from mega.core.timing import TicToc

import mpi4py
from mpi4py import MPI

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


# TODO: All mentions of mass need to be converted to Npart and mass arrays
#  need to be introduced


def graph_main(density_rank, meta):
    """

    :param meta:
    :return:
    """

    # Instantiate timer
    tictoc = TicToc(meta)
    tictoc.start()

    # ======================= Set up everything we need =======================

    # Lets read the initial data we need to hand off the work
    (prog_npart, proghalo_nparts, prog_rank_partbins, rank_part_progids,
     prog_reals, rank_prog_pids,
     prog_rank_pidbins) = io.read_prog_data(tictoc, meta, density_rank, comm)
    if meta.verbose:
        tictoc.report("Reading progenitor data")
    (nhalo, rank_partbins, reals, rank_part_haloids, halo_nparts,
     rank_pidbins) = io.read_current_data(tictoc, meta, density_rank, comm)
    if meta.verbose:
        tictoc.report("Reading halo data")
    (desc_npart, deschalo_nparts, desc_rank_partbins,
     rank_part_descids, rank_desc_pids,
     desc_rank_pidbins) = io.read_desc_data(tictoc, meta, density_rank, comm)
    if meta.verbose:
        tictoc.report("Reading descendant data")

    if meta.debug and meta.rank == 0:
        for r in range(meta.nranks):
            message(meta.rank, "Rank %d has %d particles" % (r,
                                                             rank_part_haloids.size))

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

    # Get the halos we have to work on
    my_halos, halo_location = graph_halo_decomp(tictoc, nhalo, meta, comm,
                                                density_rank, rank_pidbins)

    if meta.verbose:
        tictoc.report("Splitting halos across ranks")

    # Sort through halos on this rank and work out where
    # we need to communicate for links
    (results, other_rank_prog_parts,
     other_rank_desc_parts) = pdfind.sort_prog_desc(tictoc, meta, my_halos,
                                                    rank_prog_pids,
                                                    rank_desc_pids)

    comm.Barrier()

    if meta.verbose:
        tictoc.report("Sorting local halos")

    # Now we need to send all of our requests to other ranks to get the
    # particles they have that I need
    forn_prog_parts, forn_desc_parts = pdfind.linking_loop(tictoc,
                                                           meta, comm,
                                                           other_rank_prog_parts,
                                                           other_rank_desc_parts,
                                                           rank_part_progids,
                                                           proghalo_nparts,
                                                           rank_part_descids,
                                                           prog_reals,
                                                           deschalo_nparts,
                                                           rank_prog_pids,
                                                           rank_desc_pids)

    # Combine the results from other ranks with our own
    results = pdfind.update_halos(tictoc, meta, results, forn_prog_parts,
                                  forn_desc_parts)

    if meta.verbose:
        tictoc.report("Updating foreign links")

    comm.Barrier()

    # Clean up results removing halos that don't meet the linking criteria
    results = pdfind.clean_halos(tictoc, meta, results)

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
                             density_rank, reals)

        if meta.verbose:
            tictoc.report("Writing")

    tictoc.end()

    if meta.profile:
        tictoc.end_report(comm)
