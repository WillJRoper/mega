from core.domain_decomp import cell_domain_decomp
from core.spatial import spatial_node_task
from core.phase_space import *
import core.utilities as utils
import core.serial_io as serial_io
from core.partition import *
from core.timing import TicToc
from core.halo_tasking import get_halos
from core.halo_stitching import combine_across_ranks
from core.talking_utils import message, pad_print_middle
from core.collect_result import collect_halos

import pickle
import matplotlib.pyplot as plt
import mpi4py
from mpi4py import MPI

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


def hosthalofinder(meta):
    """ Run the halo finder, sort the output results, find subhalos and
        save to a HDF5 file.

    :param meta: Object containing all the simulation and
                 parameter file metadata
    :return: None
    """

    # Instantiate timer
    tictoc = TicToc(meta)
    tictoc.start()

    # Define MPI message tags
    tags = utils.enum('READY', 'DONE', 'EXIT', 'START')

    # Ensure the number of cells is <= number of ranks and adjust
    # such that the number of cells is a multiple of the number of ranks
    if meta.cdim ** 3 % size != 0:
        cells_per_rank = int(np.floor(meta.cdim ** 3 / size))
        meta.cdim = int((cells_per_rank * meta.nranks) ** (1 / 3))
        meta.ncells = meta.cdim ** 3

    if meta.verbose:
        message(meta.rank, "nCells adjusted to %d "
                           "(%d total cells)" % (meta.cdim, meta.cdim**3))

    # ============= Compute parameters for candidate halo testing =============

    tictoc.get_tic()

    # Compute the linking length for host halos
    linkl = meta.llcoeff * meta.mean_sep

    # Get the softening length
    # NOTE: softening is comoving and converted to physical where necessary
    soft = meta.soft

    # Compute the linking length for subhalos
    sub_linkl = meta.sub_llcoeff * meta.mean_sep

    # Define the velocity space linking length
    vlinkl_indp = (np.sqrt(meta.G / 2)
                   * (4 * np.pi * 200
                      * meta.mean_den / 3) ** (1 / 6)).value

    if meta.verbose:
        message(meta.rank, "=" * meta.table_width)
        message(meta.rank,
                pad_print_middle("Redshift/Scale Factor:",
                                 str(meta.z) + "/" + str(meta.a),
                                 length=meta.table_width))
        message(meta.rank, pad_print_middle("Npart:", list(meta.npart),
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Boxsize:", "%.2f cMpc"
                                            % meta.boxsize,
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Comoving Softening Length:",
                                            "%.4f cMpc" % meta.soft,
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Physical Softening Length:",
                                            "%.4f pMpc" % (meta.soft * meta.a),
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Spatial Host Linking Length:",
                                            "%.4f cMpc" % linkl,
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Spatial Subhalo Linking Length:",
                                            "%.4f cMpc" % sub_linkl,
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Initial Phase Space Host Linking "
                                            "Length (for 10**10 M_sun halo):",
                                         str(meta.ini_vlcoeff * vlinkl_indp *
                                             10 ** 10 ** (1 / 3)) + " km / s",
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Initial Phase Space Subhalo "
                                            "Linking Length (for 10**10 M_sun "
                                            "subhalo):",
                                         str(meta.ini_vlcoeff * vlinkl_indp *
                                             10 ** 10 ** (1 / 3) * 8
                                             ** (1 / 6)) + " km / s",
                                            length=meta.table_width))
        message(meta.rank, "=" * meta.table_width)

    tictoc.get_toc()
    tictoc.record_time("Housekeeping")
    
    # =============== Domain Decomposition ===============

    # Get the particles and tree particles on this rank
    rank_parts, rank_tree_parts, cell_ranks = cell_domain_decomp(tictoc,
                                                                 meta,
                                                                 comm)

    if meta.verbose:
        tictoc.report("Cell Domain Decomposition")

    # Get the positions for the tree
    # NOTE: for now it's more efficient to read all particles
    # and extract the particles we need, throwing away the ones
    # we don't, could be problematic with large datasets
    tree_pos = serial_io.read_subset(tictoc, meta, "PartType1/part_pos",
                                     rank_tree_parts)

    comm.Barrier()

    if meta.verbose:
        tictoc.report("Reading tree positions")

    tictoc.get_tic()

    # Build the kd tree with the boxsize argument providing 'wrapping'
    # due to periodic boundaries *** Note: Contrary to cKDTree
    # documentation compact_nodes=False and balanced_tree=False results in
    # faster queries (documentation recommends compact_nodes=True
    # and balanced_tree=True)***
    tree = cKDTree(tree_pos,
                   leafsize=16,
                   compact_nodes=False,
                   balanced_tree=False,
                   boxsize=[meta.boxsize, meta.boxsize, meta.boxsize])

    tictoc.get_toc()

    if meta.verbose:
        tictoc.report("Tree building")
        message(meta.rank, "Tree memory footprint: %d bytes"
                % utils.get_size(tree))

    tictoc.record_time("Tree-Building")

    # Get the positions for searching on this rank
    # NOTE: for now it's more efficient to read all particles
    # and extract the particles we need and throw away the ones
    # we don't, could be problematic with large datasets
    pos = serial_io.read_subset(tictoc, meta, "PartType1/part_pos", rank_parts)

    if meta.verbose:
        tictoc.report("Reading positions")

    # =========================== Find spatial halos ==========================

    # Extract the spatial halos for this tasks particles
    result, weights, qtime_dict = spatial_node_task(tictoc, meta,
                                                    rank_parts,
                                                    rank_tree_parts,
                                                    pos, tree, linkl)

    comm.Barrier()

    if meta.verbose:
        tictoc.report("Spatial search")

    # TODO: Find hydro particle halo ids here

    # ================= Combine spatial results across ranks ==================

    halo_tasks, weights = combine_across_ranks(tictoc, meta, cell_ranks,
                                               tree_pos, result,
                                               rank_tree_parts, comm,
                                               weights, qtime_dict)

    comm.Barrier()

    if meta.verbose:
        tictoc.report("Combining halos across ranks")

    # ============ Test Halos in Phase Space and find substructure ============

    # TODO: can use an offsets array to do a full decomp of halos across ranks
    tictoc.get_tic()

    # Get input data hdf5 key (DM particle for DMO, concatenated
    # array of all species otherwise)
    if meta.dmo:
        hdf_part_key = "PartType1"
    else:
        hdf_part_key = "All"

    # Define halo dictionaries and ID counters
    results = {}
    sub_results = {}
    haloID = 0
    subhaloID = 0

    tictoc.get_toc()
    tictoc.record_time("Housekeeping")

    if rank == 0:

        count = 0

        # Master process executes code below
        num_workers = size - 1
        closed_workers = 0
        while closed_workers < num_workers:

            # If all other tasks are currently working let the master
            # handle a (fast) low mass halo
            if comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):

                count += 1

                data = comm.recv(source=MPI.ANY_SOURCE,
                                 tag=MPI.ANY_TAG,
                                 status=status)
                source = status.Get_source()
                tag = status.Get_tag()

                if tag == tags.READY:

                    # Worker is ready, so send it a task
                    if len(halo_tasks) != 0:

                        tictoc.get_tic()

                        key, this_task = halo_tasks.popitem()

                        comm.send(this_task, dest=source, tag=tags.START)

                        tictoc.get_toc()

                        tictoc.record_time("Assigning")

                    else:

                        # There are no tasks left so terminate this process
                        comm.send(None, dest=source, tag=tags.EXIT)

                elif tag == tags.EXIT:

                    closed_workers += 1

            elif len(halo_tasks) > 0 and count > size * 1.5:

                count = 0

                key, this_task = halo_tasks.popitem()

                if len(this_task) > 100:

                    halo_tasks[key] = this_task

                else:

                    # Test the halo we've been given in phase space and find
                    # subhalos if we are looking for them
                    res_tup = get_halos(tictoc, this_task, meta, results,
                                        sub_results, vlinkl_indp, haloID,
                                        subhaloID, hdf_part_key)
                    results, sub_results, haloID, subhaloID = res_tup

            elif len(halo_tasks) == 0:

                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                 status=status)
                source = status.Get_source()
                tag = status.Get_tag()

                if tag == tags.EXIT:

                    closed_workers += 1

                else:

                    # There are no tasks left so terminate this process
                    comm.send(None, dest=source, tag=tags.EXIT)

    else:

        # ================ Get from master and complete tasks =================

        while True:

            comm.send(None, dest=0, tag=tags.READY)
            this_task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:

                # Test the halo we've been given in phase space and find
                # subhalos if we are looking for them
                res_tup = get_halos(tictoc, this_task, meta, results,
                                    sub_results, vlinkl_indp, haloID,
                                    subhaloID, hdf_part_key)
                results, sub_results, haloID, subhaloID = res_tup

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)

    # Collect child process results
    tictoc.get_tic()
    collected_results = comm.gather(results, root=0)
    sub_collected_results = comm.gather(sub_results, root=0)
    tictoc.get_toc()

    tictoc.record_time("Collecting")

    if rank == 0:

        # Lets collect all the halos we have collected from the other ranks
        res_tup = collect_halos(tictoc, meta, collected_results,
                                sub_collected_results)
        (newPhaseID, newPhaseSubID, results_dict, haloID_dict,
         sub_results_dict, subhaloID_dict, phase_part_haloids) = res_tup

        if meta.verbose:
            tictoc.report("Combining results")
            message(meta.rank, "Results total memory footprint: %.2f MB" % (
                    utils.get_size(results_dict) * 10 ** -6))

        # If profiling enable plot the number of halos on each rank
        if meta.profile:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.bar(np.arange(len(collected_results)),
                   [len(res) for res in collected_results],
                   color="b", edgecolor="b")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Number of halos computed")
            fig.savefig(meta.profile_path + "/plots/halos_computed_"
                        + str(meta.snap) + ".png")
        if meta.profile and meta.findsubs:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.bar(np.arange(len(sub_collected_results)),
                   [len(res) for res in sub_collected_results],
                   color="r", edgecolor="r")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Number of subhalos computed")
            fig.savefig(meta.profile_path + "/plots/subhalos_computed_"
                        + str(meta.snap) + ".png")

        # ========================== Write out data ==========================

        serial_io.write_data(tictoc, meta, newPhaseID, newPhaseSubID,
                             results_dict, haloID_dict, sub_results_dict,
                             subhaloID_dict, phase_part_haloids)

        if meta.verbose:
            tictoc.report("Writing")

    tictoc.end()

    if meta.profile:

        tictoc.end_report(comm)
    #     for r in range(meta.nranks):
    #         if r == meta.rank:
    #             tictoc.end_report()
    #         comm.Barrier()

        with open(meta.profile_path + "Halo_" + str(rank) + '_'
                  + meta.snap + '.pck', 'wb') as pfile:
            pickle.dump(tictoc.task_time, pfile)
