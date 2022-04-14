import core.domain_decomp as dd
from core.spatial import spatial_node_task, get_sub_halos
from core.phase_space import *
from core.hydro import find_hydro_haloids
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
    meta.tictoc = tictoc

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

    tictoc.start_func_time("Housekeeping")

    if meta.verbose:
        message(meta.rank, "=" * meta.table_width)
        message(meta.rank,
                pad_print_middle("Redshift/Scale Factor:",
                                 str(meta.z) + "/" + str(meta.a),
                                 length=meta.table_width))
        message(meta.rank, pad_print_middle("Npart:", list(meta.npart),
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Boxsize:", "[%.2f %.2f %.2f] cMpc"
                                            % (*meta.boxsize, ),
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Comoving Softening Length:",
                                            "%.4f cMpc" % meta.soft,
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Physical Softening Length:",
                                            "%.4f pMpc" % (meta.soft * meta.a),
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Spatial Host Linking Length:",
                                            "%.4f cMpc" % meta.linkl,
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Spatial Subhalo Linking Length:",
                                            "%.4f cMpc" % meta.sub_linkl,
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Initial Phase Space Host Linking "
                                            "Length (for 10**10 M_sun halo):",
                                         str(meta.ini_vlcoeff * meta.vlinkl_indp *
                                             10 ** 10 ** (1 / 3)) + " km / s",
                                            length=meta.table_width))
        message(meta.rank, pad_print_middle("Initial Phase Space Subhalo "
                                            "Linking Length (for 10**10 M_sun "
                                            "subhalo):",
                                         str(meta.ini_vlcoeff * meta.vlinkl_indp *
                                             10 ** 10 ** (1 / 3) * 8
                                             ** (1 / 6)) + " km / s",
                                            length=meta.table_width))
        message(meta.rank, "=" * meta.table_width)

    tictoc.stop_func_time()
    
    # ========================= Domain Decomposition =========================

    # Get the particles and tree particles on this rank
    rank_parts, rank_tree_parts, cell_ranks = dd.cell_domain_decomp(tictoc,
                                                                    meta,
                                                                    comm, 1)

    if meta.verbose:
        tictoc.report("Cell Domain Decomposition")

    # Get the positions for the tree
    # NOTE: for now it's more efficient to read all particles
    # and extract the particles we need, throwing away the ones
    # we don't, could be problematic with large datasets
    tree_pos = serial_io.read_subset(tictoc, meta, "PartType1/Coordinates",
                                     rank_tree_parts)

    if meta.verbose:
        tictoc.report("Reading tree positions")

    # Build the kd tree with the boxsize argument providing 'wrapping'
    # due to periodic boundaries *** Note: Contrary to cKDTree
    # documentation compact_nodes=False and balanced_tree=False results in
    # faster queries (documentation recommends compact_nodes=True
    # and balanced_tree=True)***
    tictoc.start_func_time("Tree-Building")
    tree = cKDTree(tree_pos,
                   leafsize=16,
                   compact_nodes=False,
                   balanced_tree=False,
                   boxsize=[*meta.boxsize, ])
    tictoc.stop_func_time()

    if meta.verbose:
        tictoc.report("Tree building")
        message(meta.rank, "Tree memory footprint: %d bytes"
                % utils.get_size(tree))

    # Get the positions for searching on this rank
    # NOTE: for now it's more efficient to read all particles
    # and extract the particles we need and throw away the ones
    # we don't, could be problematic with large datasets
    pos = serial_io.read_subset(tictoc, meta, "PartType1/Coordinates",
                                rank_parts)

    if meta.verbose:
        tictoc.report("Reading positions")

    # =========================== Find spatial halos ==========================

    # Extract the spatial halos for this tasks particles
    (halo_pids, weights, qtime_dict,
     rank_part_haloids) = spatial_node_task(tictoc, meta, rank_parts,
                                            rank_tree_parts, pos, tree)

    if meta.verbose:
        tictoc.report("Spatial search")

    # ====================== Hydro Domain Decomposition ======================

    # Nest the dark matter halo data inside dictionaries for each
    # particle type
    all_halo_pids = {1: halo_pids}
    rank_tree_parts = {1: rank_tree_parts}
    tree_pos = {1: tree_pos}

    if not meta.dmo:

        # Loop over the present particle types
        for part_type in meta.part_types:

            if part_type == 1:
                continue

            # Get the particles and tree particles on this rank
            rank_hydro_parts = dd.hydro_cell_domain_decomp(tictoc, meta, comm,
                                                           cell_ranks,
                                                           part_type)

            if meta.verbose:
                tictoc.report("Particle Type %d Domain Decomposition"
                              % part_type)

            # Get positions for this particle type
            # NOTE: for now it's more efficient to read all particles
            # and extract the particles we need, throwing away the ones
            # we don't, could be problematic with large datasets
            hydro_pos = serial_io.read_subset(tictoc, meta,
                                              "PartType%d/Coordinates"
                                              % part_type,
                                              rank_hydro_parts, part_type)

            # ============== Associate hydro particles to a halo ==============

            # Update halo particle ids dictionary with hydro particles
            all_halo_pids[part_type] = find_hydro_haloids(tictoc,
                                                          rank_hydro_parts,
                                                          all_halo_pids[1].keys(),
                                                          rank_part_haloids,
                                                          tree, hydro_pos,
                                                          meta)

            if meta.verbose:
                tictoc.report("Particle Type %d spatial search" % part_type)

            # Combine particle index arrays and positions
            rank_tree_parts[part_type] = rank_hydro_parts
            tree_pos[part_type] = hydro_pos

    # ================= Combine spatial results across ranks ==================

    halo_tasks, weights = combine_across_ranks(tictoc, meta, all_halo_pids,
                                               rank_tree_parts, comm,
                                               weights, qtime_dict)

    if meta.verbose:
        tictoc.report("Combining halos across ranks")
        if meta.rank == 0:
            if len(weights) > 0:
                message(meta.rank, "Weighting range: [%.2E - %.2E]"
                        % (np.min(list(weights.values())),
                           np.max(list(weights.values()))))
            else:
                message(meta.rank, "Weighting range: [%.2E - %.2E]"
                        % (0.0, 0.0))

    # ================== Decompose and scatter spatial halos ==================

    # Decomp halos across ranks
    (my_halo_parts, start_index, stride) = dd.halo_decomp(tictoc, meta, 
                                                          halo_tasks,
                                                          weights, comm)

    if meta.verbose:
        tictoc.report("Scattering spatial halos")

    # ============ Test Halos in Phase Space and find substructure ============

    tictoc.start_func_time("Housekeeping")

    # Define halo dictionaries and ID counters
    results = {}
    sub_results = {}
    haloID = 0
    subhaloID = 0

    tictoc.stop_func_time()

    # ================ Read in the particles data for my halos ================

    # Get the halo data on this rank
    # NOTE: for now it's more efficient to read all particles
    # and extract the particles we need, throwing away the ones
    # we don't, could be problematic with large datasets
    halo_data = serial_io.read_multi_halo_data(tictoc, meta, my_halo_parts)
    all_sim_pids, all_pos, all_vel, all_masses, all_part_types = halo_data

    if meta.verbose:
        tictoc.report("Reading halo data")

    # Loop over the tasks we have
    for (itask, begin), length in zip(enumerate(start_index), stride):

        # Get a task to work on
        this_task_inds = my_halo_parts[begin: begin + length]
        this_task_rankinds = np.arange(begin, begin + length, dtype=int)

        # Create the halo object
        tictoc.start_func_time("Create Halo")
        halo = Halo(tictoc, this_task_inds, this_task_rankinds,
                    all_sim_pids[this_task_rankinds],
                    all_pos[this_task_rankinds, :],
                    all_vel[this_task_rankinds, :],
                    all_part_types[this_task_rankinds],
                    all_masses[this_task_rankinds],
                    meta.ini_vlcoeff, meta)
        tictoc.stop_func_time()

        # Test the halo in phase space
        result = get_real_host_halos(tictoc, halo, meta)

        # Save the found halos
        for res in result:
            results[(meta.rank, haloID)] = res

            haloID += 1

        if meta.findsubs:

            # Loop over results getting spatial halos
            for host in result:

                # Get host halo particle indices to get postion
                thishalo_pids = host.shifted_inds

                # Get the particle positions
                subhalo_poss = all_pos[thishalo_pids, :]

                # Test the subhalo in phase space
                sub_result = get_sub_halos(tictoc, thishalo_pids,
                                           subhalo_poss,
                                           meta)

                # Loop over spatial subhalos
                while len(sub_result) > 0:

                    # Get a subahlo task to work on
                    _, this_stask = sub_result.popitem()

                    # Create the subhalo object
                    tictoc.start_func_time("Create Subhalo")
                    sim_inds = my_halo_parts[this_stask]
                    subhalo = Halo(tictoc, sim_inds, this_stask,
                                   all_sim_pids[this_stask],
                                   all_pos[this_stask, :],
                                   all_vel[this_stask, :],
                                   all_part_types[this_stask],
                                   all_masses[this_stask],
                                   meta.ini_vlcoeff, meta)
                    tictoc.stop_func_time()

                    # Test the subhalo in phase space
                    result = get_real_sub_halos(tictoc, subhalo, meta)

                    # Save the found subhalos
                    for res in result:
                        sub_results[(meta.rank, subhaloID)] = res

                        subhaloID += 1

    # Wait for everyone to finish so we get accurate timings
    comm.Barrier()

    # Collect child process results
    tictoc.start_func_time("Collecting")
    collected_results = comm.gather(results, root=0)
    sub_collected_results = comm.gather(sub_results, root=0)
    tictoc.stop_func_time()

    if meta.rank == 0:

        # Lets collect all the halos we have collected from the other ranks
        res_tup = collect_halos(tictoc, meta, collected_results,
                                sub_collected_results)
        nhalo, nsubhalo, results_dict, sub_results_dict = res_tup

        if meta.verbose:
            tictoc.report("Combining results")
            message(meta.rank, "Results total memory footprint: %.2f MB" % (
                    utils.get_size(results_dict) * 10 ** -6))

        # ========================== Write out data ==========================

        serial_io.write_data(tictoc, meta, nhalo, nsubhalo,
                             results_dict, sub_results_dict)

        if meta.verbose:
            tictoc.report("Writing")

    tictoc.end()

    if meta.profile:

        tictoc.end_report(comm)

        with open(meta.profile_path + "Halo_" + str(rank) + '_'
                  + meta.snap + '.pck', 'wb') as pfile:
            pickle.dump(tictoc.task_time, pfile)
