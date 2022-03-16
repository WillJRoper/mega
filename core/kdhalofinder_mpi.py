# from guppy import hpy; hp = hpy()
import pickle

import matplotlib.pyplot as plt
import mpi4py
import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

mpi4py.rc.recv_mprobe = False
import astropy.constants as const
import astropy.units as u
import time
import h5py
import gc
import sys

from domain_decomp import cell_domain_decomp
from spatial import spatial_node_task
from phase_space import *
import utilities as utils
import serial_io
from partition import *
from timing import TicToc
from halo_stitching import combine_across_ranks

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


def hosthalofinder(meta):
    """ Run the halo finder, sort the output results, find subhalos and
        save to a HDF5 file.

    :param snapshot: The snapshot ID.
    :param meta: Object containing all the simulation and
                 parameter file metadata
    :return: None
    """

    # Instantiate timer
    tictoc = TicToc()

    # Define MPI message tags
    tags = utils.enum('READY', 'DONE', 'EXIT', 'START')

    # Ensure the number of cells is <= number of ranks and adjust
    # such that the number of cells is a multiple of the number of ranks
    if meta.cdim ** 3 % size != 0:
        cells_per_rank = int(np.floor(meta.cdim ** 3 / size))
        meta.cdim = int((cells_per_rank * meta.nranks) ** (1 / 3))
        meta.ncells = meta.cdim ** 3

    if meta.verbose and meta.rank == 0:
        print("nCells adjusted to %d (%d total cells)" % (meta.cdim,
                                                          meta.cdim**3))

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

    if meta.verbose and rank == 0:
        print("=" * meta.table_width)
        print(utils.pad_print_middle("Redshift/Scale Factor:", str(meta.z) + "/" + str(meta.a), length=meta.table_width))
        print(utils.pad_print_middle("Npart:", list(meta.npart), length=meta.table_width))
        print(utils.pad_print_middle("Boxsize:", "%.2f cMpc" % meta.boxsize, length=meta.table_width))
        print(utils.pad_print_middle("Comoving Softening Length:", "%.4f cMpc" % meta.soft, length=meta.table_width))
        print(utils.pad_print_middle("Physical Softening Length:", "%.4f pMpc" % (meta.soft * meta.a), length=meta.table_width))
        print(utils.pad_print_middle("Spatial Host Linking Length:", "%.4f cMpc" % linkl, length=meta.table_width))
        print(utils.pad_print_middle("Spatial Subhalo Linking Length:", "%.4f cMpc" % sub_linkl, length=meta.table_width))
        print(utils.pad_print_middle("Initial Phase Space Host Linking Length (for 10**10 M_sun halo):",
                                         str(meta.ini_vlcoeff * vlinkl_indp * 10 ** 10 ** (1 / 3)) + " km / s", length=meta.table_width))
        print(utils.pad_print_middle("Initial Phase Space Subhalo Linking Length (for 10**10 M_sun subhalo):",
                                         str(meta.ini_vlcoeff * vlinkl_indp * 10 ** 10 ** (1 / 3) * 8 ** (1 / 6)) + " km / s", length=meta.table_width))
        print("=" * meta.table_width)

    tictoc.get_toc()

    if meta.profile:
        tictoc.task_time["Housekeeping"]["Start"].append(tictoc.tic)
        tictoc.task_time["Housekeeping"]["End"].append(tictoc.toc)
    
    # =============== Domain Decomposition ===============

    # Get the particles and tree particles on this rank
    rank_parts, rank_tree_parts, cell_ranks = cell_domain_decomp(tictoc,
                                                                 meta,
                                                                 comm)

    if meta.rank == 0:
        tictoc.report("Cell Domain Decomposition")

    tictoc.get_tic()

    # Open hdf5 file
    hdf = h5py.File(meta.inputpath + meta.snap + ".hdf5", 'r')

    # Get positions for this rank
    # NOTE: for now it's more efficient to read all particles
    # and extract the particles we need and throw away the ones
    # we don't, could be problematic with large datasets
    all_pos = hdf["PartType1"]['part_pos'][...]

    hdf.close()

    tree_pos = all_pos[rank_tree_parts, :]

    del all_pos
    gc.collect()

    tictoc.get_toc()

    if meta.verbose:
        tictoc.report("Reading positions")

    if meta.profile:
        tictoc.task_time["Reading"]["Start"].append(tictoc.tic)
        tictoc.task_time["Reading"]["End"].append(tictoc.toc)

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

    if meta.verbose and meta.rank == 0:
        tictoc.report("Tree building")

        print("Tree memory size", utils.get_size(tree), "bytes")

    if meta.profile:
        tictoc.task_time["Domain-Decomp"]["Start"].append(tictoc.tic)
        tictoc.task_time["Domain-Decomp"]["End"].append(tictoc.toc)

    tictoc.get_tic()

    # Open hdf5 file
    hdf = h5py.File(meta.inputpath + meta.snap + ".hdf5", 'r')

    # Get the positions for searching on this rank
    # NOTE: for now it's more efficient to read all particles
    # and extract the particles we need and throw away the ones
    # we don't, could be problematic with large datasets
    all_pos = hdf["PartType1"]['part_pos'][...]

    hdf.close()

    pos = all_pos[rank_parts, :]

    del all_pos
    gc.collect()

    tictoc.get_toc()

    if meta.profile:
        tictoc.task_time["Reading"]["Start"].append(tictoc.tic)
        tictoc.task_time["Reading"]["End"].append(tictoc.toc)

    # =========================== Find spatial halos ==========================

    # Initialise dictionaries for results
    results = {}

    # Initialise task ID counter
    this_task = 0

    # Extract the spatial halos for this tasks particles
    result = spatial_node_task(tictoc, meta, rank_parts,
                               rank_tree_parts, pos,
                               tree, linkl)

    # ================= Combine spatial results across ranks ==================

    halo_tasks = combine_across_ranks(tictoc, meta, cell_ranks, tree_pos, result,
                                      rank_tree_parts, comm)

    tictoc.report("Combining halos across ranks")

    # # Convert to a set for set calculations
    # rank_parts = set(rank_parts)
    #
    # comb_data = utils.combine_tasks_per_thread(results,
    #                                                rank,
    #                                                rank_parts)
    # results, halos_in_other_ranks = comb_data
    #
    # comm.Barrier()
    #
    # if rank == 0:
    #     tictoc.report("Spatial search")
    #
    # # Collect child process results
    # tictoc.get_tic()
    # collected_results = comm.gather(results, root=0)
    # halos_in_other_ranks = comm.gather(halos_in_other_ranks, root=0)
    # tictoc.get_toc()
    #
    # if meta.profile and rank != 0:
    #     tictoc.task_time["Collecting"]["Start"].append(tictoc.tic)
    #     tictoc.task_time["Collecting"]["End"].append(tictoc.toc)
    #
    # if rank == 0:
    #
    #     halos_to_combine = set().union(*halos_in_other_ranks)
    #
    #     # Combine collected results from children processes into a single dict
    #     results = {k: v for d in collected_results for k, v in d.items()}
    #
    #     print(len(results), 'spatial "halos" collected from all ranks')
    #
    #     if meta.verbose:
    #         tictoc.report("Collecting results")
    #
    #     if meta.profile:
    #         tictoc.task_time["Collecting"]["Start"].append(tictoc.tic)
    #         tictoc.task_time["Collecting"]["End"].append(tictoc.toc)
    #
    #     tictoc.get_tic()
    #
    #     halo_tasks = utils.combine_tasks_networkx(results,
    #                                                   size,
    #                                                   halos_to_combine,
    #                                                   meta.npart[1],
    #                                                   meta)
    #
    #     tictoc.get_toc()
    #
    #     if meta.verbose:
    #         tictoc.report("Combining results")
    #
    #     if meta.profile:
    #         tictoc.task_time["Housekeeping"]["Start"].append(tictoc.tic)
    #         tictoc.task_time["Housekeeping"]["End"].append(tictoc.toc)
    #
    # else:
    #
    #     halo_tasks = None
    #
    # if meta.profile:
    #     tictoc.task_time["Communication"]["Start"].append(tictoc.tic)
    #     tictoc.task_time["Communication"]["End"].append(tictoc.toc)

    # ============ Test Halos in Phase Space and find substructure ============

    tictoc.get_tic()

    # Get input data hdf5 key (DM particle for DMO, concatenated
    # array of all species otherwise)
    if meta.dmo:
        hdf_part_key = "PartType1"
    else:
        hdf_part_key = "All"

    # Extract this ranks spatial halo dictionaries
    haloID_dict = {}
    subhaloID_dict = {}
    results = {}
    sub_results = {}
    haloID = 0
    subhaloID = 0

    tictoc.get_toc()

    if meta.profile:
        tictoc.task_time["Housekeeping"]["Start"].append(tictoc.tic)
        tictoc.task_time["Housekeeping"]["End"].append(tictoc.toc)

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

                        if meta.profile:
                            tictoc.task_time["Assigning"]["Start"].append(tictoc.tic)
                            tictoc.task_time["Assigning"]["End"].append(tictoc.toc)

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

                    tictoc.get_tic()

                    this_task = utils.set_2_sorted_array(this_task)

                    # Get halo data from file
                    halo = utils.read_halo_data(this_task, meta.inputpath,
                                                    meta.snap, hdf_part_key,
                                                    meta.ini_vlcoeff, meta.boxsize,
                                                    soft, meta.z, meta.G, meta.cosmo)

                    tictoc.get_toc()

                    if meta.profile:
                        tictoc.task_time["Reading"]["Start"].append(tictoc.tic)
                        tictoc.task_time["Reading"]["End"].append(tictoc.toc)

                    tictoc.get_tic()

                    # Do the work here
                    result = get_real_host_halos(halo, meta.boxsize, vlinkl_indp,
                                                 linkl, meta.decrement, meta.z, meta.G,
                                                 meta.h,
                                                 soft, meta.min_vlcoeff, meta.cosmo)

                    # Save results
                    for res in result:
                        results[(rank, haloID)] = res

                        haloID += 1

                    tictoc.get_toc()

                    if meta.profile:
                        tictoc.task_time["Host-Phase"]["Start"].append(tictoc.tic)
                        tictoc.task_time["Host-Phase"]["End"].append(tictoc.toc)

                    if meta.findsubs:

                        spatial_sub_results = {}

                        # Loop over results getting spatial halos
                        for res in result:

                            tictoc.get_tic()

                            thishalo_pids = np.sort(res["pids"])

                            # Open hdf5 file
                            hdf = h5py.File(meta.inputpath + meta.snap + ".hdf5",
                                            'r')

                            # Get the position and velocity of each
                            # particle in this rank
                            subhalo_poss = hdf[hdf_part_key]['part_pos'][
                                           thishalo_pids, :]

                            hdf.close()

                            tictoc.get_toc()

                            if meta.profile:
                                tictoc.task_time["Reading"]["Start"].append(tictoc.tic)
                                tictoc.task_time["Reading"]["End"].append(tictoc.toc)

                            tictoc.get_tic()

                            # Do the work here
                            sub_result = get_sub_halos(thishalo_pids,
                                                       subhalo_poss,
                                                       sub_linkl)

                            while len(sub_result) > 0:
                                key, res = sub_result.popitem()
                                spatial_sub_results[subhaloID] = res

                                subhaloID += 1

                            tictoc.get_toc()

                            if meta.profile:
                                tictoc.task_time["Sub-Spatial"]["Start"].append(tictoc.tic)
                                tictoc.task_time["Sub-Spatial"]["End"].append(tictoc.toc)

                        # Loop over spatial subhalos
                        while len(spatial_sub_results) > 0:

                            tictoc.get_tic()

                            key, this_sub_task = spatial_sub_results.popitem()

                            this_sub_task = utils.set_2_sorted_array(this_sub_task)

                            # Get halo data from file
                            subhalo = utils.read_halo_data(this_sub_task,
                                                               meta.inputpath,
                                                               meta.snap,
                                                               hdf_part_key,
                                                               meta.ini_vlcoeff,
                                                               meta.boxsize,
                                                               soft, meta.z,
                                                               meta.G,
                                                               meta.cosmo)

                            tictoc.get_toc()

                            if meta.profile:
                                tictoc.task_time["Reading"]["Start"].append(tictoc.tic)
                                tictoc.task_time["Reading"]["End"].append(tictoc.toc)

                            tictoc.get_tic()

                            # Do the work here
                            result = get_real_host_halos(subhalo, meta.boxsize,
                                                         vlinkl_indp * 8 ** (
                                                                 1 / 6),
                                                         linkl, meta.decrement,
                                                         meta.z, meta.G, meta.h,
                                                         soft, meta.min_vlcoeff,
                                                         meta.cosmo)

                            # Save results
                            for res in result:
                                sub_results[(rank, subhaloID)] = res

                                subhaloID += 1

                            tictoc.get_toc()

                            if meta.profile:
                                tictoc.task_time["Sub-Phase"]["Start"].append(tictoc.tic)
                                tictoc.task_time["Sub-Phase"]["End"].append(tictoc.toc)

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

                tictoc.get_tic()

                this_task = utils.set_2_sorted_array(this_task)

                # Get halo data from file
                halo = utils.read_halo_data(this_task, meta.inputpath,
                                                meta.snap, hdf_part_key,
                                                meta.ini_vlcoeff, meta.boxsize,
                                                soft, meta.z, meta.G, meta.cosmo)

                tictoc.get_toc()

                if meta.profile:
                    tictoc.task_time["Reading"]["Start"].append(tictoc.tic)
                    tictoc.task_time["Reading"]["End"].append(tictoc.toc)

                tictoc.get_tic()

                # Do the work here
                result = get_real_host_halos(halo, meta.boxsize, vlinkl_indp,
                                             linkl, meta.decrement, meta.z, meta.G, meta.h,
                                             soft, meta.min_vlcoeff, meta.cosmo)

                # Save results
                for res in result:
                    results[(rank, haloID)] = res

                    haloID += 1

                tictoc.get_toc()

                if meta.profile:
                    tictoc.task_time["Host-Phase"]["Start"].append(tictoc.tic)
                    tictoc.task_time["Host-Phase"]["End"].append(tictoc.toc)

                if meta.findsubs:

                    spatial_sub_results = {}

                    # Loop over results getting spatial halos
                    for res in result:

                        tictoc.get_tic()

                        thishalo_pids = np.sort(res["pids"])

                        # Open hdf5 file
                        hdf = h5py.File(meta.inputpath + meta.snap + ".hdf5", 'r')

                        # Get the position and velocity of each
                        # particle in this rank
                        subhalo_poss = hdf[hdf_part_key]['part_pos'][
                                       thishalo_pids, :]

                        hdf.close()

                        tictoc.get_toc()

                        if meta.profile:
                            tictoc.task_time["Reading"]["Start"].append(tictoc.tic)
                            tictoc.task_time["Reading"]["End"].append(tictoc.toc)

                        tictoc.get_tic()

                        # Do the work here
                        sub_result = get_sub_halos(thishalo_pids,
                                                   subhalo_poss,
                                                   sub_linkl)

                        while len(sub_result) > 0:
                            key, res = sub_result.popitem()
                            spatial_sub_results[subhaloID] = res

                            subhaloID += 1

                        tictoc.get_toc()

                        if meta.profile:
                            tictoc.task_time["Sub-Spatial"]["Start"].append(tictoc.tic)
                            tictoc.task_time["Sub-Spatial"]["End"].append(tictoc.toc)

                    # Loop over spatial subhalos
                    while len(spatial_sub_results) > 0:

                        tictoc.get_tic()

                        key, this_sub_task = spatial_sub_results.popitem()

                        this_sub_task = utils.set_2_sorted_array(this_sub_task)

                        # Get halo data from file
                        subhalo = utils.read_halo_data(this_sub_task,
                                                           meta.inputpath,
                                                           meta.snap,
                                                           hdf_part_key,
                                                           meta.ini_vlcoeff,
                                                           meta.boxsize,
                                                           soft, meta.z,
                                                           meta.G,
                                                           meta.cosmo)

                        tictoc.get_toc()

                        if meta.profile:
                            tictoc.task_time["Reading"]["Start"].append(tictoc.tic)
                            tictoc.task_time["Reading"]["End"].append(tictoc.toc)

                        tictoc.get_tic()

                        # Do the work here
                        result = get_real_host_halos(subhalo, meta.boxsize,
                                                     vlinkl_indp * 8 ** (
                                                             1 / 6),
                                                     linkl, meta.decrement,
                                                     meta.z, meta.G, meta.h,
                                                     soft, meta.min_vlcoeff,
                                                     meta.cosmo)

                        # Save results
                        for res in result:
                            sub_results[(rank, subhaloID)] = res

                            subhaloID += 1

                        tictoc.get_toc()

                        if meta.profile:
                            tictoc.task_time["Sub-Phase"]["Start"].append(tictoc.tic)
                            tictoc.task_time["Sub-Phase"]["End"].append(tictoc.toc)

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)

    # Collect child process results
    tictoc.get_tic()
    collected_results = comm.gather(results, root=0)
    sub_collected_results = comm.gather(sub_results, root=0)

    tictoc.get_toc()

    if meta.profile and rank != 0:
        tictoc.task_time["Collecting"]["Start"].append(tictoc.tic)
        tictoc.task_time["Collecting"]["End"].append(tictoc.toc)

    if rank == 0:

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

        newPhaseID = 0
        newPhaseSubID = 0

        phase_part_haloids = np.full((np.sum(meta.npart), 2), -2, dtype=np.int32)

        memory_use = 0

        # Collect host halo results
        results_dict = {}
        for halo_task in collected_results:
            for halo in halo_task:
                results_dict[(halo, newPhaseID)] = halo_task[halo]
                pids = halo_task[halo]['pids']
                haloID_dict[(halo, newPhaseID)] = newPhaseID
                phase_part_haloids[pids, 0] = newPhaseID
                newPhaseID += 1
                memory_use += halo_task[halo]['memory']

        print("Halo objects total footprint:", memory_use)

        # Collect subhalo results
        sub_results_dict = {}
        for subhalo_task in sub_collected_results:
            for subhalo in subhalo_task:
                sub_results_dict[(subhalo, newPhaseSubID)] = subhalo_task[
                    subhalo]
                pids = subhalo_task[subhalo]['pids']
                subhaloID_dict[(subhalo, newPhaseSubID)] = newPhaseSubID
                phase_part_haloids[pids, 1] = newPhaseSubID
                newPhaseSubID += 1

        if meta.verbose:
            tictoc.report("Combining results")
            print("Results memory size", utils.get_size(results_dict), "bytes")

        if meta.profile:
            tictoc.task_time["Collecting"]["Start"].append(tictoc.tic)
            tictoc.task_time["Collecting"]["End"].append(tictoc.toc)

        utils.count_and_report_halos(phase_part_haloids[:, 0], meta,
                                         halo_type="Phase Space Host Halos")

        if meta.findsubs:
            utils.count_and_report_halos(phase_part_haloids[:, 1], meta,
                                             halo_type="Phase Space Subhalos")

        # ========================== Write out data ==========================

        serial_io.write_data(tictoc, meta, newPhaseID, newPhaseSubID,
                             results_dict, haloID_dict, sub_results_dict,
                             subhaloID_dict, phase_part_haloids)

    if meta.profile:
        tictoc.task_time["END"] = tictoc.get_tic()

        with open(meta.profile_path + "Halo_" + str(rank) + '_'
                  + meta.snap + '.pck', 'wb') as pfile:
            pickle.dump(tictoc.task_time, pfile)
