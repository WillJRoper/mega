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
import sys

from halo import Halo
from spatial import *
from phase_space import *
import utilities

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


def hosthalofinder(snapshot, llcoeff, sub_llcoeff, inputpath, savepath,
                   ini_vlcoeff, min_vlcoeff, decrement, verbose, findsubs,
                   ncells, profile, profile_path, cosmo, h, softs, dmo):
    """ Run the halo finder, sort the output results, find subhalos and
        save to a HDF5 file.

    :param snapshot: The snapshot ID.
    :param llcoeff: The host halo linking length coefficient.
    :param sub_llcoeff: The subhalo linking length coefficient.
    :param gadgetpath: The filepath to the gadget simulation data.
    :param batchsize: The number of particle to be queried at one time.
    :param debug_npart: The number of particles to run the program on when
                        debugging.
    :return: None
    """

    # Define MPI message tags
    tags = utilities.enum('READY', 'DONE', 'EXIT', 'START')

    # Ensure the number of cells is <= number of ranks and adjust
    # such that the number of cells is a multiple of the number of ranks
    if ncells < (size - 1):
        ncells = size - 1
    if ncells % size != 0:
        cells_per_rank = int(np.ceil(ncells / size))
        ncells = cells_per_rank * size
    else:
        cells_per_rank = ncells // size

    if verbose and rank == 0:
        print("nCells adjusted to", ncells)

    if profile:
        prof_d = {}
        prof_d["START"] = time.time()
        prof_d["Reading"] = {"Start": [], "End": []}
        prof_d["Domain-Decomp"] = {"Start": [], "End": []}
        prof_d["Communication"] = {"Start": [], "End": []}
        prof_d["Housekeeping"] = {"Start": [], "End": []}
        prof_d["Task-Munging"] = {"Start": [], "End": []}
        prof_d["Host-Spatial"] = {"Start": [], "End": []}
        prof_d["Host-Phase"] = {"Start": [], "End": []}
        prof_d["Sub-Spatial"] = {"Start": [], "End": []}
        prof_d["Sub-Phase"] = {"Start": [], "End": []}
        prof_d["Assigning"] = {"Start": [], "End": []}
        prof_d["Collecting"] = {"Start": [], "End": []}
        prof_d["Writing"] = {"Start": [], "End": []}
    else:
        prof_d = None

    # =============== Domain Decomposition ===============

    read_start = time.time()

    # Open hdf5 file
    hdf = h5py.File(inputpath + snapshot + ".hdf5", 'r')

    # Get parameters for decomposition
    mean_sep = hdf["PartType1"].attrs['mean_sep']
    boxsize = hdf.attrs['boxsize']
    npart = hdf.attrs['npart']
    if dmo:
        temp_npart = np.zeros(npart.size, dtype=np.int64)
        temp_npart[1] = npart[1]
        npart = temp_npart
    redshift = hdf.attrs['redshift']
    tot_mass = hdf["PartType1"].attrs['tot_mass'] * 10 ** 10

    hdf.close()

    if profile:
        prof_d["Reading"]["Start"].append(read_start)
        prof_d["Reading"]["End"].append(time.time())

    # ============= Compute parameters for candidate halo testing =============

    set_up_start = time.time()

    # Compute scale factor
    a = 1 / (1 + redshift)

    # Extract softening lengths
    comoving_soft = softs[0]
    max_physical_soft = softs[1]

    # Compute the linking length for host halos
    linkl = llcoeff * mean_sep

    # Compute the softening length
    # NOTE: softening is comoving and converted to physical where necessary
    if comoving_soft * a > max_physical_soft:
        soft = max_physical_soft / a
    else:
        soft = comoving_soft

    # Define the gravitational constant
    G = (const.G.to(u.km ** 3 * u.M_sun ** -1 * u.s ** -2)).value

    # Compute the linking length for subhalos
    sub_linkl = sub_llcoeff * mean_sep

    # Compute the mean density
    mean_den = (tot_mass * u.M_sun / boxsize ** 3
                / u.Mpc ** 3 * (1 + redshift) ** 3)
    mean_den = mean_den.to(u.M_sun / u.km ** 3)

    # Define the velocity space linking length
    vlinkl_indp = (np.sqrt(G / 2)
                   * (4 * np.pi * 200
                      * mean_den / 3) ** (1 / 6)).value

    if verbose and rank == 0:
        print("Redshift/Scale Factor:", str(redshift) + "/" + str(a))
        print("Npart:", npart)
        print("Boxsize:", boxsize, "cMpc")
        print("Comoving Softening Length:", soft, "cMpc")
        print("Physical Softening Length:", soft * a, "pMpc")
        print("Spatial Host Linking Length:", linkl, "cMpc")
        print("Spatial Subhalo Linking Length:", sub_linkl, "cMpc")
        print("Initial Phase Space Host Linking Length "
              "(for 10**12 M_sun a particle halo):",
              ini_vlcoeff * vlinkl_indp
              * 10 ** 12 ** (1 / 3), "km / s")
        print("Initial Phase Space Subhalo Linking Length "
              "(for 10**9 M_sun a particle subhalo):",
              ini_vlcoeff * vlinkl_indp
              * 10 ** 9 ** (1 / 3) * 8 ** (1 / 6), "km / s")

    if profile:
        prof_d["Housekeeping"]["Start"].append(set_up_start)
        prof_d["Housekeeping"]["End"].append(time.time())

    if rank == 0:

        start_dd = time.time()

        # Open hdf5 file
        hdf = h5py.File(inputpath + snapshot + ".hdf5", 'r')

        # Get positions to perform the decomposition
        pos = hdf["PartType1"]['part_pos'][...]

        hdf.close()

        if profile:
            prof_d["Reading"]["Start"].append(read_start)
            prof_d["Reading"]["End"].append(time.time())

        # Build the kd tree with the boxsize argument providing 'wrapping'
        # due to periodic boundaries *** Note: Contrary to cKDTree
        # documentation compact_nodes=False and balanced_tree=False results in
        # faster queries (documentation recommends compact_nodes=True
        # and balanced_tree=True)***
        tree = cKDTree(pos,
                       leafsize=16,
                       compact_nodes=False,
                       balanced_tree=False,
                       boxsize=[boxsize, boxsize, boxsize])

        if verbose:
            print("Domain Decomposition and tree building:",
                  time.time() - start_dd)

            print("Tree memory size", sys.getsizeof(tree), "bytes")

    else:

        start_dd = time.time()

        tree = None

    dd_data = utilities.decomp_nodes(npart[1], size, cells_per_rank, rank)
    thisRank_tasks, thisRank_parts, nnodes, rank_edges = dd_data

    if profile:
        prof_d["Domain-Decomp"]["Start"].append(start_dd)
        prof_d["Domain-Decomp"]["End"].append(time.time())

    comm_start = time.time()

    # Broadcast the KD-Tree to all ranks
    tree = comm.bcast(tree, root=0)

    if profile:
        prof_d["Communication"]["Start"].append(comm_start)
        prof_d["Communication"]["End"].append(time.time())

    set_up_start = time.time()

    # Get this ranks particles ID "edges"
    low_lim, up_lim = thisRank_parts.min(), thisRank_parts.max() + 1

    # Define this ranks index offset (the minimum particle ID
    # contained in a rank)
    rank_index_offset = thisRank_parts.min()

    if profile:
        prof_d["Housekeeping"]["Start"].append(set_up_start)
        prof_d["Housekeeping"]["End"].append(time.time())

    read_start = time.time()

    # Open hdf5 file
    hdf = h5py.File(inputpath + snapshot + ".hdf5", 'r')

    # Get the position and velocity of each particle in this rank
    pos = hdf["PartType1"]['part_pos'][low_lim: up_lim, :]

    hdf.close()

    if profile:
        prof_d["Reading"]["Start"].append(read_start)
        prof_d["Reading"]["End"].append(time.time())

    # =========================== Find spatial halos ==========================

    start = time.time()

    # Initialise dictionaries for results
    results = {}

    # Initialise task ID counter
    thisTask = 0

    # Loop over this ranks tasks
    while len(thisRank_tasks) > 0:

        # Extract this task particle IDs
        thisTask_parts = thisRank_tasks.pop()

        task_start = time.time()

        # Extract the spatial halos for this tasks particles
        result = spatial_node_task(thisTask,
                                   pos[thisTask_parts - rank_index_offset],
                                   tree, linkl, npart[1])

        # Store the results in a dictionary for later combination
        results[thisTask] = result

        thisTask += 1

        if profile:
            prof_d["Host-Spatial"]["Start"].append(task_start)
            prof_d["Host-Spatial"]["End"].append(time.time())

    # ================= Combine spatial results across ranks ==================

    combine_start = time.time()

    # Convert to a set for set calculations
    thisRank_parts = set(thisRank_parts)

    comb_data = utilities.combine_tasks_per_thread(results,
                                                   rank,
                                                   thisRank_parts)
    results, halos_in_other_ranks = comb_data

    if profile:
        prof_d["Housekeeping"]["Start"].append(combine_start)
        prof_d["Housekeeping"]["End"].append(time.time())

    if rank == 0:
        print("Spatial search finished", time.time() - start)

    # Collect child process results
    collect_start = time.time()
    collected_results = comm.gather(results, root=0)
    halos_in_other_ranks = comm.gather(halos_in_other_ranks, root=0)

    if profile and rank != 0:
        prof_d["Collecting"]["Start"].append(collect_start)
        prof_d["Collecting"]["End"].append(time.time())

    if rank == 0:

        halos_to_combine = set().union(*halos_in_other_ranks)

        # Combine collected results from children processes into a single dict
        results = {k: v for d in collected_results for k, v in d.items()}

        print(len(results), "spatial halos collected")

        if verbose:
            print("Collecting the results took",
                  time.time() - collect_start, "seconds")

        if profile:
            prof_d["Collecting"]["Start"].append(collect_start)
            prof_d["Collecting"]["End"].append(time.time())

        combine_start = time.time()

        halo_tasks = utilities.combine_tasks_networkx(results,
                                                      size,
                                                      halos_to_combine,
                                                      npart[1])

        if verbose:
            print("Combining the results took",
                  time.time() - combine_start, "seconds")

        if profile:
            prof_d["Housekeeping"]["Start"].append(combine_start)
            prof_d["Housekeeping"]["End"].append(time.time())

    else:

        halo_tasks = None

    if profile:
        prof_d["Communication"]["Start"].append(comm_start)
        prof_d["Communication"]["End"].append(time.time())

    # ============ Test Halos in Phase Space and find substructure ============

    set_up_start = time.time()

    # Get input data hdf5 key (DM particle for DMO, concatenated
    # array of all species otherwise)
    if dmo:
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

    if profile:
        prof_d["Housekeeping"]["Start"].append(set_up_start)
        prof_d["Housekeeping"]["End"].append(time.time())

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

                        assign_start = time.time()

                        key, thisTask = halo_tasks.popitem()

                        comm.send(thisTask, dest=source, tag=tags.START)

                        if profile:
                            prof_d["Assigning"]["Start"].append(assign_start)
                            prof_d["Assigning"]["End"].append(time.time())

                    else:

                        # There are no tasks left so terminate this process
                        comm.send(None, dest=source, tag=tags.EXIT)

                elif tag == tags.EXIT:

                    closed_workers += 1

            elif len(halo_tasks) > 0 and count > size * 1.5:

                count = 0

                key, thisTask = halo_tasks.popitem()

                if len(thisTask) > 100:

                    halo_tasks[key] = thisTask

                else:

                    read_start = time.time()

                    thisTask.sort()

                    # Get halo data from file
                    halo = utilities.read_halo_data(thisTask, inputpath,
                                                    snapshot, hdf_part_key,
                                                    ini_vlcoeff, boxsize,
                                                    soft, redshift, G, cosmo)

                    read_end = time.time()

                    if profile:
                        prof_d["Reading"]["Start"].append(read_start)
                        prof_d["Reading"]["End"].append(read_end)

                    task_start = time.time()

                    # Do the work here
                    result = get_real_host_halos(halo, boxsize, vlinkl_indp,
                                                 linkl, decrement, redshift, G,
                                                 h,
                                                 soft, min_vlcoeff, cosmo)

                    # Save results
                    for res in result:
                        results[(rank, haloID)] = result[res]

                        haloID += 1

                    task_end = time.time()

                    if profile:
                        prof_d["Host-Phase"]["Start"].append(task_start)
                        prof_d["Host-Phase"]["End"].append(task_end)

                    if findsubs:

                        spatial_sub_results = {}

                        # Loop over results getting spatial halos
                        while len(result) > 0:

                            read_start = time.time()

                            key, res = result.popitem()

                            thishalo_pids = np.sort(res["pids"])

                            # Open hdf5 file
                            hdf = h5py.File(inputpath + snapshot + ".hdf5",
                                            'r')

                            # Get the position and velocity of each
                            # particle in this rank
                            subhalo_poss = hdf[hdf_part_key]['part_pos'][
                                           thishalo_pids, :]

                            hdf.close()

                            read_end = time.time()

                            if profile:
                                prof_d["Reading"]["Start"].append(read_start)
                                prof_d["Reading"]["End"].append(read_end)

                            task_start = time.time()

                            # Do the work here
                            sub_result = get_sub_halos(thishalo_pids,
                                                       subhalo_poss,
                                                       sub_linkl)

                            while len(sub_result) > 0:
                                key, res = sub_result.popitem()
                                spatial_sub_results[subhaloID] = res

                                subhaloID += 1

                            task_end = time.time()

                            if profile:
                                prof_d["Sub-Spatial"]["Start"].append(
                                    task_start)
                                prof_d["Sub-Spatial"]["End"].append(task_end)

                        # Loop over spatial subhalos
                        while len(spatial_sub_results) > 0:

                            read_start = time.time()

                            key, thisSub = spatial_sub_results.popitem()

                            thisSub.sort()

                            # Get halo data from file
                            subhalo = utilities.read_halo_data(thisSub,
                                                               inputpath,
                                                               snapshot,
                                                               hdf_part_key,
                                                               ini_vlcoeff,
                                                               boxsize,
                                                               soft, redshift,
                                                               G,
                                                               cosmo)

                            read_end = time.time()

                            if profile:
                                prof_d["Reading"]["Start"].append(read_start)
                                prof_d["Reading"]["End"].append(read_end)

                            task_start = time.time()

                            # Do the work here
                            result = get_real_host_halos(subhalo, boxsize,
                                                         vlinkl_indp * 8 ** (
                                                                 1 / 6),
                                                         linkl, decrement,
                                                         redshift, G, h,
                                                         soft, min_vlcoeff,
                                                         cosmo)

                            # Save results
                            while len(result) > 0:
                                key, res = result.popitem()
                                sub_results[(rank, subhaloID)] = res

                                subhaloID += 1

                            task_end = time.time()

                            if profile:
                                prof_d["Sub-Phase"]["Start"].append(task_start)
                                prof_d["Sub-Phase"]["End"].append(task_end)

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
            thisTask = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:

                read_start = time.time()

                thisTask.sort()

                # Get halo data from file
                halo = utilities.read_halo_data(thisTask, inputpath,
                                                snapshot, hdf_part_key,
                                                ini_vlcoeff, boxsize,
                                                soft, redshift, G, cosmo)

                read_end = time.time()

                if profile:
                    prof_d["Reading"]["Start"].append(read_start)
                    prof_d["Reading"]["End"].append(read_end)

                task_start = time.time()

                # Do the work here
                result = get_real_host_halos(halo, boxsize, vlinkl_indp,
                                             linkl, decrement, redshift, G, h,
                                             soft, min_vlcoeff, cosmo)

                # Save results
                for res in result:
                    results[(rank, haloID)] = result[res]

                    haloID += 1

                task_end = time.time()

                if profile:
                    prof_d["Host-Phase"]["Start"].append(task_start)
                    prof_d["Host-Phase"]["End"].append(task_end)

                if findsubs:

                    spatial_sub_results = {}

                    # Loop over results getting spatial halos
                    while len(result) > 0:

                        read_start = time.time()

                        key, res = result.popitem()

                        thishalo_pids = np.sort(res["pids"])

                        # Open hdf5 file
                        hdf = h5py.File(inputpath + snapshot + ".hdf5", 'r')

                        # Get the position and velocity of each
                        # particle in this rank
                        subhalo_poss = hdf[hdf_part_key]['part_pos'][
                                       thishalo_pids, :]

                        hdf.close()

                        read_end = time.time()

                        if profile:
                            prof_d["Reading"]["Start"].append(read_start)
                            prof_d["Reading"]["End"].append(read_end)

                        task_start = time.time()

                        # Do the work here
                        sub_result = get_sub_halos(thishalo_pids,
                                                   subhalo_poss,
                                                   sub_linkl)

                        while len(sub_result) > 0:
                            key, res = sub_result.popitem()
                            spatial_sub_results[subhaloID] = res

                            subhaloID += 1

                        task_end = time.time()

                        if profile:
                            prof_d["Sub-Spatial"]["Start"].append(task_start)
                            prof_d["Sub-Spatial"]["End"].append(task_end)

                    # Loop over spatial subhalos
                    while len(spatial_sub_results) > 0:

                        read_start = time.time()

                        key, thisSub = spatial_sub_results.popitem()

                        thisSub.sort()

                        # Get halo data from file
                        subhalo = utilities.read_halo_data(thisSub,
                                                           inputpath,
                                                           snapshot,
                                                           hdf_part_key,
                                                           ini_vlcoeff,
                                                           boxsize,
                                                           soft, redshift,
                                                           G,
                                                           cosmo)

                        read_end = time.time()

                        if profile:
                            prof_d["Reading"]["Start"].append(read_start)
                            prof_d["Reading"]["End"].append(read_end)

                        task_start = time.time()

                        # Do the work here
                        result = get_real_host_halos(subhalo, boxsize,
                                                     vlinkl_indp * 8 ** (
                                                             1 / 6),
                                                     linkl, decrement,
                                                     redshift, G, h,
                                                     soft, min_vlcoeff,
                                                     cosmo)

                        # Save results
                        while len(result) > 0:
                            key, res = result.popitem()
                            sub_results[(rank, subhaloID)] = res

                            subhaloID += 1

                        task_end = time.time()

                        if profile:
                            prof_d["Sub-Phase"]["Start"].append(task_start)
                            prof_d["Sub-Phase"]["End"].append(task_end)

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)

    # Collect child process results
    collect_start = time.time()
    collected_results = comm.gather(results, root=0)
    sub_collected_results = comm.gather(sub_results, root=0)

    if profile and rank != 0:
        prof_d["Collecting"]["Start"].append(collect_start)
        prof_d["Collecting"]["End"].append(time.time())

    if rank == 0:

        # If profiling enable plot the number of halos on each rank
        if profile:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.bar(np.arange(len(collected_results)),
                   [len(res) for res in collected_results],
                   color="b", edgecolor="b")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Number of halos computed")
            fig.savefig(profile_path + "/plots/halos_computed_"
                        + str(snapshot) + ".png")
        if profile and findsubs:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.bar(np.arange(len(sub_collected_results)),
                   [len(res) for res in sub_collected_results],
                   color="r", edgecolor="r")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Number of subhalos computed")
            fig.savefig(profile_path + "/plots/subhalos_computed_"
                        + str(snapshot) + ".png")

        newPhaseID = 0
        newPhaseSubID = 0

        phase_part_haloids = np.full((np.sum(npart), 2), -2, dtype=np.int32)

        # Collect host halo results
        results_dict = {}
        for halo_task in collected_results:
            for halo in halo_task:
                results_dict[(halo, newPhaseID)] = halo_task[halo]
                pids = halo_task[halo]['pids']
                haloID_dict[(halo, newPhaseID)] = newPhaseID
                phase_part_haloids[pids, 0] = newPhaseID
                newPhaseID += 1

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

        if verbose:
            print("Combining the results took", time.time() - collect_start,
                  "seconds")
            print("Results memory size", sys.getsizeof(results_dict), "bytes")
            print("This Rank:", rank)
            # print(hp.heap())

        if profile:
            prof_d["Collecting"]["Start"].append(collect_start)
            prof_d["Collecting"]["End"].append(time.time())

        write_start = time.time()

        # Find the halos with 10 or more particles by finding the unique IDs in the particle
        # halo ids array and finding those IDs that are assigned to 10 or more particles
        unique, counts = np.unique(phase_part_haloids[:, 0],
                                   return_counts=True)
        unique_haloids = unique[np.where(counts >= 10)]

        # Remove the null -2 value for single particle halos
        unique_haloids = unique_haloids[np.where(unique_haloids != -2)]

        # Print the number of halos found by the halo finder in >10, >100, >1000, >10000 criteria
        print(
            "=========================== Phase halos ===========================")
        print(unique_haloids.size, 'halos found with 10 or more particles')
        print(unique[np.where(counts >= 15)].size - 1,
              'halos found with 15 or more particles')
        print(unique[np.where(counts >= 20)].size - 1,
              'halos found with 20 or more particles')
        print(unique[np.where(counts >= 50)].size - 1,
              'halos found with 50 or more particles')
        print(unique[np.where(counts >= 100)].size - 1,
              'halos found with 100 or more particles')
        print(unique[np.where(counts >= 500)].size - 1,
              'halos found with 500 or more particles')
        print(unique[np.where(counts >= 1000)].size - 1,
              'halos found with 1000 or more particles')
        print(unique[np.where(counts >= 10000)].size - 1,
              'halos found with 10000 or more particles')

        # Find the halos with 10 or more particles by finding the unique IDs in the particle
        # halo ids array and finding those IDs that are assigned to 10 or more particles
        unique, counts = np.unique(phase_part_haloids[:, 1],
                                   return_counts=True)
        unique_haloids = unique[np.where(counts >= 10)]

        # Remove the null -2 value for single particle halos
        unique_haloids = unique_haloids[np.where(unique_haloids != -2)]

        # Print the number of halos found by the halo finder in >10, >100, >1000, >10000 criteria
        print(
            "=========================== Phase subhalos ===========================")
        print(unique_haloids.size, 'halos found with 10 or more particles')
        print(unique[np.where(counts >= 15)].size - 1,
              'halos found with 15 or more particles')
        print(unique[np.where(counts >= 20)].size - 1,
              'halos found with 20 or more particles')
        print(unique[np.where(counts >= 50)].size - 1,
              'halos found with 50 or more particles')
        print(unique[np.where(counts >= 100)].size - 1,
              'halos found with 100 or more particles')
        print(unique[np.where(counts >= 500)].size - 1,
              'halos found with 500 or more particles')
        print(unique[np.where(counts >= 1000)].size - 1,
              'halos found with 1000 or more particles')
        print(unique[np.where(counts >= 10000)].size - 1,
              'halos found with 10000 or more particles')

        # ============================= Write out data =============================

        # Set up arrays to store subhalo results
        nhalo = newPhaseID
        halo_nparts = np.full(nhalo, -1, dtype=int)
        halo_masses = np.full(nhalo, -1, dtype=float)
        halo_type_masses = np.full((nhalo, 6), -1, dtype=float)
        mean_poss = np.full((nhalo, 3), -1, dtype=float)
        mean_vels = np.full((nhalo, 3), -1, dtype=float)
        reals = np.full(nhalo, 0, dtype=bool)
        halo_energies = np.full(nhalo, -1, dtype=float)
        KEs = np.full(nhalo, -1, dtype=float)
        GEs = np.full(nhalo, -1, dtype=float)
        nsubhalos = np.zeros(nhalo, dtype=float)
        rms_rs = np.zeros(nhalo, dtype=float)
        rms_vrs = np.zeros(nhalo, dtype=float)
        veldisp1ds = np.zeros((nhalo, 3), dtype=float)
        veldisp3ds = np.zeros(nhalo, dtype=float)
        vmaxs = np.zeros(nhalo, dtype=float)
        hmrs = np.zeros(nhalo, dtype=float)
        hmvrs = np.zeros(nhalo, dtype=float)
        exit_vlcoeff = np.zeros(nhalo, dtype=float)

        if findsubs:

            # Set up arrays to store host results
            nsubhalo = newPhaseSubID
            subhalo_nparts = np.full(nsubhalo, -1, dtype=int)
            subhalo_masses = np.full(nsubhalo, -1, dtype=float)
            subhalo_type_masses = np.full((nsubhalo, 6), -1, dtype=float)
            sub_mean_poss = np.full((nsubhalo, 3), -1, dtype=float)
            sub_mean_vels = np.full((nsubhalo, 3), -1, dtype=float)
            sub_reals = np.full(nsubhalo, 0, dtype=bool)
            subhalo_energies = np.full(nsubhalo, -1, dtype=float)
            sub_KEs = np.full(nsubhalo, -1, dtype=float)
            sub_GEs = np.full(nsubhalo, -1, dtype=float)
            host_ids = np.full(nsubhalo, np.nan, dtype=int)
            sub_rms_rs = np.zeros(nsubhalo, dtype=float)
            sub_rms_vrs = np.zeros(nsubhalo, dtype=float)
            sub_veldisp1ds = np.zeros((nsubhalo, 3), dtype=float)
            sub_veldisp3ds = np.zeros(nsubhalo, dtype=float)
            sub_vmaxs = np.zeros(nsubhalo, dtype=float)
            sub_hmrs = np.zeros(nsubhalo, dtype=float)
            sub_hmvrs = np.zeros(nsubhalo, dtype=float)
            sub_exit_vlcoeff = np.zeros(nhalo, dtype=float)

        else:

            # Set up dummy subhalo results
            subhalo_nparts = None
            subhalo_masses = None
            subhalo_type_masses = None
            sub_mean_poss = None
            sub_mean_vels = None
            sub_reals = None
            subhalo_energies = None
            sub_KEs = None
            sub_GEs = None
            host_ids = None
            sub_rms_rs = None
            sub_rms_vrs = None
            sub_veldisp1ds = None
            sub_veldisp3ds = None
            sub_vmaxs = None
            sub_hmrs = None
            sub_hmvrs = None
            sub_exit_vlcoeff = None

        # TODO: nPart should also be split by particle type

        # Create the root group
        snap = h5py.File(savepath + 'halos_' + str(snapshot) + '.hdf5', 'w')

        # Assign simulation attributes to the root of the z=0 snapshot
        snap.attrs[
            'snap_nPart'] = npart  # number of particles in the simulation
        snap.attrs['boxsize'] = boxsize  # box length along each axis
        snap.attrs['h'] = h  # 'little h' (hubble constant parametrisation)

        # Assign snapshot attributes
        snap.attrs['linking_length'] = linkl  # host halo linking length
        # snap.attrs['rhocrit'] = rhocrit  # critical density parameter
        snap.attrs['redshift'] = redshift
        # snap.attrs['time'] = t

        halo_ids = np.arange(newPhaseID, dtype=int)

        for res in list(results_dict.keys()):
            halo_res = results_dict.pop(res)
            halo_id = haloID_dict[res]
            halo_pids = halo_res['pids']

            mean_poss[halo_id, :] = halo_res['mean_halo_pos']
            mean_vels[halo_id, :] = halo_res['mean_halo_vel']
            halo_nparts[halo_id] = halo_res['npart']
            halo_masses[halo_id] = halo_res["halo_mass"]
            halo_type_masses[halo_id, :] = halo_res["halo_ptype_mass"]
            reals[halo_id] = halo_res['real']
            halo_energies[halo_id] = halo_res['halo_energy']
            KEs[halo_id] = halo_res['KE']
            GEs[halo_id] = halo_res['GE']
            rms_rs[halo_id] = halo_res["rms_r"]
            rms_vrs[halo_id] = halo_res["rms_vr"]
            veldisp1ds[halo_id, :] = halo_res["veldisp1d"]
            veldisp3ds[halo_id] = halo_res["veldisp3d"]
            vmaxs[halo_id] = halo_res["vmax"]
            hmrs[halo_id] = halo_res["hmr"]
            hmvrs[halo_id] = halo_res["hmvr"]
            exit_vlcoeff[halo_id] = halo_res["vlcoeff"]

            # Create datasets in the current halo's group in the HDF5 file
            halo = snap.create_group(str(halo_id))  # create halo group
            halo.create_dataset('Halo_Part_IDs', shape=halo_pids.shape,
                                dtype=int,
                                data=halo_pids)  # halo particle ids

        # Save halo property arrays
        snap.create_dataset('halo_IDs',
                            shape=halo_ids.shape,
                            dtype=int,
                            data=halo_ids,
                            compression='gzip')
        snap.create_dataset('mean_positions',
                            shape=mean_poss.shape,
                            dtype=float,
                            data=mean_poss,
                            compression='gzip')
        snap.create_dataset('mean_velocities',
                            shape=mean_vels.shape,
                            dtype=float,
                            data=mean_vels,
                            compression='gzip')
        snap.create_dataset('rms_spatial_radius',
                            shape=rms_rs.shape,
                            dtype=rms_rs.dtype,
                            data=rms_rs,
                            compression='gzip')
        snap.create_dataset('rms_velocity_radius',
                            shape=rms_vrs.shape,
                            dtype=rms_vrs.dtype,
                            data=rms_vrs,
                            compression='gzip')
        snap.create_dataset('1D_velocity_dispersion',
                            shape=veldisp1ds.shape,
                            dtype=veldisp1ds.dtype,
                            data=veldisp1ds,
                            compression='gzip')
        snap.create_dataset('3D_velocity_dispersion',
                            shape=veldisp3ds.shape,
                            dtype=veldisp3ds.dtype,
                            data=veldisp3ds,
                            compression='gzip')
        snap.create_dataset('nparts',
                            shape=halo_nparts.shape,
                            dtype=int,
                            data=halo_nparts,
                            compression='gzip')
        snap.create_dataset('total_masses',
                            shape=halo_masses.shape,
                            dtype=float,
                            data=halo_masses,
                            compression='gzip')
        snap.create_dataset('masses',
                            shape=halo_type_masses.shape,
                            dtype=float,
                            data=halo_type_masses,
                            compression='gzip')
        snap.create_dataset('real_flag',
                            shape=reals.shape,
                            dtype=bool,
                            data=reals,
                            compression='gzip')
        snap.create_dataset('halo_total_energies',
                            shape=halo_energies.shape,
                            dtype=float,
                            data=halo_energies,
                            compression='gzip')
        snap.create_dataset('halo_kinetic_energies',
                            shape=KEs.shape,
                            dtype=float,
                            data=KEs,
                            compression='gzip')
        snap.create_dataset('halo_gravitational_energies',
                            shape=GEs.shape,
                            dtype=float,
                            data=GEs,
                            compression='gzip')
        snap.create_dataset('v_max',
                            shape=vmaxs.shape,
                            dtype=vmaxs.dtype,
                            data=vmaxs,
                            compression='gzip')
        snap.create_dataset('half_mass_radius',
                            shape=hmrs.shape,
                            dtype=hmrs.dtype,
                            data=hmrs,
                            compression='gzip')
        snap.create_dataset('half_mass_velocity_radius',
                            shape=hmvrs.shape,
                            dtype=hmvrs.dtype,
                            data=hmvrs,
                            compression='gzip')
        snap.create_dataset('exit_vlcoeff',
                            shape=exit_vlcoeff.shape,
                            dtype=exit_vlcoeff.dtype,
                            data=exit_vlcoeff,
                            compression='gzip')

        # Assign the full halo IDs array to the snapshot group
        snap.create_dataset('particle_halo_IDs',
                            shape=phase_part_haloids.shape,
                            dtype=int,
                            data=phase_part_haloids,
                            compression='gzip')

        # Get how many halos were found be real
        print("Halos with unbound energies after phase space iteration:",
              halo_ids.size - halo_ids[reals].size, "of", halo_ids.size)

        if findsubs:

            subhalo_ids = np.arange(newPhaseSubID, dtype=int)

            # Create subhalo group
            sub_root = snap.create_group('Subhalos')

            for res in list(sub_results_dict.keys()):
                subhalo_res = sub_results_dict.pop(res)
                subhalo_id = subhaloID_dict[res]
                subhalo_pids = subhalo_res['pids']
                host = np.unique(phase_part_haloids[subhalo_pids, 0])

                assert len(host) == 1, \
                    "subhalo is contained in multiple hosts, " \
                    "this should not be possible"

                sub_mean_poss[subhalo_id, :] = subhalo_res['mean_halo_pos']
                sub_mean_vels[subhalo_id, :] = subhalo_res['mean_halo_vel']
                subhalo_nparts[subhalo_id] = subhalo_res['npart']
                subhalo_masses[subhalo_id] = subhalo_res["halo_mass"]
                subhalo_type_masses[subhalo_id, :] = subhalo_res[
                    "halo_ptype_mass"]
                sub_reals[subhalo_id] = subhalo_res['real']
                subhalo_energies[subhalo_id] = subhalo_res['halo_energy']
                sub_KEs[subhalo_id] = subhalo_res['KE']
                sub_GEs[subhalo_id] = subhalo_res['GE']
                host_ids[subhalo_id] = host
                nsubhalos[host] += 1
                sub_rms_rs[subhalo_id] = subhalo_res["rms_r"]
                sub_rms_vrs[subhalo_id] = subhalo_res["rms_vr"]
                sub_veldisp1ds[subhalo_id, :] = subhalo_res["veldisp1d"]
                sub_veldisp3ds[subhalo_id] = subhalo_res["veldisp3d"]
                sub_vmaxs[subhalo_id] = subhalo_res["vmax"]
                sub_hmrs[subhalo_id] = subhalo_res["hmr"]
                sub_hmvrs[subhalo_id] = subhalo_res["hmvr"]
                sub_exit_vlcoeff[subhalo_id] = subhalo_res["vlcoeff"]

                # Create subhalo group
                subhalo = sub_root.create_group(str(subhalo_id))
                subhalo.create_dataset('Halo_Part_IDs',
                                       shape=subhalo_pids.shape,
                                       dtype=int,
                                       data=subhalo_pids)

            # Save halo property arrays
            sub_root.create_dataset('subhalo_IDs',
                                    shape=subhalo_ids.shape,
                                    dtype=int,
                                    data=subhalo_ids,
                                    compression='gzip')
            sub_root.create_dataset('host_IDs',
                                    shape=host_ids.shape,
                                    dtype=int, data=host_ids,
                                    compression='gzip')
            sub_root.create_dataset('mean_positions',
                                    shape=sub_mean_poss.shape,
                                    dtype=float,
                                    data=sub_mean_poss,
                                    compression='gzip')
            sub_root.create_dataset('mean_velocities',
                                    shape=sub_mean_vels.shape,
                                    dtype=float,
                                    data=sub_mean_vels,
                                    compression='gzip')
            sub_root.create_dataset('rms_spatial_radius',
                                    shape=sub_rms_rs.shape,
                                    dtype=sub_rms_rs.dtype,
                                    data=sub_rms_rs,
                                    compression='gzip')
            sub_root.create_dataset('rms_velocity_radius',
                                    shape=sub_rms_vrs.shape,
                                    dtype=sub_rms_vrs.dtype,
                                    data=sub_rms_vrs,
                                    compression='gzip')
            sub_root.create_dataset('1D_velocity_dispersion',
                                    shape=sub_veldisp1ds.shape,
                                    dtype=sub_veldisp1ds.dtype,
                                    data=sub_veldisp1ds,
                                    compression='gzip')
            sub_root.create_dataset('3D_velocity_dispersion',
                                    shape=sub_veldisp3ds.shape,
                                    dtype=sub_veldisp3ds.dtype,
                                    data=sub_veldisp3ds,
                                    compression='gzip')
            sub_root.create_dataset('nparts',
                                    shape=subhalo_nparts.shape,
                                    dtype=int,
                                    data=subhalo_nparts,
                                    compression='gzip')
            sub_root.create_dataset('total_masses',
                                    shape=subhalo_masses.shape,
                                    dtype=float,
                                    data=subhalo_masses,
                                    compression='gzip')
            sub_root.create_dataset('masses',
                                    shape=subhalo_type_masses.shape,
                                    dtype=float,
                                    data=subhalo_type_masses,
                                    compression='gzip')
            sub_root.create_dataset('real_flag',
                                    shape=sub_reals.shape,
                                    dtype=bool,
                                    data=sub_reals,
                                    compression='gzip')
            sub_root.create_dataset('halo_total_energies',
                                    shape=subhalo_energies.shape,
                                    dtype=float,
                                    data=subhalo_energies,
                                    compression='gzip')
            sub_root.create_dataset('halo_kinetic_energies',
                                    shape=sub_KEs.shape,
                                    dtype=float,
                                    data=sub_KEs,
                                    compression='gzip')
            sub_root.create_dataset('halo_gravitational_energies',
                                    shape=sub_GEs.shape,
                                    dtype=float,
                                    data=sub_GEs,
                                    compression='gzip')
            sub_root.create_dataset('v_max',
                                    shape=sub_vmaxs.shape,
                                    dtype=sub_vmaxs.dtype,
                                    data=sub_vmaxs,
                                    compression='gzip')
            sub_root.create_dataset('half_mass_radius',
                                    shape=sub_hmrs.shape,
                                    dtype=sub_hmrs.dtype,
                                    data=sub_hmrs,
                                    compression='gzip')
            sub_root.create_dataset('half_mass_velocity_radius',
                                    shape=sub_hmvrs.shape,
                                    dtype=sub_hmvrs.dtype,
                                    data=sub_hmvrs,
                                    compression='gzip')
            sub_root.create_dataset('exit_vlcoeff',
                                    shape=sub_exit_vlcoeff.shape,
                                    dtype=sub_exit_vlcoeff.dtype,
                                    data=sub_exit_vlcoeff,
                                    compression='gzip')

        snap.create_dataset('occupancy',
                            shape=nsubhalos.shape,
                            dtype=nsubhalos.dtype,
                            data=nsubhalos,
                            compression='gzip')

        snap.close()

        if profile:
            prof_d["Writing"]["Start"].append(write_start)
            prof_d["Writing"]["End"].append(time.time())

        # assert -1 not in np.unique(KEs), "halo ids are not sequential!"

    if profile:
        prof_d["END"] = time.time()

        with open(profile_path + "Halo_" + str(rank) + '_'
                  + snapshot + '.pck', 'wb') as pfile:
            pickle.dump(prof_d, pfile)
