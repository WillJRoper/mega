from scipy.spatial import cKDTree
from collections import defaultdict
from guppy import hpy; hp = hpy()
import itertools
import numpy as np
import mpi4py
from mpi4py import MPI
mpi4py.rc.recv_mprobe = False
import astropy.constants as const
import astropy.units as u
import time
import h5py
import gc
import sys
import utilities


# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object


def find_halos(tree, pos, linkl, npart):
    """ A function which creates a KD-Tree using scipy.CKDTree and queries it to find particles
    neighbours within a linking length. From This neighbour information particles are assigned
    halo IDs and then returned.
    :param pos: The particle position vectors array.
    :param npart: The number of particles in the simulation.
    :param boxsize: The length of the simulation box along one axis.
    :param batchsize: The batchsize for each query to the KD-Tree (see Docs for more information).
    :param linkl: The linking length.
    :param debug_npart: Number of particles to sort during debugging if required.
    :return: part_haloids: The array of halo IDs assigned to each particle (where the index is the particle ID)
             assigned_parts: A dictionary containing the particle IDs assigned to each halo.
             final_halo_ids: An array of final halo IDs (where the index is the initial halo ID and value is the
             final halo ID.
             query_func: The tree query object assigned to a variable.
    """

    # =============== Initialise The Halo Finder Variables/Arrays and The KD-Tree ===============

    # Initialise the arrays and dictionaries for storing halo data
    part_haloids = np.full(npart, -1, dtype=np.int32)  # halo ID containing each particle
    assigned_parts = defaultdict(set)  # dictionary to store the particles in a particular halo
    # A dictionary where each key is an initial halo ID and the item is the halo IDs it has been linked with
    linked_halos_dict = defaultdict(set)
    final_halo_ids = np.full(npart, -1, dtype=np.int32)  # final halo ID of linked halos (index is initial halo ID)

    # Initialise the halo ID counter (IDs start at 0)
    ihaloid = -1

    # Assign the query object to a variable to save time on repeated calls
    query_func = tree.query_ball_point

    # =============== Assign Particles To Initial Halos ===============

    # Query the tree returning a list of lists
    query = query_func(pos, r=linkl)

    # Loop through query results assigning initial halo IDs
    for query_part_inds in iter(query):

        # Convert the particle index list to an array for ease of use
        query_part_inds = np.array(query_part_inds, copy=False, dtype=int)

        # Assert that the query particle is returned by the tree query. Otherwise the program fails
        assert query_part_inds.size != 0, 'Must always return particle that you are sitting on'

        # Find only the particles not already in a halo
        new_parts = query_part_inds[np.where(part_haloids[query_part_inds] == -1)]

        # If only one particle is returned by the query and it is new it is a 'single particle halo'
        if new_parts.size == query_part_inds.size == 1:

            # Assign the 'single particle halo' halo ID to the particle
            part_haloids[new_parts] = -2

        # If all particles are new increment the halo ID and assign a new halo
        elif new_parts.size == query_part_inds.size:

            # Increment the halo ID by 1 (initialising a new halo)
            ihaloid += 1

            # Assign the new halo ID to the particles
            part_haloids[new_parts] = ihaloid
            assigned_parts[ihaloid] = set(new_parts)

            # Assign the final halo ID to be the newly assigned halo ID
            final_halo_ids[ihaloid] = ihaloid
            linked_halos_dict[ihaloid] = {ihaloid}

        else:

            # ===== Get the 'final halo ID value' =====

            # Extract the IDs of halos returned by the query
            contained_halos = part_haloids[query_part_inds]

            # Get only the unique halo IDs
            uni_cont_halos = np.unique(contained_halos)

            # # Assure no single particle halos are included in the query results
            # assert any(uni_cont_halos != -2), 'Single particle halos should never be found'

            # Remove any unassigned halos
            uni_cont_halos = uni_cont_halos[np.where(uni_cont_halos != -1)]

            # If there is only one halo ID returned avoid the slower code to combine IDs
            if uni_cont_halos.size == 1:

                # Get the list of linked halos linked to the current halo from the linked halo dictionary
                linked_halos = linked_halos_dict[uni_cont_halos[0]]

            else:

                # Get all the linked halos from dictionary so as to not miss out any halos IDs that are linked
                # but not returned by this particular query
                linked_halos_set = set()  # initialise linked halo set
                linked_halos = linked_halos_set.union(*[linked_halos_dict.get(halo) for halo in uni_cont_halos])

            # Find the minimum halo ID to make the final halo ID
            final_ID = min(linked_halos)

            # Assign the linked halos to all the entries in the linked halos dictionary
            linked_halos_dict.update(dict.fromkeys(list(linked_halos), linked_halos))

            # Assign the final halo ID array entries
            final_halo_ids[list(linked_halos)] = final_ID

            # Assign new particles to the particle halo IDs array with the final ID
            part_haloids[new_parts] = final_ID

            # Assign the new particles to the final ID in halo dictionary entry
            assigned_parts[final_ID].update(new_parts)

    # =============== Reassign All Halos To Their Final Halo ID ===============

    # Loop over initial halo IDs reassigning them to the final halo ID
    for halo_id in list(assigned_parts.keys()):

        # Extract the final halo value
        final_ID = final_halo_ids[halo_id]

        # Assign this final ID to all the particles in the initial halo ID
        part_haloids[list(assigned_parts[halo_id])] = final_ID
        assigned_parts[final_ID].update(assigned_parts[halo_id])

        # Remove non final ID entry from dictionary to save memory
        if halo_id != final_ID:
            del assigned_parts[halo_id]

    return part_haloids, assigned_parts


def find_subhalos(halo_pos, sub_linkl):
    """ A function that finds subhalos within host halos by applying the same KD-Tree algorithm at a
    higher overdensity.
    :param halo_pos: The position vectors of particles within the host halo.
    :param sub_llcoeff: The linking length coefficient used to define a subhalo.
    :param boxsize: The length of the simulation box along one axis.
    :param npart: The number of particles in the simulation.
    :return: part_subhaloids: The array of subhalo IDs assigned to each particle in the host halo
             (where the index is the particle ID).
             assignedsub_parts: A dictionary containing the particle IDs assigned to each subhalo.
    """

    # =============== Initialise The Halo Finder Variables/Arrays and The KD-Tree ===============

    # Initialise arrays and dictionaries for storing subhalo data
    part_subhaloids = np.full(halo_pos.shape[0], -1, dtype=int)  # subhalo ID of the halo each particle is in
    assignedsub_parts = defaultdict(set)  # Dictionary to store the particles in a particular subhalo
    # A dictionary where each key is an initial subhalo ID and the item is the subhalo IDs it has been linked with
    linked_subhalos_dict = defaultdict(set)
    # Final subhalo ID of linked halos (index is initial subhalo ID)
    final_subhalo_ids = np.full(halo_pos.shape[0], -1, dtype=int)

    # Initialise subhalo ID counter (IDs start at 0)
    isubhaloid = -1

    npart = halo_pos.shape[0]

    # Build the halo kd tree
    # *** Note: Contrary to CKDTree documentation compact_nodes=False and balanced_tree=False results in
    # faster queries (documentation recommends compact_nodes=True and balanced_tree=True)***
    tree = cKDTree(halo_pos, leafsize=16, compact_nodes=False, balanced_tree=False)

    if npart > 10000:

        # Define an array of limits for looping defined by the batchsize
        limits = np.linspace(0, npart, int(npart / 10000) + 1, dtype=int)

    else:

        limits = [0, npart]

    # ===================== Sort Through The Results Assigning Subhalos =====================

    for ind, limit in enumerate(limits[:-1]):

        query_parts = halo_pos[limit:limits[ind + 1]]

        if query_parts.size == 0:
            continue

        query = tree.query_ball_point(query_parts, r=sub_linkl)

        # Loop through query results
        for query_part_inds in iter(query):

            # Convert the particle index list to an array for ease of use.
            query_part_inds = np.array(query_part_inds, copy=False, dtype=int)

            # Assert that the query particle is returned by the tree query. Otherwise the program fails
            assert query_part_inds.size != 0, 'Must always return particle that you are sitting on'

            # Find only the particles not already in a halo
            new_parts = query_part_inds[np.where(part_subhaloids[query_part_inds] == -1)]

            # If only one particle is returned by the query and it is new it is a 'single particle subhalo'
            if new_parts.size == query_part_inds.size == 1:

                # Assign the 'single particle subhalo' subhalo ID to the particle
                part_subhaloids[new_parts] = -2

            # If all particles are new increment the subhalo ID and assign a new subhalo ID
            elif new_parts.size == query_part_inds.size:

                # Increment the subhalo ID by 1 (initialise new halo)
                isubhaloid += 1

                # Assign the subhalo ID to the particles
                part_subhaloids[new_parts] = isubhaloid
                assignedsub_parts[isubhaloid] = set(new_parts)

                # Assign the final subhalo ID to be the newly assigned subhalo ID
                final_subhalo_ids[isubhaloid] = isubhaloid
                linked_subhalos_dict[isubhaloid] = {isubhaloid}

            else:

                # ===== Get the 'final subhalo ID value' =====

                # Extract the IDs of subhalos returned by the query
                contained_subhalos = part_subhaloids[query_part_inds]

                # Return only the unique subhalo IDs
                uni_cont_subhalos = np.unique(contained_subhalos)

                # # Assure no single particles are returned by the query
                # assert any(uni_cont_subhalos != -2), 'Single particle halos should never be found'

                # Remove any unassigned subhalos
                uni_cont_subhalos = uni_cont_subhalos[np.where(uni_cont_subhalos != -1)]

                # If there is only one subhalo ID returned avoid the slower code to combine IDs
                if uni_cont_subhalos.size == 1:

                    # Get the list of linked subhalos linked to the current subhalo from the linked subhalo dictionary
                    linked_subhalos = linked_subhalos_dict[uni_cont_subhalos[0]]

                else:

                    # Get all linked subhalos from the dictionary so as to not miss out any subhalos IDs that are linked
                    # but not returned by this particular query
                    linked_subhalos_set = set()  # initialise linked subhalo set
                    linked_subhalos = linked_subhalos_set.union(*[linked_subhalos_dict.get(subhalo)
                                                                  for subhalo in uni_cont_subhalos])

                # Find the minimum subhalo ID to make the final subhalo ID
                final_ID = min(linked_subhalos)

                # Assign the linked subhalos to all the entries in the linked subhalos dict
                linked_subhalos_dict.update(dict.fromkeys(list(linked_subhalos), linked_subhalos))

                # Assign the final subhalo array
                final_subhalo_ids[list(linked_subhalos)] = final_ID

                # Assign new parts to the subhalo IDs with the final ID
                part_subhaloids[new_parts] = final_ID

                # Assign the new particles to the final ID particles in subhalo dictionary entry
                assignedsub_parts[final_ID].update(new_parts)

    # =============== Reassign All Subhalos To Their Final Subhalo ID ===============

    # Loop over initial subhalo IDs reassigning them to the final subhalo ID
    for subhalo_id in list(assignedsub_parts.keys()):

        # Extract the final subhalo value
        final_ID = final_subhalo_ids[subhalo_id]

        # Assign this final ID to all the particles in the initial subhalo ID
        part_subhaloids[list(assignedsub_parts[subhalo_id])] = final_ID
        assignedsub_parts[final_ID].update(assignedsub_parts[subhalo_id])

        # Remove non final ID entry from dictionary to save memory
        if subhalo_id != final_ID:
            assignedsub_parts.pop(subhalo_id)

    return part_subhaloids, assignedsub_parts


def find_phase_space_halos(halo_phases, linkl, vlinkl):

    # =============== Initialise The Halo Finder Variables/Arrays and The KD-Tree ===============

    # Divide halo positions by the linking length and velocites by the velocity linking length
    halo_phases[:, :3] = halo_phases[:, :3] / linkl
    halo_phases[:, 3:] = halo_phases[:, 3:] / vlinkl

    # Initialise arrays and dictionaries for storing halo data
    phase_part_haloids = np.full(halo_phases.shape[0], -1, dtype=int)  # halo ID of the halo each particle is in
    phase_assigned_parts = {}  # Dictionary to store the particles in a particular halo
    # A dictionary where each key is an initial subhalo ID and the item is the subhalo IDs it has been linked with
    phase_linked_halos_dict = {}
    # Final halo ID of linked halos (index is initial halo ID)
    final_phasehalo_ids = np.full(halo_phases.shape[0], -1, dtype=int)

    # Initialise subhalo ID counter (IDs start at 0)
    ihaloid = -1

    # Initialise the halo kd tree in 6D phase space
    halo_tree = cKDTree(halo_phases, leafsize=16, compact_nodes=False, balanced_tree=False)

    npart = halo_phases.shape[0]

    if npart > 10000:

        # Define an array of limits for looping defined by the batchsize
        limits = np.linspace(0, npart, int(npart / 10000) + 1, dtype=int)

    else:

        limits = [0, npart]

    # =============== Query the tree ===============

    for ind, limit in enumerate(limits[:-1]):

        query_phases = halo_phases[limit:limits[ind + 1]]

        if query_phases.size == 0:
            continue

        query = halo_tree.query_ball_point(query_phases, r=np.sqrt(2))

        # Loop through query results assigning initial halo IDs
        for query_part_inds in iter(query):

            query_part_inds = np.array(query_part_inds, dtype=int)

            # Assert that the query particle is returned by the tree query. Otherwise the program fails
            # assert query_part_inds.size != 0, 'Must always return particle that you are sitting on'

            # Find only the particles not already in a halo
            new_parts = query_part_inds[np.where(phase_part_haloids[query_part_inds] < 0)]

            # If only one particle is returned by the query and it is new it is a 'single particle halo'
            if new_parts.size == query_part_inds.size == 1:

                # Assign the 'single particle halo' halo ID to the particle
                phase_part_haloids[new_parts] = -2

            # If all particles are new increment the halo ID and assign a new halo
            elif new_parts.size == query_part_inds.size:

                # Increment the halo ID by 1 (initialising a new halo)
                ihaloid += 1

                # Assign the new halo ID to the particles
                phase_part_haloids[new_parts] = ihaloid
                phase_assigned_parts[ihaloid] = set(new_parts)

                # Assign the final halo ID to be the newly assigned halo ID
                final_phasehalo_ids[ihaloid] = ihaloid
                phase_linked_halos_dict[ihaloid] = {ihaloid}

            else:

                # ===== Get the 'final halo ID value' =====

                # Extract the IDs of halos returned by the query
                contained_halos = phase_part_haloids[query_part_inds]

                # Get only the unique halo IDs
                uni_cont_halos = np.unique(contained_halos)

                # Remove any unassigned halos
                uni_cont_halos = uni_cont_halos[np.where(uni_cont_halos >= 0)]

                # If there is only one halo ID returned avoid the slower code to combine IDs
                if uni_cont_halos.size == 1:

                    # Get the list of linked halos linked to the current halo from the linked halo dictionary
                    linked_halos = phase_linked_halos_dict[uni_cont_halos[0]]

                elif uni_cont_halos.size == 0:
                    continue

                else:

                    # Get all the linked halos from dictionary so as to not miss out any halos IDs that are linked
                    # but not returned by this particular query
                    linked_halos_set = set()  # initialise linked halo set
                    linked_halos = linked_halos_set.union(*[phase_linked_halos_dict.get(halo)
                                                            for halo in uni_cont_halos])

                # Find the minimum halo ID to make the final halo ID
                final_ID = min(linked_halos)

                # Assign the linked halos to all the entries in the linked halos dictionary
                phase_linked_halos_dict.update(dict.fromkeys(list(linked_halos), linked_halos))

                # Assign the final halo ID array entries
                final_phasehalo_ids[list(linked_halos)] = final_ID

                # Assign new particles to the particle halo IDs array with the final ID
                phase_part_haloids[new_parts] = final_ID

                # Assign the new particles to the final ID in halo dictionary entry
                phase_assigned_parts[final_ID].update(new_parts)

    # =============== Reassign All Halos To Their Final Halo ID ===============

    # Loop over initial halo IDs reassigning them to the final halo ID
    for halo_id in list(phase_assigned_parts.keys()):

        # Extract the final halo value
        final_ID = final_phasehalo_ids[halo_id]

        # Assign this final ID to all the particles in the initial halo ID
        phase_part_haloids[list(phase_assigned_parts[halo_id])] = final_ID
        phase_assigned_parts[final_ID].update(phase_assigned_parts[halo_id])

        # Remove non final ID entry from dictionary to save memory
        if halo_id != final_ID:
            phase_assigned_parts.pop(halo_id)

    return phase_part_haloids, phase_assigned_parts


halo_energy_calc = utilities.halo_energy_calc_exact


def spatial_node_task(thisTask, pos, tree, linkl, npart):

    # =============== Run The Halo Finder And Reduce The Output ===============

    # Run the halo finder for this snapshot at the host linking length and get the spatial catalog
    task_part_haloids, task_assigned_parts = find_halos(tree, pos, linkl, npart)

    # Get the positions
    halo_pids = {}
    for halo in task_assigned_parts:
        part_inds = list(task_assigned_parts[halo])
        halo_pids[(thisTask, halo)] = part_inds

    return halo_pids


def get_real_host_halos(thisTask, pids, pos, vel, boxsize, vlinkl_halo_indp, linkl, pmass, vlcoeff, decrement,
                        redshift, G, h, soft, min_vlcoeff):

    # Extract halo data for this halo ID
    s_halo_pids = np.arange(len(pids), dtype=int)
    full_sim_halo_pids = np.array(list(pids))
    full_halo_poss = pos  # Positions *** NOTE: these are shifted below ***
    full_halo_vels = vel  # Velocities *** NOTE: these are shifted below ***
    full_halo_npart = s_halo_pids.size

    newID_iter = -9

    # =============== Compute mean positions and velocities and wrap the halos ===============

    # Define the comparison particle as the maximum position in the current dimension
    max_part_pos = full_halo_poss.max(axis=0)

    # Compute all the halo particle separations from the maximum position
    sep = max_part_pos - full_halo_poss

    # If any separations are greater than 50% the boxsize (i.e. the halo is split over the boundary)
    # bring the particles at the lower boundary together with the particles at the upper boundary
    # (ignores halos where constituent particles aren't separated by at least 50% of the boxsize)
    # *** Note: fails if halo's extent is greater than 50% of the boxsize in any dimension ***
    full_halo_poss[np.where(sep > 0.5 * boxsize)] += boxsize

    # Compute the shifted mean position in the dimension ixyz
    mean_halo_pos = full_halo_poss.mean(axis=0)

    # Centre the halos about the mean in the dimension ixyz
    full_halo_poss -= mean_halo_pos

    # Define the velocity space linking length
    vlinkl = vlcoeff * vlinkl_halo_indp * pmass ** (1 / 3) * full_halo_npart ** (1 / 3)

    # Define the phase space vectors for this halo
    halo_phases = np.concatenate((full_halo_poss, full_halo_vels), axis=1)

    # Query this halo in velocity space to split apart halos which are found to be distinct in velocity space
    result = find_phase_space_halos(halo_phases, linkl, vlinkl)
    phase_part_haloids, phase_assigned_parts = result

    # Find the halos with 10 or more particles by finding the unique IDs in the particle
    # halo ids array and finding those IDs that are assigned to 10 or more particles
    phase_unique, phase_counts = np.unique(phase_part_haloids, return_counts=True)
    unique_phase_haloids = phase_unique[np.where(phase_counts >= 10)]

    # Remove the null -2 value for single particle halos
    unique_phase_haloids = unique_phase_haloids[np.where(unique_phase_haloids >= 0)]

    extra_halo_pids = {}
    extra_halo_poss = {}
    extra_halo_vels = {}
    iter_vlcoeffs = {}
    results = {}

    # Loop over the halos returned from velocity space
    for pID in unique_phase_haloids:

        if len(phase_assigned_parts[pID]) == 0:
            continue

        # Extract halo data for this velocity space defined halo ID
        # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
        phalo_pids = np.array(list(phase_assigned_parts[pID]), dtype=int)
        halo_pids = s_halo_pids[phalo_pids]
        sim_halo_pids = full_sim_halo_pids[phalo_pids]
        halo_poss = pos[halo_pids, :]  # Positions *** NOTE: these are shifted below ***
        halo_vels = vel[halo_pids, :]  # Velocities *** NOTE: these are shifted below ***
        halo_npart = halo_pids.size

        # =============== Compute mean positions and velocities and wrap the halos ===============

        # Define the comparison particle as the maximum position in the current dimension
        max_part_pos = halo_poss.max(axis=0)

        # Compute all the halo particle separations from the maximum position
        sep = max_part_pos - halo_poss

        # If any separations are greater than 50% the boxsize (i.e. the halo is split over the boundary)
        # bring the particles at the lower boundary together with the particles at the upper boundary
        # (ignores halos where constituent particles aren't separated by at least 50% of the boxsize)
        # *** Note: fails if halo's extent is greater than 50% of the boxsize in any dimension ***
        halo_poss[np.where(sep > 0.5 * boxsize)] += boxsize

        # Compute the shifted mean position
        mean_halo_pos = halo_poss.mean(axis=0)

        # Centre the halos about the mean in the dimension ixyz
        halo_poss -= mean_halo_pos

        # Compute halo's energy
        halo_energy, KE, GE = halo_energy_calc(halo_poss, halo_vels, halo_npart, pmass, redshift, G, h, soft)

        new_vlcoeff = vlcoeff

        iter_halo_pids = halo_pids
        iter_sim_halo_pids = sim_halo_pids
        iter_halo_poss = halo_poss
        iter_halo_vels = halo_vels
        itercount = 0

        while KE / GE >= 1 and halo_npart >= 10 and new_vlcoeff >= 0.8:

            new_vlcoeff -= decrement

            # Define the phase space linking length
            vlinkl = new_vlcoeff * vlinkl_halo_indp * pmass ** (1 / 3) * halo_npart ** (1 / 3)

            # Define the phase space vectors for this halo
            halo_phases = np.concatenate((iter_halo_poss, iter_halo_vels), axis=1)

            # Query this halo in phase space to split apart halos which are found to
            # be distinct in phase space
            result = find_phase_space_halos(halo_phases, linkl, vlinkl)
            iter_part_haloids, iter_assigned_parts = result

            # Find the halos with 10 or more particles by finding the unique IDs in the particle
            # halo ids array and finding those IDs that are assigned to 10 or more particles
            iter_unique, iter_counts = np.unique(iter_part_haloids, return_counts=True)
            unique_iter_haloids = iter_unique[np.where(iter_counts >= 10)]
            iter_counts = iter_counts[np.where(iter_counts >= 10)]

            # Remove the null -2 value for single particle halos
            iter_counts = iter_counts[np.where(unique_iter_haloids >= 0)]
            unique_iter_haloids = unique_iter_haloids[np.where(unique_iter_haloids >= 0)]

            if unique_iter_haloids.size != 0:

                # Sort IDs by count
                unique_iter_haloids = unique_iter_haloids[np.argsort(iter_counts)]

                iterID = unique_iter_haloids[0]
                for iID in unique_iter_haloids[1:]:

                    # Store halo for testing as another task
                    extra_pids = np.array(list(iter_assigned_parts[iID]), dtype=int)
                    extra_halo_pids[newID_iter] = iter_sim_halo_pids[extra_pids]
                    extra_halo_poss[newID_iter] = iter_halo_poss[extra_pids, :]
                    extra_halo_vels[newID_iter] = iter_halo_vels[extra_pids, :]
                    iter_vlcoeffs[newID_iter] = new_vlcoeff - decrement
                    newID_iter -= 100

                # Extract halo data for this phase space defined halo ID
                # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
                phalo_pids = np.array(list(iter_assigned_parts[iterID]), dtype=int)
                iter_halo_pids = iter_halo_pids[phalo_pids]
                iter_sim_halo_pids = iter_sim_halo_pids[phalo_pids]
                iter_halo_poss = iter_halo_poss[phalo_pids, :]  # Positions *** NOTE: these are shifted below ***
                iter_halo_vels = iter_halo_vels[phalo_pids, :]  # Velocities *** NOTE: these are shifted below ***
                halo_npart = iter_halo_pids.size

                # Define the comparison particle as the maximum position in the current dimension
                max_part_pos = iter_halo_poss.max(axis=0)

                # Compute all the halo particle separations from the maximum position
                sep = max_part_pos - iter_halo_poss

                # If any separations are greater than 50% the boxsize (i.e. the halo is split over the boundary)
                # bring the particles at the lower boundary together with the particles at the upper boundary
                # (ignores halos where constituent particles aren't separated by at least 50% of the boxsize)
                # *** Note: fails if halo's extent is greater than 50% of the boxsize in any dimension ***
                iter_halo_poss[np.where(sep > 0.5 * boxsize)] += boxsize

                # Compute halo's energy
                halo_energy, KE, GE = halo_energy_calc(iter_halo_poss, iter_halo_vels, halo_npart,
                                                       pmass, redshift, G, h, soft)
                itercount += 1

            else:
                halo_npart = 0

        if halo_npart >= 10:

            # Extract halo data for this phase space defined halo ID
            # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
            sim_halo_pids = iter_sim_halo_pids
            halo_poss = iter_halo_poss  # Positions *** NOTE: these are shifted below ***
            halo_vels = iter_halo_vels  # Velocities *** NOTE: these are shifted below ***
            halo_npart = sim_halo_pids.size

            # =============== Compute mean positions and velocities and wrap the halos ===============

            # Compute the shifted mean position in the dimension ixyz
            mean_halo_pos = halo_poss.mean(axis=0)

            # Centre the halos about the mean in the dimension ixyz
            halo_poss -= mean_halo_pos

            # May need to wrap if the halo extends over the upper edge of the box
            mean_halo_pos = mean_halo_pos % boxsize

            # Compute the mean velocity in the dimension ixyz
            mean_halo_vel = halo_vels.mean(axis=0)

            if KE / GE <= 1 and vlcoeff >= min_vlcoeff:

                # Define realness flag
                real = True

                results[pID] = {'pids': sim_halo_pids, 'pos': halo_poss, 'vel': halo_vels,
                                'npart': halo_npart, 'real': real, 'mean_halo_pos': mean_halo_pos,
                                'mean_halo_vel': mean_halo_vel, 'halo_energy': halo_energy, 'KE': KE, 'GE': GE}

            else:

                # Define realness flag
                real = False

                results[pID] = {'pids': sim_halo_pids, 'pos': halo_poss, 'vel': halo_vels,
                                'npart': halo_npart, 'real': real, 'mean_halo_pos': mean_halo_pos,
                                'mean_halo_vel': mean_halo_vel, 'halo_energy': halo_energy, 'KE': KE, 'GE': GE}

    return thisTask, results, extra_halo_pids, extra_halo_poss, extra_halo_vels, iter_vlcoeffs


def get_sub_halos(thisTask, halo_pids, halo_pos, halo_vel, sub_linkl):

    # Do a spatial search for subhalos
    part_subhaloids, assignedsub_parts = find_subhalos(halo_pos, sub_linkl)

    # Get the positions
    subhalo_poss = {}
    subhalo_vels = {}
    subhalo_pids = {}
    for halo in assignedsub_parts:
        subhalo_poss[halo] = halo_pos[list(assignedsub_parts[halo])]
        subhalo_vels[halo] = halo_vel[list(assignedsub_parts[halo])]
        subhalo_pids[halo] = halo_pids[list(assignedsub_parts[halo])]

    return thisTask, subhalo_poss, subhalo_vels, subhalo_pids


def get_real_sub_halos(thisTask, pids, pos, vel, boxsize, vlinkl_halo_indp, linkl, pmass, vlcoeff, decrement,
                       redshift, G, h, soft, min_vlcoeff):

    # Extract halo data for this halo ID
    s_halo_pids = np.arange(len(pids), dtype=int)
    full_sim_halo_pids = np.array(list(pids))
    full_halo_poss = pos  # Positions *** NOTE: these are shifted below ***
    full_halo_vels = vel  # Velocities *** NOTE: these are shifted below ***
    full_halo_npart = s_halo_pids.size

    newID_iter = -9

    # =============== Compute mean positions and velocities and wrap the halos ===============

    # Define the comparison particle as the maximum position in the current dimension
    max_part_pos = full_halo_poss.max(axis=0)

    # Compute all the halo particle separations from the maximum position
    sep = max_part_pos - full_halo_poss

    # If any separations are greater than 50% the boxsize (i.e. the halo is split over the boundary)
    # bring the particles at the lower boundary together with the particles at the upper boundary
    # (ignores halos where constituent particles aren't separated by at least 50% of the boxsize)
    # *** Note: fails if halo's extent is greater than 50% of the boxsize in any dimension ***
    full_halo_poss[np.where(sep > 0.5 * boxsize)] += boxsize

    # Compute the shifted mean position in the dimension ixyz
    mean_halo_pos = full_halo_poss.mean(axis=0)

    # Centre the halos about the mean in the dimension ixyz
    full_halo_poss -= mean_halo_pos

    # Define the velocity space linking length
    vlinkl = vlcoeff * vlinkl_halo_indp * (1600 / 200)**(1/6) * pmass ** (1 / 3) * full_halo_npart ** (1 / 3)

    # Define the phase space vectors for this halo
    halo_phases = np.concatenate((full_halo_poss, full_halo_vels), axis=1)

    # Query this halo in velocity space to split apart halos which are found to be distinct in velocity space
    result = find_phase_space_halos(halo_phases, linkl, vlinkl)
    phase_part_haloids, phase_assigned_parts = result

    # Find the halos with 10 or more particles by finding the unique IDs in the particle
    # halo ids array and finding those IDs that are assigned to 10 or more particles
    phase_unique, phase_counts = np.unique(phase_part_haloids, return_counts=True)
    unique_phase_haloids = phase_unique[np.where(phase_counts >= 10)]

    # Remove the null -2 value for single particle halos
    unique_phase_haloids = unique_phase_haloids[np.where(unique_phase_haloids >= 0)]

    extra_halo_pids = {}
    extra_halo_poss = {}
    extra_halo_vels = {}
    iter_vlcoeffs = {}
    results = {}

    # Loop over the halos returned from velocity space
    for pID in unique_phase_haloids:

        if len(phase_assigned_parts[pID]) == 0:
            continue

        # Extract halo data for this velocity space defined halo ID
        # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
        phalo_pids = np.array(list(phase_assigned_parts[pID]), dtype=int)
        halo_pids = s_halo_pids[phalo_pids]
        sim_halo_pids = full_sim_halo_pids[phalo_pids]
        halo_poss = pos[halo_pids, :]  # Positions *** NOTE: these are shifted below ***
        halo_vels = vel[halo_pids, :]  # Velocities *** NOTE: these are shifted below ***
        halo_npart = halo_pids.size

        # =============== Compute mean positions and velocities and wrap the halos ===============

        # Define the comparison particle as the maximum position in the current dimension
        max_part_pos = halo_poss.max(axis=0)

        # Compute all the halo particle separations from the maximum position
        sep = max_part_pos - halo_poss

        # If any separations are greater than 50% the boxsize (i.e. the halo is split over the boundary)
        # bring the particles at the lower boundary together with the particles at the upper boundary
        # (ignores halos where constituent particles aren't separated by at least 50% of the boxsize)
        # *** Note: fails if halo's extent is greater than 50% of the boxsize in any dimension ***
        halo_poss[np.where(sep > 0.5 * boxsize)] += boxsize

        # Compute the shifted mean position
        mean_halo_pos = halo_poss.mean(axis=0)

        # Centre the halos about the mean in the dimension ixyz
        halo_poss -= mean_halo_pos

        # Compute halo's energy
        halo_energy, KE, GE = halo_energy_calc(halo_poss, halo_vels, halo_npart, pmass, redshift, G, h, soft)

        new_vlcoeff = vlcoeff

        iter_halo_pids = halo_pids
        iter_sim_halo_pids = sim_halo_pids
        iter_halo_poss = halo_poss
        iter_halo_vels = halo_vels
        itercount = 0

        while KE / GE >= 1 and halo_npart >= 10 and new_vlcoeff >= 0.8:

            new_vlcoeff -= decrement

            # Define the phase space linking length
            vlinkl = new_vlcoeff * vlinkl_halo_indp * pmass ** (1 / 3) * halo_npart ** (1 / 3)

            # Define the phase space vectors for this halo
            halo_phases = np.concatenate((iter_halo_poss, iter_halo_vels), axis=1)

            # Query this halo in phase space to split apart halos which are found to
            # be distinct in phase space
            result = find_phase_space_halos(halo_phases, linkl, vlinkl)
            iter_part_haloids, iter_assigned_parts = result

            # Find the halos with 10 or more particles by finding the unique IDs in the particle
            # halo ids array and finding those IDs that are assigned to 10 or more particles
            iter_unique, iter_counts = np.unique(iter_part_haloids, return_counts=True)
            unique_iter_haloids = iter_unique[np.where(iter_counts >= 10)]
            iter_counts = iter_counts[np.where(iter_counts >= 10)]

            # Remove the null -2 value for single particle halos
            iter_counts = iter_counts[np.where(unique_iter_haloids >= 0)]
            unique_iter_haloids = unique_iter_haloids[np.where(unique_iter_haloids >= 0)]

            if unique_iter_haloids.size != 0:

                # Sort IDs by count
                unique_iter_haloids = unique_iter_haloids[np.argsort(iter_counts)]

                iterID = unique_iter_haloids[0]
                for iID in unique_iter_haloids[1:]:

                    # Store halo for testing as another task
                    extra_pids = np.array(list(iter_assigned_parts[iID]), dtype=int)
                    extra_halo_pids[newID_iter] = iter_sim_halo_pids[extra_pids]
                    extra_halo_poss[newID_iter] = iter_halo_poss[extra_pids, :]
                    extra_halo_vels[newID_iter] = iter_halo_vels[extra_pids, :]
                    iter_vlcoeffs[newID_iter] = new_vlcoeff - decrement
                    newID_iter -= 100

                # Extract halo data for this phase space defined halo ID
                # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
                phalo_pids = np.array(list(iter_assigned_parts[iterID]), dtype=int)
                iter_halo_pids = iter_halo_pids[phalo_pids]
                iter_sim_halo_pids = iter_sim_halo_pids[phalo_pids]
                iter_halo_poss = iter_halo_poss[phalo_pids, :]  # Positions *** NOTE: these are shifted below ***
                iter_halo_vels = iter_halo_vels[phalo_pids, :]  # Velocities *** NOTE: these are shifted below ***
                halo_npart = iter_halo_pids.size

                # Define the comparison particle as the maximum position in the current dimension
                max_part_pos = iter_halo_poss.max(axis=0)

                # Compute all the halo particle separations from the maximum position
                sep = max_part_pos - iter_halo_poss

                # If any separations are greater than 50% the boxsize (i.e. the halo is split over the boundary)
                # bring the particles at the lower boundary together with the particles at the upper boundary
                # (ignores halos where constituent particles aren't separated by at least 50% of the boxsize)
                # *** Note: fails if halo's extent is greater than 50% of the boxsize in any dimension ***
                iter_halo_poss[np.where(sep > 0.5 * boxsize)] += boxsize

                # Compute halo's energy
                halo_energy, KE, GE = halo_energy_calc(iter_halo_poss, iter_halo_vels, halo_npart,
                                                       pmass, redshift, G, h, soft)
                itercount += 1

            else:
                halo_npart = 0

        if halo_npart >= 10:

            # Extract halo data for this phase space defined halo ID
            # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
            sim_halo_pids = iter_sim_halo_pids
            halo_poss = iter_halo_poss  # Positions *** NOTE: these are shifted below ***
            halo_vels = iter_halo_vels  # Velocities *** NOTE: these are shifted below ***
            halo_npart = sim_halo_pids.size

            # =============== Compute mean positions and velocities and wrap the halos ===============

            # Compute the shifted mean position in the dimension ixyz
            mean_halo_pos = halo_poss.mean(axis=0)

            # Centre the halos about the mean in the dimension ixyz
            halo_poss -= mean_halo_pos

            # May need to wrap if the halo extends over the upper edge of the box
            mean_halo_pos = mean_halo_pos % boxsize

            # Compute the mean velocity in the dimension ixyz
            mean_halo_vel = halo_vels.mean(axis=0)

            if KE / GE <= 1 and vlcoeff >= min_vlcoeff:

                # Define realness flag
                real = True

                results[pID] = {'pids': sim_halo_pids, 'pos': halo_poss, 'vel': halo_vels,
                                'npart': halo_npart, 'real': real, 'mean_halo_pos': mean_halo_pos,
                                'mean_halo_vel': mean_halo_vel, 'halo_energy': halo_energy, 'KE': KE, 'GE': GE}

            else:

                # Define realness flag
                real = False

                results[pID] = {'pids': sim_halo_pids, 'pos': halo_poss, 'vel': halo_vels,
                                'npart': halo_npart, 'real': real, 'mean_halo_pos': mean_halo_pos,
                                'mean_halo_vel': mean_halo_vel, 'halo_energy': halo_energy, 'KE': KE, 'GE': GE}

    return thisTask, results, extra_halo_pids, extra_halo_poss, extra_halo_vels, iter_vlcoeffs


def hosthalofinder(snapshot, llcoeff, sub_llcoeff, inputpath,
                   batchsize, savepath, ini_vlcoeff, min_vlcoeff, decrement, verbose, internal_input, findsubs, ncells):
    """ Run the halo finder, sort the output results, find subhalos and save to a HDF5 file.
        NOTE: MPI task allocation adapted with thanks from:
              https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py
    :param snapshot: The snapshot ID.
    :param llcoeff: The host halo linking length coefficient.
    :param sub_llcoeff: The subhalo linking length coefficient.
    :param gadgetpath: The filepath to the gadget simulation data.
    :param batchsize: The number of particle to be queried at one time.
    :param debug_npart: The number of particles to run the program on when debugging.
    :return: None
    """

    # Define MPI message tags
    tags = utilities.enum('READY', 'DONE', 'EXIT', 'START')

    # =============== Domain Decomposition ===============
    if verbose:
        print("This rank:", rank)

    if rank == 0:

        # Open hdf5 file
        hdf = h5py.File(inputpath + "mega_inputs_" + snapshot + ".hdf5", 'r')

        # Get parameters for decomposition
        mean_sep = hdf.attrs['mean_sep']
        boxsize = hdf.attrs['boxsize']
        npart = hdf.attrs['npart']
        redshift = hdf.attrs['redshift']
        pmass = hdf.attrs['pmass']
        h = hdf.attrs['h']
        pos = hdf['part_pos'][...]

        hdf.close()

        start_dd = time.time()

        # Build the kd tree with the boxsize argument providing 'wrapping' due to periodic boundaries
        # *** Note: Contrary to cKDTree documentation compact_nodes=False and balanced_tree=False results in
        # faster queries (documentation recommends compact_nodes=True and balanced_tree=True)***
        tree = cKDTree(pos, leafsize=16, compact_nodes=False, balanced_tree=False, boxsize=[boxsize, boxsize, boxsize])

        nodes = utilities.decomp_nodes(npart, ncells)

        if verbose:
            print("Domain Decomposition and tree building:", time.time() - start_dd)

            print("Nodes memory size", sys.getsizeof(nodes), "bytes")
            print("Tree memory size", sys.getsizeof(tree), "bytes")

            print(hp.heap())

        # ========================== Compute parameters for candidate halo testing ==========================

        # Compute the linking length for host halos
        linkl = llcoeff * mean_sep

        # Compute the softening length
        soft = 0.05 * boxsize / npart ** (1. / 3.)

        # Define the gravitational constant
        G = (const.G.to(u.km ** 3 * u.M_sun ** -1 * u.s ** -2)).value

        # Define and convert particle mass to M_sun
        pmass *= 1e10 * 1 / h

        # Compute the linking length for subhalos
        sub_linkl = sub_llcoeff * mean_sep

        # Compute the mean density
        mean_den = npart * pmass * u.M_sun / boxsize ** 3 / u.Mpc ** 3 * (1 + redshift) ** 3
        mean_den = mean_den.to(u.M_sun / u.km ** 3)

        # Define the velocity space linking length
        vlinkl_indp = (np.sqrt(G / 2) * (4 * np.pi * 200 * mean_den / 3) ** (1 / 6) * (1 + redshift) ** 0.5).value

    else:

        # Open hdf5 file
        hdf = h5py.File(inputpath + "mega_inputs_" + snapshot + ".hdf5", 'r')

        # Get parameters for decomposition
        boxsize = hdf.attrs['boxsize']
        npart = hdf.attrs['npart']
        redshift = hdf.attrs['redshift']
        pmass = hdf.attrs['pmass']
        h = hdf.attrs['h']
        pos = hdf['part_pos'][...]

        hdf.close()

        # Define and convert particle mass to M_sun
        pmass *= 1e10 * 1 / h

        tree = None
        nodes = None
        linkl = None
        soft = None
        G = None
        sub_linkl = None
        mean_den = None
        vlinkl_indp = None

    nodes, tree, linkl, soft, G, sub_linkl, mean_den, vlinkl_halo_indp = comm.bcast((nodes, tree, linkl, soft, G,
                                                                                     sub_linkl, mean_den, vlinkl_indp),
                                                                                    root=0)

    nnodes = len(nodes)

    if verbose:
        print("There are", len(nodes), "nodes")
        print("This Rank:", rank)
        print(hp.heap())

    if rank == 0:

        # Initialise dictionaries to store halo data
        results = {}

        start = time.time()

        # Master process executes code below
        tasks = set(range(nnodes))
        num_workers = size - 1
        closed_workers = 0
        while closed_workers < num_workers:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == tags.READY:

                # Worker is ready, so send it a task
                if len(tasks) != 0:

                    # Get this task
                    thisTask = tasks.pop()

                    comm.send(thisTask, dest=source, tag=tags.START)

                else:

                    # There are no tasks left so terminate this process
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:
                result = data

            elif tag == tags.EXIT:

                closed_workers += 1

        print("Spatial search finished", time.time() - start)

    else:

        results = {}

        # Worker processes execute code below
        name = MPI.Get_processor_name()
        # print("I am a worker with rank %d on %s." % (rank, name))
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            thisTask = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:

                result = spatial_node_task(thisTask, pos[nodes[thisTask], :], tree, linkl, npart)
                results[thisTask] = result
                comm.send(None, dest=0, tag=tags.DONE)

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)

    if verbose:
        print("This Rank:", rank)
        print(hp.heap())

    # Collect child process results
    collect_start = time.time()
    collected_results = comm.gather(results, root=0)

    if rank == 0:

        # Initialise particle halo id array for full simulation for spatial halos and phase space halos
        spatial_part_haloids = np.full((npart, 2), -2, dtype=int)
        phase_part_haloids = np.full((npart, 2), -2, dtype=int)

        # Combine collected results from children processes into a single dict
        results = {k: v for d in collected_results for k, v in d.items()}

        if verbose:
            print("Collecting the results took", time.time() - collect_start, "seconds")

        halo_pids = {}
        vlcoeffs = {}

        combine_start = time.time()

        # Store halo ids and halo data for the halos found out in the spatial search
        newtaskID = 0
        for task in results:
            for halo in results[task]:

                parts = results[task][halo]
                uni_part_haloids = np.unique(spatial_part_haloids[parts, 0])
                uni_part_haloids = uni_part_haloids[uni_part_haloids >= 0]

                if len(uni_part_haloids) == 0:

                    # Assign new halo to
                    spatial_part_haloids[parts, 0] = newtaskID
                    halo_pids[(1, newtaskID)] = set(parts)

                    newtaskID += 1

                elif len(uni_part_haloids) == 1:

                    spatial_part_haloids[parts, 0] = uni_part_haloids[0]
                    halo_pids[(1, uni_part_haloids[np.where(uni_part_haloids != -2)].min())].update(parts)

                else:

                    existing_halos = uni_part_haloids
                    final_id = existing_halos.min()
                    other_parts = set()
                    for halo in existing_halos:
                        other_parts.update(halo_pids.pop((1, halo)))
                    other_parts.update(parts)
                    spatial_part_haloids[list(other_parts), 0] = uni_part_haloids[0]
                    halo_pids[(1, final_id)] = other_parts

        combined_data = utilities.combine_across_boundaries(snapshot, spatial_part_haloids, ini_vlcoeff,
                                                            vlcoeffs, halo_pids, boxsize, inputpath)
        halo_poss, halo_vels, halo_pids, vlcoeffs, unique_haloids = combined_data

        if verbose:
            print("Combining the results took", time.time() - combine_start, "seconds")

        # Define dictionaries to collate results
        results_dict = {}
        sub_results_dict = {}

        # Initialise dictionaries to store halo data
        subhalo_poss = {}
        subhalo_vels = {}
        subhalo_pids = {}
        sub_vlcoeffs = {}

        # Master process executes code below
        halo_tasks = set(zip(np.full(len(unique_haloids), 1), unique_haloids))
        newSpatialSubID = 0
        newPhaseID = 0
        newPhaseSubID = 0
        num_workers = size - 1
        closed_workers = 0
        while closed_workers < num_workers:
            halo_data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == tags.READY:

                # Worker is ready, so send it a task
                if len(halo_tasks) != 0:

                    halo_task = halo_tasks.pop()

                    if halo_task[0] == 1:

                        data_tosend = (halo_task,
                                       halo_pids.pop(halo_task),
                                       halo_poss.pop(halo_task),
                                       halo_vels.pop(halo_task),
                                       vlcoeffs.pop(halo_task))
                        comm.send(data_tosend, dest=source, tag=tags.START)

                    elif halo_task[0] == 2:

                        data_tosend = (halo_task,
                                       subhalo_pids.pop(halo_task),
                                       subhalo_poss.pop(halo_task),
                                       subhalo_vels.pop(halo_task))
                        comm.send(data_tosend, dest=source, tag=tags.START)

                    elif halo_task[0] == 3:

                        data_tosend = (halo_task,
                                       subhalo_pids.pop(halo_task),
                                       subhalo_poss.pop(halo_task),
                                       subhalo_vels.pop(halo_task),
                                       sub_vlcoeffs.pop(halo_task))
                        comm.send(data_tosend, dest=source, tag=tags.START)

                else:

                    # There are no tasks left so terminate this process
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:

                halo_results = halo_data

                if halo_results[0][0] == 1:

                    extra_halo_pids, extra_halo_poss, extra_halo_vels, extra_vlcoeffs = (halo_results[2],
                                                                                         halo_results[3],
                                                                                         halo_results[4],
                                                                                         halo_results[5])

                    # Create subhalos tasks from the completed halos
                    for res in halo_results[1]:

                        if findsubs:  # Only create sub halo tasks if sub halo flag is true
                            halo_tasks.update({(2, newtaskID)})
                            subhalo_pids[(2, newtaskID)] = halo_results[1][res]['pids']
                            subhalo_poss[(2, newtaskID)] = halo_results[1][res]['pos']
                            subhalo_vels[(2, newtaskID)] = halo_results[1][res]['vel']

                        phase_part_haloids[halo_results[1][res]['pids'], 0] = newPhaseID
                        halo_results[1][res]['id'] = newPhaseID

                        # Increment task ID
                        newtaskID += 1
                        newPhaseID += 1

                    results_dict[halo_results[0]] = halo_results[1]

                    for key in extra_halo_pids:
                        halo_tasks.update({(1, newtaskID)})
                        halo_pids[(1, newtaskID)] = extra_halo_pids[key]
                        halo_poss[(1, newtaskID)] = extra_halo_poss[key]
                        halo_vels[(1, newtaskID)] = extra_halo_vels[key]
                        vlcoeffs[(1, newtaskID)] = extra_vlcoeffs[key]

                        # Increment task ID
                        newtaskID += 1

                elif halo_results[0][0] == 2:

                    thisTask, task_subhalo_poss, task_subhalo_vels, task_subhalo_pids = halo_results

                    # Create subhalos tasks to test subhalos in phase space
                    for halo in task_subhalo_pids:
                        halo_tasks.update({(3, newtaskID)})
                        subhalo_pids[(3, newtaskID)] = task_subhalo_pids[halo]
                        subhalo_poss[(3, newtaskID)] = task_subhalo_poss[halo]
                        subhalo_vels[(3, newtaskID)] = task_subhalo_vels[halo]
                        sub_vlcoeffs[(3, newtaskID)] = ini_vlcoeff

                        spatial_part_haloids[task_subhalo_pids[halo], 1] = newSpatialSubID

                        # Increment task ID
                        newtaskID += 1
                        newSpatialSubID += 1

                elif halo_results[0][0] == 3:

                    extra_halo_pids, extra_halo_poss, extra_halo_vels, extra_vlcoeffs = (halo_results[2],
                                                                                         halo_results[3],
                                                                                         halo_results[4],
                                                                                         halo_results[5])

                    # Create subhalos tasks from the completed halos
                    for res in halo_results[1]:

                        phase_part_haloids[halo_results[1][res]['pids'], 1] = newPhaseSubID
                        halo_results[1][res]['id'] = newPhaseSubID

                        # Increment task ID
                        newtaskID += 1
                        newPhaseSubID += 1

                    sub_results_dict[halo_results[0]] = halo_results[1]

                    for key in extra_halo_pids:
                        halo_tasks.update({(3, newtaskID)})
                        subhalo_pids[(3, newtaskID)] = extra_halo_pids[key]
                        subhalo_poss[(3, newtaskID)] = extra_halo_poss[key]
                        subhalo_vels[(3, newtaskID)] = extra_halo_vels[key]
                        sub_vlcoeffs[(3, newtaskID)] = extra_vlcoeffs[key]

                        # Increment task ID
                        newtaskID += 1

            elif tag == tags.EXIT:

                closed_workers += 1

        if verbose:
            print("Results memory size", sys.getsizeof(results_dict), "bytes")
            print("This Rank:", rank)
            print(hp.heap())

        # Find the halos with 10 or more particles by finding the unique IDs in the particle
        # halo ids array and finding those IDs that are assigned to 10 or more particles
        unique, counts = np.unique(spatial_part_haloids[:, 0], return_counts=True)
        unique_haloids = unique[np.where(counts >= 10)]

        # Remove the null -2 value for single particle halos
        unique_haloids = unique_haloids[np.where(unique_haloids != -2)]

        # Print the number of halos found by the halo finder in >10, >100, >1000, >10000 criteria
        print("=========================== Spatial halos ===========================")
        print(unique_haloids.size, 'halos found with 10 or more particles')
        print(unique[np.where(counts >= 15)].size - 1, 'halos found with 15 or more particles')
        print(unique[np.where(counts >= 20)].size - 1, 'halos found with 20 or more particles')
        print(unique[np.where(counts >= 50)].size - 1, 'halos found with 50 or more particles')
        print(unique[np.where(counts >= 100)].size - 1, 'halos found with 100 or more particles')
        print(unique[np.where(counts >= 500)].size - 1, 'halos found with 500 or more particles')
        print(unique[np.where(counts >= 1000)].size - 1, 'halos found with 1000 or more particles')
        print(unique[np.where(counts >= 10000)].size - 1, 'halos found with 10000 or more particles')

        # Find the halos with 10 or more particles by finding the unique IDs in the particle
        # halo ids array and finding those IDs that are assigned to 10 or more particles
        unique, counts = np.unique(phase_part_haloids[:, 0], return_counts=True)
        unique_haloids = unique[np.where(counts >= 10)]

        # Remove the null -2 value for single particle halos
        unique_haloids = unique_haloids[np.where(unique_haloids != -2)]

        # Print the number of halos found by the halo finder in >10, >100, >1000, >10000 criteria
        print("=========================== Phase halos ===========================")
        print(unique_haloids.size, 'halos found with 10 or more particles')
        print(unique[np.where(counts >= 15)].size - 1, 'halos found with 15 or more particles')
        print(unique[np.where(counts >= 20)].size - 1, 'halos found with 20 or more particles')
        print(unique[np.where(counts >= 50)].size - 1, 'halos found with 50 or more particles')
        print(unique[np.where(counts >= 100)].size - 1, 'halos found with 100 or more particles')
        print(unique[np.where(counts >= 500)].size - 1, 'halos found with 500 or more particles')
        print(unique[np.where(counts >= 1000)].size - 1, 'halos found with 1000 or more particles')
        print(unique[np.where(counts >= 10000)].size - 1, 'halos found with 10000 or more particles')

        # Find the halos with 10 or more particles by finding the unique IDs in the particle
        # halo ids array and finding those IDs that are assigned to 10 or more particles
        unique, counts = np.unique(spatial_part_haloids[:, 1], return_counts=True)
        unique_haloids = unique[np.where(counts >= 10)]

        # Remove the null -2 value for single particle halos
        unique_haloids = unique_haloids[np.where(unique_haloids != -2)]

        # Print the number of halos found by the halo finder in >10, >100, >1000, >10000 criteria
        print("=========================== Spatial subhalos ===========================")
        print(unique_haloids.size, 'halos found with 10 or more particles')
        print(unique[np.where(counts >= 15)].size - 1, 'halos found with 15 or more particles')
        print(unique[np.where(counts >= 20)].size - 1, 'halos found with 20 or more particles')
        print(unique[np.where(counts >= 50)].size - 1, 'halos found with 50 or more particles')
        print(unique[np.where(counts >= 100)].size - 1, 'halos found with 100 or more particles')
        print(unique[np.where(counts >= 500)].size - 1, 'halos found with 500 or more particles')
        print(unique[np.where(counts >= 1000)].size - 1, 'halos found with 1000 or more particles')
        print(unique[np.where(counts >= 10000)].size - 1, 'halos found with 10000 or more particles')

        # Find the halos with 10 or more particles by finding the unique IDs in the particle
        # halo ids array and finding those IDs that are assigned to 10 or more particles
        unique, counts = np.unique(phase_part_haloids[:, 1], return_counts=True)
        unique_haloids = unique[np.where(counts >= 10)]

        # Remove the null -2 value for single particle halos
        unique_haloids = unique_haloids[np.where(unique_haloids != -2)]

        # Print the number of halos found by the halo finder in >10, >100, >1000, >10000 criteria
        print("=========================== Phase subhalos ===========================")
        print(unique_haloids.size, 'halos found with 10 or more particles')
        print(unique[np.where(counts >= 15)].size - 1, 'halos found with 15 or more particles')
        print(unique[np.where(counts >= 20)].size - 1, 'halos found with 20 or more particles')
        print(unique[np.where(counts >= 50)].size - 1, 'halos found with 50 or more particles')
        print(unique[np.where(counts >= 100)].size - 1, 'halos found with 100 or more particles')
        print(unique[np.where(counts >= 500)].size - 1, 'halos found with 500 or more particles')
        print(unique[np.where(counts >= 1000)].size - 1, 'halos found with 1000 or more particles')
        print(unique[np.where(counts >= 10000)].size - 1, 'halos found with 10000 or more particles')

        print(newtaskID - 1, "tasks completed")

        # ============================= Write out data =============================

        # Set up arrays to store host results
        nhalo = newPhaseID
        halo_nparts = np.full(nhalo, -1, dtype=int)
        mean_poss = np.full((nhalo, 3), -1, dtype=float)
        mean_vels = np.full((nhalo, 3), -1, dtype=float)
        reals = np.full(nhalo, 0, dtype=bool)
        halo_energies = np.full(nhalo, -1, dtype=float)
        KEs = np.full(nhalo, -1, dtype=float)
        GEs = np.full(nhalo, -1, dtype=float)
        nsubhalos = np.zeros(nhalo, dtype=float)

        if findsubs:

            # Set up arrays to store host results
            nsubhalo = newPhaseSubID
            subhalo_nparts = np.full(nsubhalo, -1, dtype=int)
            sub_mean_poss = np.full((nsubhalo, 3), -1, dtype=float)
            sub_mean_vels = np.full((nsubhalo, 3), -1, dtype=float)
            sub_reals = np.full(nsubhalo, 0, dtype=bool)
            subhalo_energies = np.full(nsubhalo, -1, dtype=float)
            sub_KEs = np.full(nsubhalo, -1, dtype=float)
            sub_GEs = np.full(nsubhalo, -1, dtype=float)
            host_ids = np.full(nsubhalo, np.nan, dtype=int)

        # # Create the root group
        # snap = h5py.File(savepath + 'halos_' + str(snapshot) + '.hdf5', 'w')
        #
        # # Assign simulation attributes to the root of the z=0 snapshot
        # snap.attrs['snap_nPart'] = npart  # number of particles in the simulation
        # snap.attrs['boxsize'] = boxsize  # box length along each axis
        # snap.attrs['part_mass'] = pmass  # particle mass
        # snap.attrs['h'] = h  # 'little h' (hubble constant parametrisation)
        #
        # # Assign snapshot attributes
        # snap.attrs['linking_length'] = linkl  # host halo linking length
        # # snap.attrs['rhocrit'] = rhocrit  # critical density parameter
        # snap.attrs['redshift'] = redshift
        # # snap.attrs['time'] = t

        halo_ids = np.arange(newPhaseID, dtype=int)

        for res in list(results_dict.keys()):

            results = results_dict.pop(res)

            for halo in list(results.keys()):

                halo_res = results.pop(halo)
                halo_id = halo_res['id']
                halo_pids = halo_res['pids']

                mean_poss[halo_id, :] = halo_res['mean_halo_pos']
                mean_vels[halo_id, :] = halo_res['mean_halo_vel']
                halo_nparts[halo_id] = halo_res['npart']
                reals[halo_id] = halo_res['real']
                halo_energies[halo_id] = halo_res['halo_energy']
                KEs[halo_id] = halo_res['KE']
                GEs[halo_id] = halo_res['GE']

                # # Create datasets in the current halo's group in the HDF5 file
                # halo = snap.create_group(str(halo_id))  # create halo group
                # halo.create_dataset('Halo_Part_IDs', shape=halo_pids.shape, dtype=int,
                #                     data=halo_pids)  # halo particle ids

        # # Save halo property arrays
        # snap.create_dataset('halo_IDs', shape=halo_ids.shape, dtype=int, data=halo_ids, compression='gzip')
        # snap.create_dataset('mean_positions', shape=mean_poss.shape, dtype=float, data=mean_poss, compression='gzip')
        # snap.create_dataset('mean_velocities', shape=mean_vels.shape, dtype=float, data=mean_vels, compression='gzip')
        # snap.create_dataset('nparts', shape=halo_nparts.shape, dtype=int, data=halo_nparts, compression='gzip')
        # snap.create_dataset('real_flag', shape=reals.shape, dtype=bool, data=reals, compression='gzip')
        # snap.create_dataset('halo_total_energies', shape=halo_energies.shape, dtype=float, data=halo_energies,
        #                     compression='gzip')
        # snap.create_dataset('halo_kinetic_energies', shape=KEs.shape, dtype=float, data=KEs, compression='gzip')
        # snap.create_dataset('halo_gravitational_energies', shape=GEs.shape, dtype=float, data=GEs, compression='gzip')
        #
        # # Assign the full halo IDs array to the snapshot group
        # snap.create_dataset('particle_halo_IDs', shape=phase_part_haloids.shape, dtype=int, data=phase_part_haloids,
        #                     compression='gzip')

        if findsubs:

            subhalo_ids = np.arange(newPhaseSubID, dtype=int)

            # # Create subhalo group
            # sub_root = snap.create_group('Subhalos')

            for res in list(sub_results_dict.keys()):

                results = sub_results_dict.pop(res)

                for subhalo in list(results.keys()):

                    subhalo_res = results.pop(subhalo)
                    subhalo_id = subhalo_res['id']
                    subhalo_pids = subhalo_res['pids']
                    host = np.unique(phase_part_haloids[subhalo_pids, 0])

                    assert len(host) == 1, "subhalo is contained in multiple hosts, this should not be possible"

                    sub_mean_poss[subhalo_id, :] = subhalo_res['mean_halo_pos']
                    sub_mean_vels[subhalo_id, :] = subhalo_res['mean_halo_vel']
                    subhalo_nparts[subhalo_id] = subhalo_res['npart']
                    sub_reals[subhalo_id] = subhalo_res['real']
                    subhalo_energies[subhalo_id] = subhalo_res['halo_energy']
                    sub_KEs[subhalo_id] = subhalo_res['KE']
                    sub_GEs[subhalo_id] = subhalo_res['GE']
                    host_ids[subhalo_id] = host
                    nsubhalos[host] += 1

                    # # Create datasets in the current halo's group in the HDF5 file
                    # subhalo = sub_root.create_group(str(subhalo_id))  # create halo group
                    # subhalo.create_dataset('Halo_Part_IDs', shape=subhalo_pids.shape, dtype=int,
                    #                        data=subhalo_pids)  # halo particle ids

        #     # Save halo property arrays
        #     sub_root.create_dataset('subhalo_IDs', shape=subhalo_ids.shape, dtype=int, data=subhalo_ids,
        #                             compression='gzip')
        #     sub_root.create_dataset('host_IDs', shape=host_ids.shape, dtype=int, data=host_ids, compression='gzip')
        #     sub_root.create_dataset('mean_positions', shape=sub_mean_poss.shape, dtype=float, data=sub_mean_poss,
        #                         compression='gzip')
        #     sub_root.create_dataset('mean_velocities', shape=sub_mean_vels.shape, dtype=float, data=sub_mean_vels,
        #                         compression='gzip')
        #     sub_root.create_dataset('nparts', shape=halo_nparts.shape, dtype=int, data=halo_nparts, compression='gzip')
        #     sub_root.create_dataset('real_flag', shape=sub_reals.shape, dtype=bool, data=sub_reals, compression='gzip')
        #     sub_root.create_dataset('halo_total_energies', shape=subhalo_energies.shape, dtype=float,
        #                             data=subhalo_energies, compression='gzip')
        #     sub_root.create_dataset('halo_kinetic_energies', shape=sub_KEs.shape, dtype=float, data=sub_KEs,
        #                             compression='gzip')
        #     sub_root.create_dataset('halo_gravitational_energies', shape=sub_GEs.shape, dtype=float, data=sub_GEs,
        #                             compression='gzip')
        #
        # snap.close()

        assert -1 not in np.unique(KEs), "halo ids are not sequential!"

    else:

        results_dict = None
        spatial_part_haloids = None
        newtaskID = None
        phase_part_haloids = None

        # Worker processes execute code below
        name = MPI.Get_processor_name()
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            thisTask = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:

                if thisTask[0][0] == 1:

                    # Do the work here
                    result = get_real_host_halos(thisTask[0], thisTask[1], thisTask[2], thisTask[3], boxsize,
                                                 vlinkl_halo_indp, linkl, pmass, thisTask[4], decrement,
                                                 redshift, G, h, soft, min_vlcoeff)
                    comm.send(result, dest=0, tag=tags.DONE)

                elif thisTask[0][0] == 2:

                    # Do the work here
                    result = get_sub_halos(thisTask[0], thisTask[1], thisTask[2], thisTask[3], sub_linkl)
                    comm.send(result, dest=0, tag=tags.DONE)

                elif thisTask[0][0] == 3:

                    result = get_real_sub_halos(thisTask[0], thisTask[1], thisTask[2], thisTask[3], boxsize,
                                                vlinkl_halo_indp, sub_linkl, pmass, thisTask[4], decrement,
                                                redshift, G, h, soft, min_vlcoeff)
                    comm.send(result, dest=0, tag=tags.DONE)

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)
