from scipy.spatial import cKDTree
from collections import defaultdict
import pickle
import numpy as np
from guppy import hpy; hp = hpy()
import multiprocessing as mp
import astropy.constants as const
import astropy.units as u
import time
import h5py
import numba as nb
import random
import sys
import os
import utilities


def find_halos(pos, npart, boxsize, batchsize, linkl):
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

    # Build the kd tree with the boxsize argument providing 'wrapping' due to periodic boundaries
    # *** Note: Contrary to CKDTree documentation compact_nodes=False and balanced_tree=False results in
    # faster queries (documentation recommends compact_nodes=True and balanced_tree=True)***
    tree = cKDTree(pos, leafsize=16, compact_nodes=False, balanced_tree=False, boxsize=[boxsize, boxsize, boxsize])
    print(pos.shape, sys.getsizeof(tree))
    print(hp.heap())
    # Assign the query object to a variable to save time on repeated calls
    query_func = tree.query_ball_point

    # =============== Assign Particles To Initial Halos ===============

    assert batchsize < npart / 2, "batchsize must be less than half the total number of particles"

    # Define an array of limits for looping defined by the batchsize
    limits = np.linspace(0, npart, int(npart/batchsize), dtype=np.int32)

    # Loop over particle batches
    for ind, limit in enumerate(limits[:-1]):

        # Print progress
        print('Processed: {x}/{y}'.format(x=limit, y=npart))

        # Remove already assigned particles from the query particles to save time avoiding double checks
        query_parts_poss = pos[limit:limits[ind + 1]]

        # Query the tree in batches of 1000000 for speed returning a list of lists
        query = query_func(query_parts_poss, r=linkl, n_jobs=-1)

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
            assigned_parts.pop(halo_id)

    print('Assignment Complete')

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
    halo_tree = cKDTree(halo_phases, leafsize=10, compact_nodes=False, balanced_tree=False)

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


def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] <= r
    return A[mask]


def kinetic(halo_vels, halo_npart, redshift, pmass):

    # Compute kinetic energy of the halo
    vel_disp = np.zeros(3, dtype=np.float32)
    for ixyz in [0, 1, 2]:
        vel_disp[ixyz] = np.var(halo_vels[:, ixyz])
    KE = 0.5 * halo_npart * pmass * np.sum(vel_disp) * 1 / (1 + redshift)

    return KE


def grav(rij_2, soft, pmass, redshift, h, G):

    # Compute the sum of the gravitational energy of each particle from
    # GE = G*Sum_i(m_i*Sum_{j<i}(m_j/sqrt(r_{ij}**2+s**2)))
    invsqu_dist = 1 / np.sqrt(rij_2 + soft ** 2)
    GE = G * pmass ** 2 * np.sum(invsqu_dist)

    # Convert GE to be in the same units as KE (M_sun km^2 s^-2)
    GE = GE * h * (1 + redshift) * 1 / 3.086e+19

    return GE


def get_seps_lm(halo_poss, halo_npart):

    # Compute the separations of all halo particles along each dimension
    seps = np.zeros((halo_npart, halo_npart, 3), dtype=np.float32)
    for ixyz in [0, 1, 2]:
        rows, cols = np.atleast_2d(halo_poss[:, ixyz], halo_poss[:, ixyz])
        seps[:, :, ixyz] = rows - cols.T

    # Compute the separation between all particles
    # NOTE: this is a symmetric matrix where we only need the upper right half
    rij2 = np.sum(seps * seps, axis=-1)

    return rij2


def get_grav_hm(halo_poss, halo_npart, soft, pmass, redshift, h, G):

    GE = 0

    for i in range(1, halo_npart):
        sep = (halo_poss[:i, :] - halo_poss[i, :])
        rij2 = np.sum(sep * sep, axis=-1)
        invsqu_dist = np.sum(1 / np.sqrt(rij2 + soft ** 2))

        GE += G * pmass ** 2 * invsqu_dist

    # Convert GE to be in the same units as KE (M_sun km^2 s^-2)
    GE = GE * h * (1 + redshift) * 1 / 3.086e+19

    return GE


def halo_energy_calc_exact(halo_poss, halo_vels, halo_npart, pmass, redshift, G, h, soft):

    # Compute kinetic energy of the halo
    KE = kinetic(halo_vels, halo_npart, redshift, pmass)

    if halo_npart < 10000:

        rij2 = get_seps_lm(halo_poss, halo_npart)

        # Extract only the upper triangle of rij
        rij_2 = upper_tri_masking(rij2)

        # Compute gravitational potential energy
        GE = grav(rij_2, soft, pmass, redshift, h, G)

    else:

        GE = get_grav_hm(halo_poss, halo_npart, soft, pmass, redshift, h, G)

    # Compute halo's energy
    halo_energy = KE - GE

    return halo_energy, KE, GE


@nb.jit(nogil=True, parallel=True)
def wrap_halo(halo_poss, boxsize):

    # Define the comparison particle as the maximum position in the current dimension
    max_part_pos = halo_poss.max(axis=0)

    # Compute all the halo particle separations from the maximum position
    sep = max_part_pos - halo_poss

    # If any separations are greater than 50% the boxsize (i.e. the halo is split over the boundary)
    # bring the particles at the lower boundary together with the particles at the upper boundary
    # (ignores halos where constituent particles aren't separated by at least 50% of the boxsize)
    # *** Note: fails if halo's extent is greater than 50% of the boxsize in any dimension ***
    halo_poss[np.where(sep > 0.5 * boxsize)] += boxsize

    # Compute the shifted mean position in the dimension ixyz
    mean_halo_pos = halo_poss.mean(axis=0)

    # Centre the halos about the mean in the dimension ixyz
    halo_poss -= mean_halo_pos

    return halo_poss, mean_halo_pos


def halo_energy_calc_approx(halo_poss, halo_vels, halo_npart, pmass, redshift, G, h, soft):

    # Compute kinetic energy of the halo
    vel_disp = np.var(halo_vels, axis=0)
    KE = 0.5 * halo_npart * pmass * np.sum(vel_disp) * 1 / (1 + redshift)

    halo_radii = np.sqrt(halo_poss[:, 0]**2 + halo_poss[:, 1]**2 + halo_poss[:, 2]**2)

    srtd_halo_radii = np.sort(halo_radii)

    n_within_radii = np.arange(0, halo_radii.size)
    GE = np.sum(G * pmass**2 * n_within_radii / srtd_halo_radii)

    # Compute halo's energy
    halo_energy = KE - GE * h * (1 + redshift) * 1 / 3.086e+19

    return halo_energy


halo_energy_calc = halo_energy_calc_exact


def get_real_host_halos(thisTask, pids, pos, vel, boxsize, vlinkl_halo_indp, linkl, pmass, vlcoeff, decrement,
                        redshift, G, h, soft):

    # Extract halo data for this halo ID
    s_sim_halo_pids = np.array(list(pids))  # particle ID NOTE: Overwrites IDs which started at 1
    full_halo_poss = pos  # Positions *** NOTE: these are shifted below ***
    full_halo_vels = vel  # Velocities *** NOTE: these are shifted below ***
    full_halo_npart = s_sim_halo_pids.size

    newID_iter = -9

    # =============== Compute mean positions and velocities and wrap the halos ===============

    # Compute the shifted mean position in the dimension ixyz
    mean_halo_pos = full_halo_poss.mean(axis=0)

    # Centre the halos about the mean in the dimension ixyz
    full_halo_poss -= mean_halo_pos

    # Define the velocity space linking length
    vlinkl = vlcoeff * vlinkl_halo_indp * pmass**(1/3) * full_halo_npart**(1/3)

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

        if len(phase_assigned_parts[pID]) < 10:
            continue

        # Extract halo data for this velocity space defined halo ID
        # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
        phalo_pids = np.array(list(phase_assigned_parts[pID]), dtype=int)
        sim_halo_pids = s_sim_halo_pids[phalo_pids]
        halo_poss = full_halo_poss[phalo_pids, :]  # Positions *** NOTE: these are shifted below ***
        halo_vels = full_halo_vels[phalo_pids, :]  # Velocities *** NOTE: these are shifted below ***
        halo_npart = phalo_pids.size

        # =============== Compute mean positions and velocities and wrap the halos ===============

        # Compute the shifted mean position
        mean_halo_pos = halo_poss.mean(axis=0)

        # Centre the halos about the mean in the dimension ixyz
        halo_poss -= mean_halo_pos

        # Compute the mean velocity
        mean_halo_vel = halo_vels.mean(axis=0)

        # Compute halo's energy
        halo_energy, KE, GE = halo_energy_calc(halo_poss, halo_vels, halo_npart, pmass, redshift, G, h, soft)

        if KE / GE <= 1 and vlcoeff >= 0.8:

            # Define realness flag
            real = True

            # Wrap halo if it's outside the box
            mean_halo_pos %= boxsize

            results[pID] = {'pids': sim_halo_pids, 'npart': halo_npart, 'real': real,
                            'mean_halo_pos': mean_halo_pos, 'mean_halo_vel': mean_halo_vel,
                            'halo_energy': halo_energy, 'KE': KE, 'GE': GE}

        elif KE / GE >= 1 and vlcoeff > 0.8:

            # Store halo for testing as another task
            extra_halo_pids[newID_iter] = sim_halo_pids
            extra_halo_poss[newID_iter] = halo_poss
            extra_halo_vels[newID_iter] = halo_vels
            iter_vlcoeffs[newID_iter] = vlcoeff - decrement
            newID_iter -= 100

        elif KE / GE > 1 and vlcoeff < 0.8:

            # Define realness flag
            real = False

            # Wrap halo if it's outside the box
            mean_halo_pos %= boxsize

            results[pID] = {'pids': sim_halo_pids, 'npart': halo_npart, 'real': real,
                            'mean_halo_pos': mean_halo_pos, 'mean_halo_vel': mean_halo_vel,
                            'halo_energy': halo_energy, 'KE': KE, 'GE': GE}

    return thisTask, results, extra_halo_pids, extra_halo_poss, extra_halo_vels, iter_vlcoeffs


def hosthalofinder(snapshot, llcoeff, sub_llcoeff, inputpath,
                   batchsize, savepath, ini_vlcoeff, min_vlcoeff, decrement, verbose, internal_input, findsubs):
    """ Run the halo finder, sort the output results, find subhalos and save to a HDF5 file.

    :param snapshot: The snapshot ID.
    :param llcoeff: The host halo linking length coefficient.
    :param sub_llcoeff: The subhalo linking length coefficient.
    :param gadgetpath: The filepath to the gadget simulation data.
    :param batchsize: The number of particle to be queried at one time.
    :param debug_npart: The number of particles to run the program on when debugging.

    :return: None
    """

    # =============== Load Simulation Data, Compute The Linking Length And Sort Simulation Data ===============

    # Open hdf5 file
    hdf = h5py.File(inputpath + "mega_inputs_" + snapshot + ".hdf5", 'r')

    # Get parameters for decomposition
    mean_sep = hdf.attrs['mean_sep']
    boxsize = hdf.attrs['boxsize']
    npart = hdf.attrs['npart']
    redshift = hdf.attrs['redshift']
    t = hdf.attrs['t']
    rhocrit = hdf.attrs['rhocrit']
    pmass = hdf.attrs['pmass']
    h = hdf.attrs['h']
    pos = hdf['part_pos'][...]
    vel = hdf['part_vel'][...]

    hdf.close()

    # Change variable types to save memory
    pos.astype(np.float64)
    vel.astype(np.float64)

    # Compute the linking length for host halos
    linkl = llcoeff * mean_sep

    # Compute the softening length
    soft = 0.05 * boxsize / npart**(1./3.)

    # Initialise "not real" halo counter
    notreal = 0

    # Define the gravitational constant
    G = (const.G.to(u.km**3 * u.M_sun**-1 * u.s**-2)).value

    # Define and convert particle mass to M_sun
    pmass *= 1e10 * 1 / h

    # Compute the mean separation
    mean_sep = boxsize / npart**(1/3)

    # Compute the linking length for subhalos
    sub_linkl = sub_llcoeff * mean_sep

    # Compute the mean density
    mean_den = npart * pmass * u.M_sun / boxsize ** 3 / u.Mpc ** 3 * (1 + redshift) ** 3
    mean_den = mean_den.to(u.M_sun / u.km**3)

    # Define the velocity space linking length
    vlinkl_halo_indp = (np.sqrt(G / 2) * (4 * np.pi * 200 * mean_den / 3) ** (1 / 6)
                        * (1 + redshift) ** 0.5).value

    # =============== Run The Halo Finder And Reduce The Output ===============

    # Run the halo finder for this snapshot at the host linking length and
    # assign the results to the relevant variables
    assin_start = time.time()
    part_haloids, assigned_parts = find_halos(pos, npart, boxsize, batchsize, linkl)
    print(snapshot, 'Initial assignment: ', time.time()-assin_start)

    # Find the halos with 10 or more particles by finding the unique IDs in the particle
    # halo ids array and finding those IDs that are assigned to 10 or more particles
    unique, counts = np.unique(part_haloids, return_counts=True)
    unique_haloids = unique[np.where(counts >= 10)]

    # Remove the null -2 value for single particle halos
    unique_haloids = unique_haloids[np.where(unique_haloids != -2)]

    # Define pre velocity space number of halos
    pre_mt10 = unique_haloids.size
    pre_mt20 = unique[np.where(counts >= 20)].size - 1
    pre_mt100 = unique[np.where(counts >= 100)].size - 1
    pre_mt1000 = unique[np.where(counts >= 1000)].size - 1
    pre_mt10000 = unique[np.where(counts >= 10000)].size - 1

    # Print the number of halos found by the halo finder in >10, >100, >1000, >10000 criteria
    print(unique_haloids.size, 'halos found with 10 or more particles')
    print(unique[np.where(counts >= 15)].size - 1, 'halos found with 15 or more particles')
    print(unique[np.where(counts >= 20)].size - 1, 'halos found with 20 or more particles')
    print(unique[np.where(counts >= 50)].size - 1, 'halos found with 50 or more particles')
    print(unique[np.where(counts >= 100)].size - 1, 'halos found with 100 or more particles')
    print(unique[np.where(counts >= 500)].size - 1, 'halos found with 500 or more particles')
    print(unique[np.where(counts >= 1000)].size - 1, 'halos found with 1000 or more particles')
    print(unique[np.where(counts >= 10000)].size - 1, 'halos found with 10000 or more particles')

    # Initialise the counters for the number of subhalos in a mass bin
    plus10subs = 0
    plus20subs = 0
    plus100subs = 0
    plus1000subs = 0
    plus10000subs = 0

    # =============== Assign The Halo Data To A HDF5 File ===============

    # Create the root group
    snap = h5py.File(savepath + 'halos_' + str(snapshot) + '.hdf5', 'w')

    # Assign simulation attributes to the root of the z=0 snapshot
    if snapshot == '061':
        # *** Note: Potentially groups can be moved to a master root group with simulation attributes in the root ****
        snap.attrs['snap_nPart'] = npart  # number of particles in the simulation
        snap.attrs['boxsize'] = boxsize  # box length along each axis
        snap.attrs['partMass'] = pmass  # particle mass
        snap.attrs['h'] = h  # 'little h' (hubble constant parametrisation)

    # Assign snapshot attributes
    snap.attrs['linkingLength'] = linkl  # host halo linking length
    snap.attrs['rhocrit'] = rhocrit  # critical density parameter
    snap.attrs['redshift'] = redshift
    snap.attrs['time'] = t

    snap.close()

    # Overwrite the halo IDs array so that at write out it only contains sequential halo IDs of halos above
    # the 10 particle threshold, any halos below this limit are assigned -2 and hence forth considered
    # single particles
    part_haloids = np.full((npart, 2), -2, dtype=int)

    energy_dict = {}
    energy_dict['before'] = {}
    energy_dict['after'] = {}
    valpha_dict = {}

    unique_haloids = list(unique_haloids)
    iter_vlcoeffs = dict(zip(unique_haloids, np.full(len(unique_haloids), ini_vlcoeff)))
    i = 0

    # Sort, derive and assign each of the halo's data to the relevant group within the HDF5 file
    newID = -1
    sub_newID = -1
    newID_iter = -9
    while i < len(unique_haloids):

        ID = unique_haloids[i]
        # Increment phase halo index pointer
        i += 1
        # print(ID, i)
        # if type(ID) == float:
        #     print(assigned_parts[ID])

        vlcoeff = iter_vlcoeffs[ID]

        assert len(list(assigned_parts[ID])) >= 10, "Halos with less than 10 particles can't exist"

        # Extract halo data for this halo ID
        s_halo_pids = np.array(list(assigned_parts[ID]))  # particle ID NOTE: Overwrites IDs which started at 1
        full_halo_poss = pos[s_halo_pids, :]  # Positions *** NOTE: these are shifted below ***
        full_halo_vels = vel[s_halo_pids, :]  # Velocities *** NOTE: these are shifted below ***
        full_halo_npart = s_halo_pids.size

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

        result_tup = get_real_host_halos(ID, s_halo_pids, full_halo_poss, full_halo_vels, boxsize, vlinkl_halo_indp,
                                         linkl, pmass, iter_vlcoeffs[ID], decrement, redshift, G, h, soft)
        thisTask, results, extra_halo_pids, _, _, this_vlcoeffs = result_tup
        for iID in extra_halo_pids:
            unique_haloids.append(newID_iter)
            assigned_parts[newID_iter] = extra_halo_pids[iID]
            iter_vlcoeffs[newID_iter] = this_vlcoeffs[iID]
            newID_iter -= 100

        for resID in results:

            # Extract halo data for this halo ID
            halo_pids = results[resID]['pids']  # particle ID NOTE: Overwrites IDs which started at 1
            halo_poss = pos[halo_pids, :]  # Positions *** NOTE: these are shifted below ***
            halo_vels = vel[halo_pids, :]  # Velocities *** NOTE: these are shifted below ***
            halo_npart = s_halo_pids.size
            real = results[resID]['real']
            mean_halo_pos = results[resID]['mean_halo_pos']
            mean_halo_vel = results[resID]['mean_halo_vel']
            halo_energy = results[resID]['halo_energy']
            KE = results[resID]['KE']
            GE = results[resID]['GE']

            # ============ If substructure or energy criteria has been satisfied write out the result =============

            # Increment the newID such that the IDs are sequential
            newID += 1

            # Reassign the IDs such that they are sequential for halos with 10 or more particles
            part_haloids[halo_pids, 0] = newID

            # Open root group
            hdf = h5py.File(savepath + 'halos_' + str(snapshot) + '.hdf5', 'r+')

            # Create datasets in the current halo's group in the HDF5 file
            halo = hdf.create_group(str(newID))  # create halo group
            halo.create_dataset('Halo_Part_IDs', shape=[len(halo_pids)], dtype=int,
                                compression='gzip', data=halo_pids)  # halo particle ids
            halo.create_dataset('Halo_Pos', shape=halo_poss.shape, dtype=float,
                                compression='gzip', data=halo_poss)  # halo centered positions
            halo.create_dataset('Halo_Vel', shape=halo_vels.shape, dtype=float,
                                compression='gzip', data=halo_vels)  # halo centered velocities
            halo.create_dataset('mean_pos', shape=mean_halo_pos.shape, dtype=float,
                                compression='gzip', data=mean_halo_pos)  # mean position
            halo.create_dataset('mean_vel', shape=mean_halo_vel.shape, dtype=float,
                                compression='gzip', data=mean_halo_vel)  # mean velocity
            halo.attrs['halo_nPart'] = halo_npart  # number of particles in halo
            halo.attrs['Real'] = real  # realness flag
            # halo.attrs['nSubhalo'] = unique_final_subhaloids.size  # number of subhalos
            halo.attrs['halo_energy'] = halo_energy  # halo energy
            halo.attrs['KE'] = KE  # kinetic energy
            halo.attrs['GE'] = GE  # gravitational binding energy

            hdf.close()

    # Open root group
    hdf = h5py.File(savepath + 'halos_' + str(snapshot) + '.hdf5', 'r+')

    # Assign the full halo IDs array to the snapshot group
    hdf.create_dataset('Halo_IDs', shape=part_haloids.shape, dtype=int, compression='gzip', data=part_haloids)

    hdf.close()

    # Find the halos with 10 or more particles by finding the unique IDs in the particle
    # halo ids array and finding those IDs that are assigned to 10 or more particles
    unique, counts = np.unique(part_haloids[:, 0], return_counts=True)

    unique_haloids = unique[np.where(counts >= 10)]

    # Remove the null -2 value for single particle halos
    unique_haloids = unique_haloids[np.where(unique_haloids != -2)]

    # Print the number of halos found by the halo finder in >10, >100, >1000, >10000 criteria
    print('Post Phase Space')
    print(unique_haloids.size, 'halos found with 10 or more particles')
    print(unique[np.where(counts >= 15)].size - 1, 'halos found with 15 or more particles')
    print(unique[np.where(counts >= 20)].size - 1, 'halos found with 20 or more particles')
    print(unique[np.where(counts >= 50)].size - 1, 'halos found with 50 or more particles')
    print(unique[np.where(counts >= 100)].size - 1, 'halos found with 100 or more particles')
    print(unique[np.where(counts >= 500)].size - 1, 'halos found with 500 or more particles')
    print(unique[np.where(counts >= 1000)].size - 1, 'halos found with 1000 or more particles')
    print(unique[np.where(counts >= 10000)].size - 1, 'halos found with 10000 or more particles')

    # Print out the total number of subhalos in each mass bin
    print(plus10subs, 'subhalos found with 10 or more particles')
    print(plus20subs, 'subhalos found with 20 or more particles')
    print(plus100subs, 'subhalos found with 100 or more particles')
    print(plus1000subs, 'subhalos found with 1000 or more particles')
    print(plus10000subs, 'subhalos found with 10000 or more particles')

    print('Halos found to be not real in snapshot', snapshot, ':', notreal)

    if 'logs' not in os.listdir(os.getcwd()):
        os.mkdir('logs')
    
    f = open('logs/' + snapshot + 'iter_log.txt', 'w')
    f.write('Initial halos\n')
    f.write(str(pre_mt10) + ' halos found with 10 or more particles\n')
    f.write(str(pre_mt20) + ' halos found with 20 or more particles\n')
    f.write(str(pre_mt100) + ' halos found with 100 or more particles\n')
    f.write(str(pre_mt1000) + ' halos found with 1000 or more particles\n')
    f.write(str(pre_mt10000) + ' halos found with 10000 or more particles\n')
    f.write('Post Sub Structure test\n')
    f.write(str(unique_haloids.size) + ' halos found with 10 or more particles\n')
    f.write(str(unique[np.where(counts >= 20)].size - 1) + ' halos found with 20 or more particles\n')
    f.write(str(unique[np.where(counts >= 100)].size - 1) + ' halos found with 100 or more particles\n')
    f.write(str(unique[np.where(counts >= 1000)].size - 1) + ' halos found with 1000 or more particles\n')
    f.write(str(unique[np.where(counts >= 10000)].size - 1) + ' halos found with 10000 or more particles\n')
    f.write('Halos found to be not real: ' + str(notreal))
    f.close()

    # if 'energy_dicts' not in os.listdir(os.getcwd()):
    #     os.mkdir('energy_dicts')
    #
    # with open('energy_dicts/energy_ba_' + snapshot + '.pck', 'wb') as pfile:
    #     pickle.dump(energy_dict, pfile)
    #
    # if 'vel_linkl' not in os.listdir(os.getcwd()):
    #     os.mkdir('vel_linkl')
    #
    # with open('vel_linkl/valpha_' + snapshot + '.pck', 'wb') as pfile:
    #     pickle.dump(valpha_dict, pfile)
