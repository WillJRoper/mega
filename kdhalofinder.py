from scipy.spatial import cKDTree
from collections import defaultdict
import readgadgetdata
import pickle
import numpy as np
import multiprocessing as mp
import astropy.constants as const
import astropy.units as u
import time
import h5py
import numba as nb
import random
import sys
import os


def read_sim(snapshot, PATH, llcoeff):
    """ Reads in gadget-2 simulation data and computes the host halo linking length. (For more information see Docs)

    :param snapshot: The snapshot ID as a string (e.g. '061')
    :param PATH: The filepath to the directory containing the simulation data.
    :param llcoeff: The host halo linking length coefficient.

    :return: pid: An array containing the particle IDs.
             pos: An array of the particle position vectors.
             vel: An array of the particle velocity vectors.
             npart: The number of particles used in the simulation.
             boxsize: The length of the simulation box along a single axis.
             redshift: The redshift of the current snapshot.
             t: The elapsed time of the current snapshot.
             rhocrit: The critical density at the current snapshot.
             pmass: The mass of a dark matter particle.
             h: 'Little h', The hubble parameter parametrisation.
             linkl: The linking length.

    """

    # =============== Load Simulation Data ===============

    # Load snapshot data from gadget-2 file *** Note: will need to be changed for use with other simulations data ***
    snap = readgadgetdata.readsnapshot(snapshot, PATH)
    pid, pos, vel = snap[0:3]  # pid=particle ID, pos=all particle's position, vel=all particle's velocity
    head = snap[3:]  # header values
    npart = head[0]  # number of particles in simulation
    boxsize = head[3]  # simulation box length(/size) along each axis
    redshift = head[1]
    t = head[2]  # elapsed time of the snapshot
    rhocrit = head[4]  # Critical density
    pmass = head[5]  # Particle mass
    h = head[6]  # 'little h' (hubble parameter parametrisation)

    # =============== Compute Linking Length ===============

    # Compute the mean separation
    mean_sep = boxsize / npart**(1./3.)

    # Compute the linking length for host halos
    linkl = llcoeff * mean_sep

    return pid, pos, vel, npart, boxsize, redshift, t, rhocrit, pmass, h, linkl


def find_halos(pos, npart, boxsize, batchsize, linkl, debug_npart):
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

    # Assign the query object to a variable to save time on repeated calls
    query_func = tree.query_ball_point

    # Overwrite npart for debugging purposes if called in arguments
    if debug_npart is not None:
        npart = debug_npart

    # =============== Assign Particles To Initial Halos ===============

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

    if npart > 1000:

        # Define an array of limits for looping defined by the batchsize
        limits = np.linspace(0, npart, int(npart / 1000) + 1, dtype=int)

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

    if npart > 1000:

        # Define an array of limits for looping defined by the batchsize
        limits = np.linspace(0, npart, int(npart / 1000) + 1, dtype=int)

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


@nb.njit(nogil=True, parallel=True)
def kinetic(halo_vels, halo_npart, redshift, pmass):

    # Compute kinetic energy of the halo
    vel_disp = np.zeros(3, dtype=np.float32)
    for ixyz in [0, 1, 2]:
        vel_disp[ixyz] = np.var(halo_vels[:, ixyz])
    KE = 0.5 * halo_npart * pmass * np.sum(vel_disp) * 1 / (1 + redshift)

    return KE


@nb.njit(nogil=True, parallel=True)
def grav(rij_2, soft, pmass, redshift, h, G):

    # Compute the sum of the gravitational energy of each particle from
    # GE = G*Sum_i(m_i*Sum_{j<i}(m_j/sqrt(r_{ij}**2+s**2)))
    invsqu_dist = 1 / np.sqrt(rij_2 + soft ** 2)
    GE = G * pmass ** 2 * np.sum(invsqu_dist)

    # Convert GE to be in the same units as KE (M_sun km^2 s^-2)
    GE = GE * h * (1 + redshift) * 1 / 3.086e+19

    return GE


@nb.njit(nogil=True, parallel=False)
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


@nb.njit(nogil=True, parallel=True)
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


@nb.jit(nogil=True, parallel=True)
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


def hosthalofinder(snapshot, llcoeff=0.2, sub_llcoeff=0.1, gadgetpath='snapshotdata/snapdir_', batchsize=2000000,
                   debug_npart=None, savepath='halo_snapshots', vlcoeff=1.0):
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

    pid, pos, vel, npart, boxsize, redshift, t, rhocrit, pmass, h, linkl = read_sim(snapshot,
                                                                                    PATH=gadgetpath, llcoeff=llcoeff)

    # Change variable types to save memory
    pid.astype(np.int32)
    pos.astype(np.float64)
    vel.astype(np.float64)

    # Sort the simulation data arrays by the particle ID
    sinds = pid.argsort()
    pos = pos[sinds, :]
    vel = vel[sinds, :]

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
    vlinkl_halo_indp = (vlcoeff * np.sqrt(G / 2) * (4 * np.pi * 200 * mean_den / 3) ** (1 / 6)
                        * (1 + redshift) ** 0.5).value

    # =============== Run The Halo Finder And Reduce The Output ===============

    # Run the halo finder for this snapshot at the host linking length and
    # assign the results to the relevant variables
    assin_start = time.time()
    part_haloids, assigned_parts = find_halos(pos, npart, boxsize, batchsize, linkl, debug_npart)
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
    print(unique[np.where(counts >= 20)].size - 1, 'halos found with 20 or more particles')
    print(unique[np.where(counts >= 100)].size - 1, 'halos found with 100 or more particles')
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
    i = 0

    # Sort, derive and assign each of the halo's data to the relevant group within the HDF5 file
    newID = -1
    sub_newID = -1
    newID_iter = 9999999999
    while i < len(unique_haloids):

        ID = unique_haloids[i]
        # Increment phase halo index pointer
        i += 1
        # print(ID, i)
        # if type(ID) == float:
        #     print(assigned_parts[ID])

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

        # Compute the shifted mean position in the dimension ixyz
        mean_halo_pos = full_halo_poss.mean(axis=0)

        # Centre the halos about the mean in the dimension ixyz
        full_halo_poss -= mean_halo_pos

        # Define the velocity space linking length
        vlinkl = vlinkl_halo_indp * pmass**(1/3) * full_halo_npart**(1/3)

        # Define the phase space vectors for this halo
        halo_phases = np.concatenate((full_halo_poss, full_halo_vels), axis=1)

        pq_start = time.time()
        # Query this halo in velocity space to split apart halos which are found to be distinct in velocity space
        result = find_phase_space_halos(halo_phases, linkl, vlinkl)
        phase_part_haloids, phase_assigned_parts = result
        if (time.time() - pq_start) > 10:
            print('phase query: ', newID + 1, full_halo_vels.shape[0], time.time() - pq_start)

        # Find the halos with 10 or more particles by finding the unique IDs in the particle
        # halo ids array and finding those IDs that are assigned to 10 or more particles
        phase_unique, phase_counts = np.unique(phase_part_haloids, return_counts=True)
        unique_phase_haloids = phase_unique[np.where(phase_counts >= 10)]

        # Remove the null -2 value for single particle halos
        unique_phase_haloids = unique_phase_haloids[np.where(unique_phase_haloids >= 0)]

        if unique_phase_haloids.size == 0:
            continue

        # Loop over the halos returned from velocity space
        for pID in unique_phase_haloids:

            if len(phase_assigned_parts[pID]) == 0:
                continue

            # Extract halo data for this velocity space defined halo ID
            # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
            phalo_pids = np.array(list(phase_assigned_parts[pID]), dtype=int)
            halo_pids = s_halo_pids[phalo_pids]
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
            final_vlcoeff = vlcoeff

            iter_halo_pids = halo_pids
            iter_halo_poss = halo_poss
            iter_halo_vels = halo_vels
            itercount = 0

            # iter_stage_log = {}

            if halo_energy > 0:
                energy_dict['after'][str(ID) + '.' + str(pID)] = {}
                energy_dict['before'][str(ID) + '.' + str(pID)] = {}
                energy_dict['before'][str(ID) + '.' + str(pID)]['M'] = halo_npart
                energy_dict['before'][str(ID) + '.' + str(pID)]['E'] = KE / GE

            while KE/GE >= 1 and halo_npart >= 10 and new_vlcoeff > 0.8:

                new_vlcoeff -= 0.05

                print('--')

                final_vlcoeff = new_vlcoeff

                # Define the phase space linking length
                vlinkl = new_vlcoeff / vlcoeff * vlinkl_halo_indp * pmass ** (1 / 3) * halo_npart ** (1 / 3)

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
                        unique_haloids.append(newID_iter)
                        assigned_parts[newID_iter] = iter_halo_pids[np.array(list(iter_assigned_parts[iID]), dtype=int)]
                        newID_iter += 100

                    # Extract halo data for this phase space defined halo ID
                    # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
                    phalo_pids = np.array(list(iter_assigned_parts[iterID]), dtype=int)
                    iter_halo_pids = iter_halo_pids[phalo_pids]
                    iter_halo_poss = pos[iter_halo_pids, :]  # Positions *** NOTE: these are shifted below ***
                    iter_halo_vels = vel[iter_halo_pids, :]  # Velocities *** NOTE: these are shifted below ***
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

                    print(halo_energy)
                    print('--')

                    energy_dict['after'][str(ID) + '.' + str(pID)]['M'] = halo_npart
                    energy_dict['after'][str(ID) + '.' + str(pID)]['E'] = KE / GE

                    itercount += 1

                else:
                    halo_npart = 0

            if halo_npart >= 10:

                if halo_energy > 0:
                    notreal += 1
                    real = False
                else:
                    real = True

                valpha_dict[newID + 1] = final_vlcoeff

                # Extract halo data for this phase space defined halo ID
                # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
                halo_pids = iter_halo_pids
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

                # Compute the shifted mean position in the dimension ixyz
                mean_halo_pos = halo_poss.mean(axis=0)

                # Centre the halos about the mean in the dimension ixyz
                halo_poss -= mean_halo_pos

                # May need to wrap if the halo extends over the upper edge of the box
                mean_halo_pos = mean_halo_pos % boxsize

                # Compute the mean velocity in the dimension ixyz
                mean_halo_vel = halo_vels.mean(axis=0)

                # Compute halo's energy
                halo_energy, KE, GE = halo_energy_calc(halo_poss, halo_vels, halo_npart,
                                                       pmass, redshift, G, h, soft)

                # ============ If substructure or energy criteria has been satisfied write out the result =============

                # Increment the newID such that the IDs are sequential
                newID += 1

                # Reassign the IDs such that they are sequential for halos with 10 or more particles
                part_haloids[halo_pids, 0] = newID

                # Open root group
                snap = h5py.File(savepath + 'halos_' + str(snapshot) + '.hdf5', 'r+')

                # Create datasets in the current halo's group in the HDF5 file
                halo = snap.create_group(str(newID))  # create halo group
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

                snap.close()

                # Query this halo in velocity space to split apart halos which are found to be
                # distinct in velocity space
                result = find_subhalos(halo_poss, sub_linkl)
                part_subhaloids, assigned_subparts = result

                # Find the halos with 10 or more particles by finding the unique IDs in the particle
                # halo ids array and finding those IDs that are assigned to 10 or more particles
                sub_unique, sub_counts = np.unique(part_subhaloids, return_counts=True)
                unique_subhaloids = sub_unique[np.where(sub_counts >= 10)]
                sub_counts = sub_counts[np.where(sub_counts >= 10)]

                # Remove any single particle halos
                unique_subhaloids = list(unique_subhaloids[np.where(unique_subhaloids >= 0)])

                # Reset subhalo id array and dictionary
                final_part_subhaloids = np.full(halo_npart, -2, dtype=int)
                newsubID_iter = 9999999
                si = 0

                while si < len(unique_subhaloids):

                    sID = unique_subhaloids[si]
                    si += 1

                    # Extract halo data for this velocity space defined halo ID
                    # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
                    full_subhalo_pids = np.array(list(assigned_subparts[sID]))
                    subhalo_poss = halo_poss[full_subhalo_pids, :]  # Positions *** NOTE: these are shifted below ***
                    subhalo_vels = halo_vels[full_subhalo_pids, :]  # Velocities *** NOTE: these are shifted below ***
                    subhalo_npart = full_subhalo_pids.size

                    # Define the velocity space linking length
                    sub_vlinkl = vlinkl_halo_indp * (1600 / 200)**(1/6) * (pmass * subhalo_npart) ** (1 / 3)

                    # Define the phase space vectors for this halo
                    subhalo_phases = np.concatenate((subhalo_poss, subhalo_vels), axis=1)

                    pq_start = time.time()
                    # Query this halo in velocity space to split apart halos which are distinct in velocity space
                    result = find_phase_space_halos(subhalo_phases, sub_linkl, sub_vlinkl)
                    phase_part_subhaloids, phase_assigned_subparts = result
                    if (time.time() - pq_start) > 10:
                        print('phase query: ', newID + 1, sub_newID, subhalo_phases.shape[0], time.time() - pq_start)

                    # Find the halos with 10 or more particles by finding the unique IDs in the particle
                    # halo ids array and finding those IDs that are assigned to 10 or more particles
                    phase_subunique, subphase_counts = np.unique(phase_part_subhaloids, return_counts=True)
                    unique_phase_subhaloids = phase_subunique[np.where(subphase_counts >= 10)]
                    subphase_counts = subphase_counts[np.where(subphase_counts >= 10)]

                    # Remove the null -2 value for single particle halos
                    subphase_counts = subphase_counts[np.where(unique_phase_subhaloids >= 0)]
                    unique_phase_subhaloids = unique_phase_subhaloids[np.where(unique_phase_subhaloids >= 0)]

                    for spID in unique_phase_subhaloids:

                        # Extract specific halo data for halo ID
                        # Particle ID, NOTE: Overwrites the C/FORTRAN IDs which start at 1
                        subhalo_pids = full_subhalo_pids[list(phase_assigned_subparts[spID])]
                        subhalo_poss = halo_poss[subhalo_pids, :]  # Positions *** NOTE: These are shifted below ***
                        subhalo_vels = halo_vels[subhalo_pids, :]  # Velocities *** NOTE: These are shifted below ***
                        subhalo_npart = len(subhalo_pids)

                        # =============== Compute mean positions and velocities and wrap the halos ===============

                        # Compute the shifted mean position in the dimension ixyz
                        mean_subhalo_pos = subhalo_poss.mean(axis=0)

                        # Centre the halos about the mean in the dimension ixyz
                        subhalo_poss -= mean_subhalo_pos

                        # Compute halo's energy
                        subhalo_energy, sKE, sGE = halo_energy_calc(subhalo_poss, subhalo_vels, subhalo_npart,
                                                                    pmass, redshift, G, h, soft)

                        new_subvlcoeff = vlcoeff / 2

                        iter_subhalo_pids = subhalo_pids
                        iter_subhalo_poss = subhalo_poss
                        iter_subhalo_vels = subhalo_vels

                        while sKE / sGE > 1 and subhalo_npart >= 10 and new_subvlcoeff >= 0.8:

                            new_subvlcoeff -= 0.1

                            # Define the velocity space linking length
                            sub_vlinkl = new_subvlcoeff / vlcoeff * vlinkl_halo_indp * (1600 / 200) ** (1 / 6) \
                                         * (pmass * subhalo_npart) ** (1 / 3)

                            # Define the phase space vectors for this halo
                            subhalo_phases = np.concatenate((iter_subhalo_poss, iter_subhalo_vels), axis=1)

                            # Query this halo in phase space to split apart halos which are found to
                            # be distinct in phase space
                            result = find_phase_space_halos(subhalo_phases, sub_linkl, sub_vlinkl)
                            iter_part_subhaloids, iter_assigned_subparts = result

                            # Find the halos with 10 or more particles by finding the unique IDs in the particle
                            # halo ids array and finding those IDs that are assigned to 10 or more particles
                            iter_unique, iter_subcounts = np.unique(iter_part_subhaloids, return_counts=True)
                            unique_iter_subhaloids = iter_unique[np.where(iter_subcounts >= 10)]
                            iter_subcounts = iter_subcounts[np.where(iter_subcounts >= 10)]

                            # Remove the null -2 value for single particle halos
                            iter_subcounts = iter_subcounts[np.where(unique_iter_subhaloids >= 0)]
                            unique_iter_subhaloids = unique_iter_subhaloids[np.where(unique_iter_subhaloids >= 0)]

                            if unique_iter_subhaloids.size != 0:

                                # Sort IDs by count
                                unique_iter_subhaloids = unique_iter_subhaloids[np.argsort(iter_subcounts)]

                                iterID = unique_iter_subhaloids[0]
                                for siID in unique_iter_subhaloids[1:]:
                                    unique_subhaloids.append(newsubID_iter)
                                    assigned_subparts[newsubID_iter] = iter_subhalo_pids[
                                        np.array(list(iter_assigned_subparts[siID]), dtype=int)]
                                    newsubID_iter += 100

                                # Extract halo data for this phase space defined halo ID
                                # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
                                psubhalo_pids = np.array(list(iter_assigned_subparts[iterID]), dtype=int)
                                iter_subhalo_pids = iter_subhalo_pids[psubhalo_pids]
                                iter_subhalo_poss = halo_poss[iter_subhalo_pids, :]
                                iter_subhalo_vels = halo_vels[iter_subhalo_pids, :]
                                subhalo_npart = iter_subhalo_pids.size

                                # Compute halo's energy
                                subhalo_energy, sKE, sGE = halo_energy_calc(iter_subhalo_poss, iter_subhalo_vels,
                                                                            subhalo_npart, pmass, redshift, G, h, soft)

                            else:
                                subhalo_npart = 0

                        if subhalo_npart >= 10:

                            # Extract halo data for this phase space defined halo ID
                            # Particle ID *** NOTE: Overwrites IDs which started at 1 ***
                            subhalo_pids = iter_subhalo_pids
                            subhalo_poss = halo_poss[subhalo_pids, :] + mean_halo_pos  # Positions *** NOTE: these are shifted below ***
                            subhalo_vels = halo_vels[subhalo_pids, :]  # Velocities *** NOTE: these are shifted below ***
                            subhalo_npart = subhalo_pids.size

                            # Increment mass counter
                            if 10 <= subhalo_npart < 20:
                                plus10subs += 1
                            elif 20 <= subhalo_npart < 100:
                                plus20subs += 1
                            elif 100 <= subhalo_npart < 1000:
                                plus100subs += 1
                            elif 1000 <= subhalo_npart < 10000:
                                plus1000subs += 1
                            elif subhalo_npart >= 10000:
                                plus10000subs += 1

                            # =============== Compute mean positions and velocities and wrap the halos ===============

                            # Compute the shifted mean position in the dimension ixyz
                            mean_subhalo_pos = subhalo_poss.mean(axis=0)

                            # Centre the halos about the mean in the dimension ixyz
                            subhalo_poss -= mean_subhalo_pos

                            # Compute the mean velocity in the dimension ixyz
                            mean_subhalo_vel = subhalo_vels.mean(axis=0)

                            # Increment the newID such that the IDs are sequential
                            sub_newID += 1

                            # Reassign the IDs such that they are sequential for halos with 10 or more particles
                            final_part_subhaloids[subhalo_pids] = sub_newID

                            # Reassign the IDs such that they are sequential for halos with 10 or more particles
                            part_haloids[halo_pids[subhalo_pids], 1] = sub_newID

                            # Open root group
                            snap = h5py.File(savepath + 'halos_' + str(snapshot) + '.hdf5', 'r+')
                            halo = snap[str(newID)]

                            # Create datasets in the current halo's group in the HDF5 file
                            subhalo = halo.create_group(str(sub_newID))  # create halo group
                            subhalo.create_dataset('Subhalo_Part_IDs', shape=[len(subhalo_pids)],
                                                   dtype=int, compression='gzip', data=halo_pids[subhalo_pids])
                            subhalo.create_dataset('Subhalo_Pos', shape=subhalo_poss.shape, dtype=float,
                                                   compression='gzip', data=subhalo_poss)
                            subhalo.create_dataset('Subhalo_Vel', shape=subhalo_vels.shape, dtype=float,
                                                   compression='gzip', data=subhalo_vels)
                            subhalo.create_dataset('subhalo_mean_pos', shape=mean_subhalo_pos.shape, dtype=float,
                                                   compression='gzip', data=mean_subhalo_pos)  # mean position
                            subhalo.create_dataset('subhalo_mean_vel', shape=mean_subhalo_vel.shape, dtype=float,
                                                   compression='gzip', data=mean_subhalo_vel)  # mean velocity
                            subhalo.attrs['subhalo_nPart'] = subhalo_npart  # number of particles in halo
                            subhalo.attrs['halo_energy'] = subhalo_energy  # halo energy
                            subhalo.attrs['KE'] = sKE  # kinetic energy
                            subhalo.attrs['GE'] = sGE  # gravitational binding energy

                            # # Assign the full halo IDs array to the snapshot group
                            # halo.create_dataset('Subhalo_IDs', shape=final_part_subhaloids.shape, dtype=int,
                            #                     compression='gzip', data=final_part_subhaloids)

                            snap.close()

    # Open root group
    snap = h5py.File(savepath + 'halos_' + str(snapshot) + '.hdf5', 'r+')

    # Assign the full halo IDs array to the snapshot group
    snap.create_dataset('Halo_IDs', shape=part_haloids.shape, dtype=int, compression='gzip', data=part_haloids)

    snap.close()

    # Find the halos with 10 or more particles by finding the unique IDs in the particle
    # halo ids array and finding those IDs that are assigned to 10 or more particles
    unique, counts = np.unique(part_haloids, return_counts=True)

    unique_haloids = unique[np.where(counts >= 10)]

    # Remove the null -2 value for single particle halos
    unique_haloids = unique_haloids[np.where(unique_haloids != -2)]

    # Print the number of halos found by the halo finder in >10, >100, >1000, >10000 criteria
    print('Post Sub Structure Test')
    print(unique_haloids.size, 'halos found with 10 or more particles')
    print(unique[np.where(counts >= 20)].size - 1, 'halos found with 20 or more particles')
    print(unique[np.where(counts >= 100)].size - 1, 'halos found with 100 or more particles')
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

    if 'energy_dicts' not in os.listdir(os.getcwd()):
        os.mkdir('energy_dicts')

    with open('energy_ba_' + snapshot + '.pck', 'wb') as pfile:
        pickle.dump(energy_dict, pfile)

    if 'vel_linkl' not in os.listdir(os.getcwd()):
        os.mkdir('vel_linkl')

    with open('vel_linkl/valpha_' + snapshot + '.pck', 'wb') as pfile:
        pickle.dump(valpha_dict, pfile)


# # Convert task ID from apollo jobs to the correct form for the halo finder
# snapID = int(sys.argv[1])
# if snapID < 10:
#     snap = '00' + str(snapID)
# else:
#     snap = '0' + str(snapID)
# start = time.time()
# hosthalofinder(snap, batchsize=2000000, savepath='halo_snapshots/', vlcoeff=1.25)
# print('Total: ', time.time()-start, snap)

# def seq_check(halopath='halo_snapshots/'):
#
#     non_seq = []
#     snaplist = []
#     for snap in range(0, 62):
#         if snap < 10:
#             snaplist.append('00' + str(snap))
#         elif snap >= 10:
#             snaplist.append('0' + str(snap))
#
#     for snap in snaplist:
#
#         hdf = h5py.File(halopath + 'halos_' + str(snap) + '.hdf5', 'r')
#
#         try:
#             haloids = hdf['Halo_IDs'][...]
#         except KeyError:
#             non_seq.append(snap)
#             continue
#
#         uni_haloids = np.unique(haloids)
#
#         uni_haloids = uni_haloids[np.where(uni_haloids >= 0)]
#
#         for i in range(uni_haloids.size - 1):
#             if uni_haloids[i + 1] != uni_haloids[i] + 1:
#                 non_seq.append(snap)
#                 break
#
#     print(non_seq)
#     return non_seq

def main(snap):
    hosthalofinder(snap, llcoeff=0.2, sub_llcoeff=0.1, batchsize=2000000, savepath='halo_snapshots/', vlcoeff=10.0)


# # Create a snapshot list (past to present day) for looping
# snaplist = []
# for snap in range(0, 62):
#     if snap < 10:
#         try:
#             hdf = h5py.File(f'halo_snapshots/halos_00' + str(snap) + '.hdf5', 'r')
#
#             if 'Halo_IDs' in list(hdf.keys()):
#                 continue
#             else:
#                 print(f'Halo IDs not in {snap}')
#             hdf.close()
#         except OSError:
#             print(f'{snap} does not exist...')
#
#         snaplist.append('00' + str(snap))
#     elif snap >= 10:
#         try:
#             hdf = h5py.File(f'halo_snapshots/halos_0' + str(snap) + '.hdf5', 'r')
#
#             if 'Halo_IDs' in list(hdf.keys()):
#                 continue
#             else:
#                 print(f'Halo IDs not in {snap}')
#             hdf.close()
#         except OSError:
#             print(f'{snap} does not exist...')
#
#         snaplist.append('0' + str(snap))
#
# print(snaplist)

# start = time.time()
# if __name__ == '__main__':
# #     start = time.time()
# #     main(snap)
# #     print(f'{snap}: ', time.time() - start)
# #     snaplist = seq_check()
#     pool = mp.Pool(mp.cpu_count() - 2)
#     pool.map(main, snaplist)
#     pool.close()
#     pool.join()
#     print('Total: ', time.time() - start)

# if __name__ == '__main__':
#
#     for snap in snaplist:
#         start = time.time()
#         main(snap)
#         print('Total: ', time.time() - start, snap)
#         break


# def main(snap):
#     directProgDescWriter(snap, halopath='halo_snapshots/', savepath='MergerGraphs/', part_threshold=10)
#
#
# # Create a snapshot list (past to present day) for looping
# snaplist = []
# for snap in range(0, 62):
#     # if snap % 2 != 0: continue
#     if snap < 10:
#         snaplist.append('00' + str(snap))
#     elif snap >= 10:
#         snaplist.append('0' + str(snap))
#
# if __name__ == '__main__':
#     pool = mp.Pool(int(mp.cpu_count() - 2))
#     pool.map(main, snaplist)
#     pool.close()
#     pool.join()

# for snap in snaplist:
#     print(snap)
#     main(snap)
#
#     break