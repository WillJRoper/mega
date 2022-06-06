import numpy as np
from scipy.spatial import cKDTree

import mega.core.utilities as utils
from mega.halo_core.halo import Halo
from mega.core.timing import timer


def find_phase_space_halos(halo_phases):
    """

    :param halo_phases:
    :return:
    """
    # =============== Initialise The Halo Finder Variables/Arrays and The KD-Tree ===============

    # Initialise arrays and dictionaries for storing halo data
    phase_part_haloids = np.full(halo_phases.shape[0], -1,
                                 dtype=int)  # halo ID of the halo each particle is in
    phase_assigned_parts = {}  # Dictionary to store the particles in a particular halo
    # A dictionary where each key is an initial subhalo ID and the item is the subhalo IDs it has been linked with
    phase_linked_halos_dict = {}
    # Final halo ID of linked halos (index is initial halo ID)
    final_phasehalo_ids = np.full(halo_phases.shape[0], -1, dtype=int)

    # Initialise subhalo ID counter (IDs start at 0)
    ihaloid = -1

    # Initialise the halo kd tree in 6D phase space
    halo_tree = cKDTree(halo_phases, leafsize=16, compact_nodes=True,
                        balanced_tree=True)

    query = halo_tree.query_ball_point(halo_phases, r=np.sqrt(2))

    # Loop through query results assigning initial halo IDs
    for query_part_inds in iter(query):

        query_part_inds = np.array(query_part_inds, dtype=int)

        # If only one particle is returned by the query and it is new it is a 'single particle halo'
        if query_part_inds.size == 1:
            # Assign the 'single particle halo' halo ID to the particle
            phase_part_haloids[query_part_inds] = -2

        # # Find the previous halo ID associated to these particles
        # this_halo_ids = halo_ids[query_part_inds]
        # uni_this_halo_ids = set(this_halo_ids)
        # if len(uni_this_halo_ids) > 1:
        #     query_part_inds = query_part_inds[np.where(this_halo_ids == this_halo_ids[0])]

        # Find only the particles not already in a halo
        new_parts = query_part_inds[
            np.where(phase_part_haloids[query_part_inds] < 0)]

        # If all particles are new increment the halo ID and assign a new halo
        if new_parts.size == query_part_inds.size:

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
                linked_halos = linked_halos_set.union(
                    *[phase_linked_halos_dict.get(halo)
                      for halo in uni_cont_halos])

            # Find the minimum halo ID to make the final halo ID
            final_ID = min(linked_halos)

            # Assign the linked halos to all the entries in the linked halos dictionary
            phase_linked_halos_dict.update(
                dict.fromkeys(list(linked_halos), linked_halos))

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


def get_real_halos_recurse(tictoc, halo, vlinkl_halo_indp, linkl, meta):
    """

    :param halo:
    :param boxsize:
    :param vlinkl_halo_indp:
    :param linkl:
    :param decrement:
    :param redshift:
    :param G:
    :param soft:
    :param min_vlcoeff:
    :param cosmo:
    :return:
    """

    # Initialise list to store finished halos
    results = []

    # Define the phase space linking length
    vlinkl = halo.vlcoeff * vlinkl_halo_indp * halo.mass ** (1 / 3)

    # Define the phase space vectors for this halo
    halo_phases = np.concatenate((halo.pos / linkl,
                                  halo.vel_with_hubflow / vlinkl),
                                 axis=1)

    # Query these particles in phase space to find distinct bound halos
    part_haloids, assigned_parts = find_phase_space_halos(halo_phases)

    # Loop over the halos found in phase space
    while len(assigned_parts) > 0:

        # Get the next halo from the dictionary and ensure
        # it has more than 10 particles
        key, val = assigned_parts.popitem()
        if len(val) < 10:
            continue

        # Extract halo particle data
        this_pids = list(val)

        # Instantiate halo object (auto calculates energy)
        new_halo = Halo(tictoc, halo.pids[this_pids],
                        halo.shifted_inds[this_pids],
                        halo.sim_pids[this_pids],
                        halo.pos[this_pids, :],
                        halo.vel[this_pids, :],
                        halo.types[this_pids],
                        halo.masses[this_pids],
                        halo.vlcoeff, meta)

        if new_halo.real or new_halo.vlcoeff <= meta.min_vlcoeff:

            # Compute the halo properties
            new_halo.compute_props(meta.G)

            # Get and store the memory footprint of this halo
            new_halo.memory = utils.get_size(new_halo)

            # Limit memory footprint of stored halo
            new_halo.clean_halo()

            # Store the resulting halo
            results.append(new_halo)

        else:

            # Decrement the velocity space linking length coefficient
            new_halo.decrement(meta.decrement)

            # We need to run this halo again
            temp_res = get_real_halos(tictoc, new_halo, vlinkl_halo_indp,
                                      linkl, meta)

            # Include these results
            for h in temp_res:
                results.append(h)

    return results


def get_real_halos(tictoc, halo, vlinkl_halo_indp, linkl, meta):
    """

    :param halo:
    :param boxsize:
    :param vlinkl_halo_indp:
    :param linkl:
    :param decrement:
    :param redshift:
    :param G:
    :param soft:
    :param min_vlcoeff:
    :param cosmo:
    :return:
    """

    # Initialise list to store finished halos
    results = []

    # Intialise set to hold halos
    test_halos = {halo, }

    # Loop until we have no halos to test
    while len(test_halos) > 0:

        # Get a halo to work on
        halo = test_halos.pop()

        # Define the phase space linking length
        vlinkl = halo.vlcoeff * vlinkl_halo_indp * halo.mass ** (1 / 3)

        # Define the phase space vectors for this halo
        halo_phases = np.concatenate((halo.true_pos / linkl,
                                      halo.vel_with_hubflow / vlinkl),
                                     axis=1)

        # Query these particles in phase space to find distinct bound halos
        part_haloids, assigned_parts = find_phase_space_halos(halo_phases)

        # Loop over the halos found in phase space
        while len(assigned_parts) > 0:

            # Get the next halo from the dictionary and ensure
            # it has more than 10 particles
            key, val = assigned_parts.popitem()
            if len(val) < 10:
                continue

            # Extract halo particle data
            this_pids = list(val)

            # Instantiate halo object (auto calculates energy)
            new_halo = Halo(tictoc, halo.pids[this_pids],
                            halo.shifted_inds[this_pids],
                            halo.sim_pids[this_pids],
                            halo.true_pos[this_pids, :],
                            halo.true_vel[this_pids, :],
                            halo.types[this_pids],
                            halo.masses[this_pids],
                            halo.int_nrg[this_pids],
                            halo.vlcoeff, meta, parent=halo)

            if new_halo.real or new_halo.vlcoeff < meta.min_vlcoeff:

                # Compute the halo properties
                new_halo.compute_props(meta)

                # Limit memory footprint of stored halo
                new_halo.clean_halo()

                # Get the memory footprint of this halo
                new_halo.memory = utils.get_size(new_halo)

                # Store the resulting halo
                results.append(new_halo)

            else:

                # Decrement the velocity space linking length coefficient
                new_halo.decrement(meta.decrement)

                # Add this halo to test_halos to be tested later
                test_halos.update({new_halo, })

    return results


@timer("Host-Phase")
def get_real_host_halos(tictoc, halo, meta):
    """

        NOTE: We use the maximum spatial linking length in the eventuality of
        multiple options since the virialisation and energy considerations
        are what allow a halo to exit the iteration not the spatial
        linking length itself.

    :param tictoc:
    :param halo:
    :param meta:

    :return:
    """

    return get_real_halos(tictoc, halo, meta.vlinkl_indp, np.max(meta.linkl),
                          meta)


@timer("Sub-Phase")
def get_real_sub_halos(tictoc, halo, meta):
    """

        NOTE: We use the maximum spatial linking length in the eventuality of
        multiple options since the virialisation and energy considerations
        are what allow a halo to exit the iteration not the spatial
        linking length itself.

    :param tictoc:
    :param halo:
    :param meta:

    :return:
    """
    return get_real_halos(tictoc, halo, meta.sub_vlinkl_indp,
                          np.max(meta.sub_linkl), meta)
