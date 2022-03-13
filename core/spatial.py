import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree


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
    part_haloids = np.full(npart, -1,
                           dtype=np.int32)  # halo ID containing each particle
    assigned_parts = defaultdict(
        set)  # dictionary to store the particles in a particular halo
    # A dictionary where each key is an initial halo ID and the item is the halo IDs it has been linked with
    linked_halos_dict = defaultdict(set)
    final_halo_ids = np.full(npart, -1,
                             dtype=np.int32)  # final halo ID of linked halos (index is initial halo ID)

    # Initialise the halo ID counter (IDs start at 0)
    ihaloid = -1

    # =============== Assign Particles To Initial Halos ===============

    # Query the tree returning a list of lists
    query = tree.query_ball_point(pos, r=linkl)

    # Loop through query results assigning initial halo IDs
    for query_part_inds in iter(query):

        # Convert the particle index list to an array for ease of use
        query_part_inds = np.array(query_part_inds, copy=False, dtype=int)

        # Assert that the query particle is returned by the tree query. Otherwise the program fails
        assert query_part_inds.size != 0, 'Must always return particle that you are sitting on'

        # Find only the particles not already in a halo
        new_parts = query_part_inds[
            np.where(part_haloids[query_part_inds] == -1)]

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
                linked_halos = linked_halos_set.union(
                    *[linked_halos_dict.get(halo) for halo in uni_cont_halos])

            # Find the minimum halo ID to make the final halo ID
            final_ID = min(linked_halos)

            # Assign the linked halos to all the entries in the linked halos dictionary
            linked_halos_dict.update(
                dict.fromkeys(list(linked_halos), linked_halos))

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
    part_subhaloids = np.full(halo_pos.shape[0], -1,
                              dtype=int)  # subhalo ID of the halo each particle is in
    assignedsub_parts = defaultdict(
        set)  # Dictionary to store the particles in a particular subhalo
    # A dictionary where each key is an initial subhalo ID and the item is the subhalo IDs it has been linked with
    linked_subhalos_dict = defaultdict(set)
    # Final subhalo ID of linked halos (index is initial subhalo ID)
    final_subhalo_ids = np.full(halo_pos.shape[0], -1, dtype=int)

    # Initialise subhalo ID counter (IDs start at 0)
    isubhaloid = -1

    npart = halo_pos.shape[0]

    # Build the halo kd tree
    tree = cKDTree(halo_pos, leafsize=32, compact_nodes=True,
                   balanced_tree=True)

    query = tree.query_ball_point(halo_pos, r=sub_linkl)

    # Loop through query results
    for query_part_inds in iter(query):

        # Convert the particle index list to an array for ease of use.
        query_part_inds = np.array(query_part_inds, copy=False, dtype=int)

        # Assert that the query particle is returned by the tree query. Otherwise the program fails
        assert query_part_inds.size != 0, 'Must always return particle that you are sitting on'

        # Find only the particles not already in a halo
        new_parts = query_part_inds[
            np.where(part_subhaloids[query_part_inds] == -1)]

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
            uni_cont_subhalos = uni_cont_subhalos[
                np.where(uni_cont_subhalos != -1)]

            # If there is only one subhalo ID returned avoid the slower code to combine IDs
            if uni_cont_subhalos.size == 1:

                # Get the list of linked subhalos linked to the current subhalo from the linked subhalo dictionary
                linked_subhalos = linked_subhalos_dict[uni_cont_subhalos[0]]

            else:

                # Get all linked subhalos from the dictionary so as to not miss out any subhalos IDs that are linked
                # but not returned by this particular query
                linked_subhalos_set = set()  # initialise linked subhalo set
                linked_subhalos = linked_subhalos_set.union(
                    *[linked_subhalos_dict.get(subhalo)
                      for subhalo in uni_cont_subhalos])

            # Find the minimum subhalo ID to make the final subhalo ID
            final_ID = min(linked_subhalos)

            # Assign the linked subhalos to all the entries in the linked subhalos dict
            linked_subhalos_dict.update(
                dict.fromkeys(list(linked_subhalos), linked_subhalos))

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


def spatial_node_task(thisTask, pos, tree, linkl, npart):
    # =============== Run The Halo Finder And Reduce The Output ===============

    # Run the halo finder for this snapshot at the host linking length and get the spatial catalog
    task_part_haloids, task_assigned_parts = find_halos(tree, pos, linkl,
                                                        npart)

    # Get the positions
    halo_pids = {}
    while len(task_assigned_parts) > 0:
        item = task_assigned_parts.popitem()
        halo, part_inds = item
        # halo_pids[(thisTask, halo)] = np.array(list(part_inds))
        halo_pids[(thisTask, halo)] = frozenset(part_inds)

    return halo_pids


def get_sub_halos(halo_pids, halo_pos, sub_linkl):
    # Do a spatial search for subhalos
    part_subhaloids, assignedsub_parts = find_subhalos(halo_pos, sub_linkl)

    # Get the positions
    subhalo_pids = {}
    for halo in assignedsub_parts:
        subhalo_pids[halo] = halo_pids[list(assignedsub_parts[halo])]

    return subhalo_pids
