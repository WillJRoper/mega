import numpy as np


def find_hydro_part_haloids(part_haloids, part_types, tree, pos, npart):
    # Initialise the halo ID array
    all_part_haloids = np.full(npart, -1, dtype=int)

    # Define boolean array for particle type
    not_dm_inds = part_types != 1

    # Assign dark matter particle halo ids
    all_part_haloids[~not_dm_inds] = part_haloids

    # Query the tree
    query = tree.query(pos[not_dm_inds], k=1)[1]

    # Assign haloids
    for ind, dm_ind in enumerate(query):
        all_part_haloids[not_dm_inds][ind] = part_haloids[dm_ind]

    return all_part_haloids
