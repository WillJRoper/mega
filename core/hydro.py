import numpy as np

from core.timing import timer


@timer("Hydro-Spatial")
def find_hydro_haloids(tictoc, rank_hydro_parts, halo_ids,
                       dm_part_haloids, tree, pos, meta):
    """A function to find the nearest dark matter particle to each
        hydro particle and assign the hydro particle the halo ID of the dark
        matter particle.

        NOTE: For now this is dumb and redoes the DM too. In reality this
        has a minimal wallclock contribution but does double up work
        previously done. We could shift the dm indices to their position in
        the all type array but I doubt there's much time to be saved.

    :param tictoc:
    :param halo_pids:
    :param rank_hydro_parts:
    :param dm_part_haloids:
    :param tree:
    :param pos:
    :param part_types:
    :return:
    """

    # Initialise a dictionary to store the found halos
    halo_pids = {i: set() for i in halo_ids}

    # TODO: We can stick with this tree and only need to query hydro particles
    #  in the central cells but we need the same DM particles as the tree
    #  otherwise the linking falls apart.
    #  Actually we only need sorted arrays
    #  for merger graph which hugely simplifies this song and dance,
    #  we don't even need the input file!

    # Define particle bins for the search
    part_bins = np.linspace(0, rank_hydro_parts.size,
                            int(np.ceil(rank_hydro_parts.size
                                        / meta.spatial_task_size)) + 1,
                            dtype=int)

    # Loop over the search bins
    for ind in range(part_bins.size - 1):

        # Get the edges of this search bin
        low, high = part_bins[ind], part_bins[ind + 1]

        # Query the tree
        query = tree.query(pos[low: high, :], k=1)[1]

        # Assign halo ids
        for ind, dm_ind in enumerate(query):

            # Assign hydro particle
            halo_id = dm_part_haloids[dm_ind]

            if halo_id >= 0:
                halo_pids[halo_id].update({ind, })

    return halo_pids

