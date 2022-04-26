from mega.core.timing import timer


@timer("Hydro-DM-CrossRef")
def link_halo_species(tictoc, meta, halo_pinds, tree, pos, linkl):
    """

    :param tictoc:
    :param halo_pinds:
    :param tree:
    :param pos:
    :param linkl:
    :return:
    """

    # Initialise a dictionary to store the found halos
    other_parts = {i: set() for i in halo_pinds}

    # Loop over the search bins
    for halo in halo_pinds:

        # Query the tree for the nearest particle within the linking length
        dists, query = tree.query(pos[list(halo_pinds[halo]), :], k=1,
                           distance_upper_bound=linkl)

        # Store the particles we have found for this halo
        for d, ind in zip(dists, query):

            # If a neighbour was returned include it
            # NOTE: if a neighbour was not returned the distance is inf
            if d < meta.boxsize[0]:
                other_parts[halo].update({ind, })

    return other_parts

