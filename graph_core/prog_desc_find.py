import numpy as np
from core.timing import timer
from graph_core.graph_halo import Halo


@timer("Progenitor-Linking")
def get_direct_prog(tictoc, meta, prog_haloids, prog_reals, prog_nparts):
    """

    :param meta:
    :param prog_haloids:
    :param prog_reals:
    :param prog_nparts:
    :param npart:
    :return:
    """

    # Find the unique halo IDs and the number of times each appears
    uniprog_haloids, uniprog_counts = np.unique(prog_haloids,
                                                return_counts=True)

    # Remove single particle halos (ID=-2)
    okinds = np.where(uniprog_haloids >= 0)
    uniprog_haloids = uniprog_haloids[okinds]
    uniprog_counts = uniprog_counts[okinds]

    # # Halos only linked if they have link_thresh or more particles in common
    # okinds = uniprog_counts >= meta.link_thresh
    # uniprog_haloids = uniprog_haloids[okinds]
    # uniprog_counts = uniprog_counts[okinds]

    # Get the reality flag
    preals = prog_reals[uniprog_haloids]

    # Get only real halos
    uniprog_haloids = uniprog_haloids[preals]
    uniprog_counts = uniprog_counts[preals]
    preals = preals[preals]

    # Find the number of progenitor halos from the size of the unique array
    nprog = uniprog_haloids.size

    # Assign the corresponding number of particles in each progenitor
    # for sorting and storing.
    prog_npart = prog_nparts[uniprog_haloids]

    # Sort the halo IDs and number of particles in each progenitor
    # halo by their cont to the current halo (number of particles
    # from the current halo in the progenitor or descendant)
    sorting_inds = uniprog_counts.argsort()[::-1]
    prog_npart = prog_npart[sorting_inds]
    prog_haloids = uniprog_haloids[sorting_inds]
    prog_npart_cont = uniprog_counts[sorting_inds]
    preals = preals[sorting_inds]

    return nprog, prog_haloids, prog_npart, prog_npart_cont, preals


@timer("Descendant-Linking")
def get_direct_desc(tictoc, meta, desc_haloids, desc_nparts):
    """

    :param meta:
    :param desc_haloids:
    :param desc_nparts:
    :param npart:
    :return:
    """

    # Find the unique halo IDs and the number of times each appears
    unidesc_haloids, unidesc_counts = np.unique(desc_haloids,
                                                return_counts=True)

    # Remove single particle halos (ID=-2)
    okinds = np.where(unidesc_haloids >= 0)
    unidesc_haloids = unidesc_haloids[okinds]
    unidesc_counts = unidesc_counts[okinds]

    # # Halos only linked if they have link_thresh or more particles in common
    # okinds = unidesc_counts >= meta.link_thresh
    # unidesc_haloids = unidesc_haloids[okinds]
    # unidesc_counts = unidesc_counts[okinds]

    # Find the number of descendant halos from the size of the unique array
    ndesc = unidesc_haloids.size

    # Assign the corresponding number of particles in each progenitor
    # for sorting and storing. This can be done simply by using the
    # ID of the progenitor since again np.unique returns
    # sorted results.
    desc_npart = desc_nparts[unidesc_haloids]

    # Sort the halo IDs and number of particles in each progenitor
    # halo by their cont to the current halo (number of particles
    # from the current halo in the progenitor or descendant)
    sorting_inds = unidesc_counts.argsort()[::-1]
    desc_npart = desc_npart[sorting_inds]
    desc_haloids = unidesc_haloids[sorting_inds]
    desc_npart_cont = unidesc_counts[sorting_inds]

    return ndesc, desc_haloids, desc_npart, desc_npart_cont


@timer("Local-Linking")
def local_linking_loop(tictoc, meta, halo_tasks, part_progids, prog_nparts,
                       part_descids, prog_reals, desc_nparts,
                       prog_rank_pidbins, desc_rank_pidbins,
                       prog_pids, desc_pids):

    # Initialise dictionary for results
    results = {}

    # Set up a dictionary to store particles on other ranks
    other_rank_prog_parts = {r: {} for r in range(meta.nranks)}
    other_rank_desc_parts = {r: {} for r in range(meta.nranks)}

    # Loop over the halo tasks on this rank
    for ihalo in halo_tasks:

        # Extract particles for this halo
        parts = halo_tasks[ihalo]
        npart = len(parts)

        # If the progenitor snapshot exists
        if not meta.isfirst:

            # Get the ranks for each particle, returned values are
            # the index of right bin edge
            prog_ranks = np.digitize(parts, prog_rank_pidbins) - 1

            # Store progenitor particles on other ranks
            prog_parts = []
            for r, part in zip(prog_ranks, parts):
                if r == meta.rank:
                    prog_parts.append(part)
                else:
                    other_rank_prog_parts[r].setdefault(ihalo, []).append(part)
            prog_parts = np.array(prog_parts, dtype=int)

            # Link progenitors on this rank
            progids = part_progids[np.in1d(prog_pids, prog_parts)]
            (nprog, prog_haloids, prog_npart,
             prog_npart_cont, preals) = get_direct_prog(tictoc, meta,
                                                        progids,
                                                        prog_reals,
                                                        prog_nparts)

        else:  # there is no progenitor snapshot
            nprog = 0
            prog_npart = np.array([], copy=False, dtype=int)
            prog_haloids = np.array([], copy=False, dtype=int)
            prog_npart_cont = np.array([], copy=False, dtype=int)
            preals = np.array([], copy=False, dtype=bool)

        # If descendant snapshot exists
        if not meta.isfinal:

            # Get the ranks for each particle, returned values are
            # the index of right bin edge
            desc_ranks = np.digitize(parts, desc_rank_pidbins) - 1

            # Store descendant particles on other ranks
            desc_parts = []
            for r, part in zip(desc_ranks, parts):
                if r == meta.rank:
                    desc_parts.append(part)
                else:
                    other_rank_desc_parts[r].setdefault(ihalo, []).append(part)
            desc_parts = np.array(desc_parts, dtype=int)

            # Link descendants on this rank
            descids = part_descids[np.in1d(desc_pids, desc_parts)]
            (ndesc, desc_haloids, desc_npart,
             desc_npart_cont) = get_direct_desc(tictoc, meta, descids,
                                                desc_nparts)

        else:  # there is no descendant snapshot
            ndesc = 0
            desc_npart = np.array([], copy=False, dtype=int)
            desc_haloids = np.array([], copy=False, dtype=int)
            desc_npart_cont = np.array([], copy=False, dtype=int)

        # Populate halo object with results
        results[ihalo] = Halo(parts, npart, nprog, prog_haloids, prog_npart,
                              prog_npart_cont, None, None, preals,
                              ndesc, desc_haloids, desc_npart,
                              desc_npart_cont,
                              None, None)

    return results, other_rank_prog_parts, other_rank_desc_parts


@timer("Foreign-Linking")
def foreign_linking_loop(tictoc, meta, comm, other_rank_prog_parts,
                         other_rank_desc_parts, part_progids, prog_nparts,
                         part_descids, prog_reals, desc_nparts,
                         prog_pids, desc_pids):

    # Lets share the halos with progenitors and descendants
    for other_rank in range(meta.nranks):
        other_rank_prog_parts[other_rank] = comm.gather(
            other_rank_prog_parts[other_rank],
            root=other_rank)
        other_rank_desc_parts[other_rank] = comm.gather(
            other_rank_desc_parts[other_rank],
            root=other_rank)

    # If the progenitor snapshot exists
    if not meta.isfirst:

        # We can now loop over the halos we've been given by other
        # ranks for progenitors
        for other_rank, halo_dict in enumerate(
                other_rank_prog_parts[meta.rank]):
            other_rank_prog_parts[other_rank] = {}
            for ihalo in halo_dict:
                # Extract particles for this halo
                prog_parts = halo_dict[ihalo]

                # Link progenitors on this rank
                progids = part_progids[np.in1d(prog_pids, prog_parts)]
                (nprog, prog_haloids, prog_npart,
                 prog_npart_cont, preals) = get_direct_prog(tictoc,
                                                            meta,
                                                            progids,
                                                            prog_reals,
                                                            prog_nparts)

                # Store what we have found for this halo
                d = other_rank_prog_parts[other_rank]
                d[ihalo] = Halo(None, None, nprog, prog_haloids, prog_npart,
                                prog_npart_cont, None, None, preals,
                                None, None, None, None, None, None)
    else:
        for other_rank, halo_dict in enumerate(
                other_rank_prog_parts[meta.rank]):
            other_rank_prog_parts[other_rank] = {}

    # If descendant snapshot exists
    if not meta.isfinal:

        # We can now loop over the halos we've been given by other
        # ranks for descendants
        for other_rank, halo_dict in enumerate(
                other_rank_desc_parts[meta.rank]):
            other_rank_desc_parts[other_rank] = {}
            for ihalo in halo_dict:
                # Extract particles for this halo
                desc_parts = halo_dict[ihalo]

                # Link descendants on this rank
                descids = part_descids[np.in1d(desc_pids, desc_parts)]
                (ndesc, desc_haloids, desc_npart,
                 desc_npart_cont) = get_direct_desc(tictoc, meta, descids,
                                                    desc_nparts)

                # Store what we have found for this halo
                d = other_rank_desc_parts[other_rank]
                d[ihalo] = Halo(None, None, None, None, None, None, None, None,
                                None, ndesc, desc_haloids, desc_npart,
                                desc_npart_cont, None, None)

    else:
        for other_rank, halo_dict in enumerate(
                other_rank_desc_parts[meta.rank]):
            other_rank_desc_parts[other_rank] = {}

    # Send back the progenitors and descendants we have found
    for other_rank in range(meta.nranks):
        other_rank_prog_parts[other_rank] = comm.gather(
            other_rank_prog_parts[other_rank],
            root=other_rank)
        other_rank_desc_parts[other_rank] = comm.gather(
            other_rank_desc_parts[other_rank],
            root=other_rank)

    return other_rank_prog_parts, other_rank_desc_parts


@timer("Stitching")
def update_halos(tictoc, meta, results,
                 other_rank_prog_parts, other_rank_desc_parts):
    # Loop over progenitor results from other ranks
    for halo_dict in other_rank_prog_parts[meta.rank]:

        # Avoid empty entries
        if halo_dict is not None:

            # Get halo keys
            halos = list(halo_dict.keys())

            # Loop over foreign progenitor results
            for ihalo in halos:
                # Update our version of this halo
                results[ihalo].update_progs(halo_dict.pop(ihalo))

    # Loop over descendant results from other ranks
    for halo_dict in other_rank_desc_parts[meta.rank]:

        # Avoid empty entries
        if halo_dict is not None:

            # Get halo keys
            halos = list(halo_dict.keys())

            # Loop over foreign descendant results
            for ihalo in halos:
                # Update our version of this halo
                results[ihalo].update_descs(halo_dict.pop(ihalo))

    return results


@timer("Cleaning")
def clean_halos(tictoc, meta, results):

    # Loop over halos and clean up progenitors and descendants
    for ihalo in results:
        results[ihalo].clean_progs(meta)
        results[ihalo].clean_descs(meta)

    return results
