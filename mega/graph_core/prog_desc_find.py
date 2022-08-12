import numpy as np
from mega.core.timing import timer
from mega.graph_core.graph_halo import Halo


@timer("Progenitor-Linking")
def get_direct_prog(tictoc, meta, prog_haloids, prog_nparts):
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

    return nprog, prog_haloids, prog_npart, prog_npart_cont


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


@timer("PartID-Decomp")
def sort_prog_desc(tictoc, meta, halo_tasks, prog_pids, desc_pids):

    # Initialise dictionary for results
    results = {}

    # Set up a dictionary to store particles on other ranks
    other_rank_prog_parts = {r: {} for r in range(meta.nranks)}
    other_rank_desc_parts = {r: {} for r in range(meta.nranks)}

    # Loop over the halo tasks on this rank
    for ihalo in halo_tasks:

        # Extract particles for this halo
        parts = halo_tasks[ihalo]
        npart = parts.size

        # If the progenitor snapshot exists
        if not meta.isfirst:

            # Assign the particles on this rank to this rank
            okinds = np.in1d(parts, prog_pids)
            parts_thisrank = parts[okinds]
            other_rank_prog_parts[meta.rank][ihalo] = parts_thisrank

            # If all parts are on this rank there's nothing to do
            if parts_thisrank.size != npart:

                # Get particles not on this rank
                foreign_parts = parts[~okinds]

                # Give the other ranks the particles not on this rank
                for r in range(meta.nranks):
                    if r != meta.rank:
                        other_rank_prog_parts[r][ihalo] = foreign_parts

        # If descendant snapshot exists
        if not meta.isfinal:

            # Assign the particles on this rank to this rank
            okinds = np.in1d(parts, desc_pids)
            parts_thisrank = parts[okinds]
            other_rank_desc_parts[meta.rank][ihalo] = parts_thisrank

            # If all parts are on this rank there's nothing to do
            if parts_thisrank.size != npart:

                # Get particles not on this rank
                foreign_parts = parts[~okinds]

                # Give the other ranks the particles not on this rank
                for r in range(meta.nranks):
                    if r != meta.rank:
                        other_rank_desc_parts[r][ihalo] = foreign_parts

        # Populate halo object with results
        null_entry = np.array([], dtype=int)
        results[ihalo] = Halo(parts, npart, 0, null_entry, null_entry,
                              null_entry, None, None,
                              0, null_entry, null_entry,
                              null_entry,
                              None, None)

    return results, other_rank_prog_parts, other_rank_desc_parts


@timer("Linking")
def linking_loop(tictoc, meta, comm, other_rank_prog_parts,
                 other_rank_desc_parts, part_progids, prog_nparts,
                 part_descids, prog_reals, desc_nparts,
                 prog_pids, desc_pids):

    # Lets share the halos with progenitors and descendants
    comm_tic = tictoc.get_extic()
    for other_rank in range(meta.nranks):
        other_rank_prog_parts[other_rank] = comm.gather(
            other_rank_prog_parts[other_rank],
            root=other_rank)
        other_rank_desc_parts[other_rank] = comm.gather(
            other_rank_desc_parts[other_rank],
            root=other_rank)
    comm_toc = tictoc.get_extoc()

    # Tell the world we've finished communicating
    if meta.verbose:
        tictoc.report("Communicating halos to be linked", comm_tic, comm_toc)

    # If the progenitor snapshot exists
    prog_tic = tictoc.get_extic()
    if not meta.isfirst:

        # We can now loop over the halos we've been given by other
        # ranks for progenitors
        for other_rank, halo_dict in enumerate(
                other_rank_prog_parts[meta.rank]):
            other_rank_prog_parts[other_rank] = {}
            for ihalo in halo_dict:
                # Extract particles for this halo
                prog_parts = halo_dict[ihalo]

                if len(prog_parts) == 0:
                    continue

                # Remove extraneous partiles to shrink the query
                okinds = np.logical_and(prog_pids <= np.max(prog_parts),
                                        prog_pids >= np.min(prog_parts))

                # Get prog ids present on this rank
                parts_on_rank = np.in1d(prog_pids[okinds], prog_parts)
                progids = part_progids[okinds][parts_on_rank]

                # Link progenitors on this rank
                if progids.size > 0:
                    (nprog, prog_haloids, prog_npart,
                     prog_npart_cont) = get_direct_prog(tictoc,
                                                        meta,
                                                        progids,
                                                        prog_nparts)

                    # Store what we have found for this halo
                    d = other_rank_prog_parts[other_rank]
                    d[ihalo] = Halo(None, None, nprog, prog_haloids,
                                    prog_npart, prog_npart_cont, None,
                                    None, None, None, None,
                                    None, None, None)
    else:
        for other_rank, halo_dict in enumerate(
                other_rank_prog_parts[meta.rank]):
            other_rank_prog_parts[other_rank] = {}
    prog_toc = tictoc.get_extoc()

    # Tell the world we've finished progenitor linking
    if meta.verbose:
        tictoc.report("Linking Progenitors", prog_tic, prog_toc)

    # If descendant snapshot exists
    desc_tic = tictoc.get_extic()
    if not meta.isfinal:

        # We can now loop over the halos we've been given by other
        # ranks for descendants
        for other_rank, halo_dict in enumerate(
                other_rank_desc_parts[meta.rank]):
            other_rank_desc_parts[other_rank] = {}
            for ihalo in halo_dict:
                # Extract particles for this halo
                desc_parts = halo_dict[ihalo]

                if len(desc_parts) == 0:
                    continue

                # Remove extraneous partiles to shrink the query
                okinds = np.logical_and(desc_pids <= np.max(desc_parts),
                                        desc_pids >= np.min(desc_parts))

                # Get prog ids present on this rank
                parts_on_rank = np.in1d(desc_pids[okinds], desc_parts)
                descids = part_descids[okinds][parts_on_rank]

                # Link descendants on this rank
                if descids.size > 0:
                    (ndesc, desc_haloids, desc_npart,
                     desc_npart_cont) = get_direct_desc(tictoc, meta, descids,
                                                        desc_nparts)

                    # Store what we have found for this halo
                    d = other_rank_desc_parts[other_rank]
                    d[ihalo] = Halo(None, None, None, None, None, None,
                                    None, None, ndesc, desc_haloids,
                                    desc_npart, desc_npart_cont, None, None)

    else:
        for other_rank, halo_dict in enumerate(
                other_rank_desc_parts[meta.rank]):
            other_rank_desc_parts[other_rank] = {}
    desc_toc = tictoc.get_extoc()

    # Tell the world we've finished progenitor linking
    if meta.verbose:
        tictoc.report("Linking Descendants", desc_tic, desc_toc)

    # Send back the progenitors and descendants we have found
    comm_tic = tictoc.get_extic()
    for other_rank in range(meta.nranks):
        other_rank_prog_parts[other_rank] = comm.gather(
            other_rank_prog_parts[other_rank],
            root=other_rank)
        other_rank_desc_parts[other_rank] = comm.gather(
            other_rank_desc_parts[other_rank],
            root=other_rank)
    comm_toc = tictoc.get_extoc()

    if meta.verbose:
        tictoc.report("Communicating links", comm_tic, comm_toc)

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
def clean_halos(tictoc, meta, halo_cells):

    # Loop over halos and clean up progenitors and descendants
    results = {}
    for halos in halo_cells.values():
        for halo in halos:
            halo.clean_progs(meta)
            halo.clean_descs(meta)
            results[halo.halo_id] = halo
            results[halo.halo_id].clean_halo()

    return results
