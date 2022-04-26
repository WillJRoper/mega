import numpy as np

from mega.core.talking_utils import count_and_report_halos, message
from mega.core.timing import timer
from mega.halo_core.hydro import link_halo_species


def add_halo(ihalo, parts, part_haloids, halo_pid_dict):

    # Convert particles to a list
    parts = list(parts)

    # Get the halo ids associated to these particles
    haloids = np.unique(part_haloids[parts])
    haloids = haloids[haloids >= 0]

    # Add this halo to the dictionary
    if haloids.size == 0:

        # We have a new halo, it's ID is the halo counter
        halo_pid_dict.setdefault(ihalo, set()).update(parts)
        part_haloids[parts] = ihalo

        # Increment the halo counter
        ihalo += 1

    elif haloids.size == 1:

        # This halo already exists, get the minimum ID associated
        # to these particles
        halo_id = haloids[0]

        # Include these particles in the halo
        halo_pid_dict[halo_id].update(parts)
        part_haloids[parts] = halo_id

    else:

        # This halo already exists, get the minimum ID associated
        # to these particles
        halo_id = np.min(haloids)

        # Include these particles in the halo
        halo_pid_dict[halo_id].update(parts)
        part_haloids[parts] = halo_id

        # Loop over the halo ids returned and delete redundant halos
        for other_halo in haloids:

            # Skip the halo we are stitching on to
            if other_halo == halo_id:
                continue

            # Get the particles from other_halo
            other_parts = list(halo_pid_dict[other_halo])

            # Take the particles from other_halo and assign them to halo_id,
            # these halos are linked
            halo_pid_dict[halo_id].update(other_parts)
            part_haloids[other_parts] = halo_id

            # Delete the now redundant other_halo
            del halo_pid_dict[other_halo]

    return ihalo, part_haloids, halo_pid_dict


def add_halo_multitype(ihalo, dm_ihalo, parts, part_haloids, halo_pid_dict,
                       weights, weight,
                       qtime_dict, q_time, part_type):
    # Convert particles to a list
    parts = list(parts)

    # The machinery is different for DM and other particle types
    # Namely we only increment a halo counter for DM, otherwise the
    # DM halo id is used. DM is done first by definition of meta.part_types:
    # meta.part_types = [1, ] + [i for i in range(len(self.npart))
    #                            if self.npart[i] != 0 and i != 1]

    # Get the halo ids associated to these particles
    haloids = np.unique(part_haloids[parts])
    haloids = haloids[haloids >= 0]

    if part_type == 1:

        # Add this halo to the dictionary
        if haloids.size == 0:

            # We have a new halo, it's ID is the halo counter
            halo_pid_dict.setdefault(ihalo, set()).update(parts)
            part_haloids[parts] = ihalo
            weights[ihalo] = weight
            qtime_dict[ihalo] = q_time

            # Increment the halo counter
            dm_ihalo = ihalo
            ihalo += 1

        elif haloids.size == 1:

            # This halo already exists, get the minimum ID associated
            # to these particles
            halo_id = haloids[0]
            dm_ihalo = halo_id

            # Include these particles in the halo
            halo_pid_dict[halo_id].update(parts)
            part_haloids[parts] = halo_id
            weights[halo_id] += weight
            qtime_dict[halo_id] = np.mean((q_time, qtime_dict[halo_id]))

        else:

            # This halo already exists, get the minimum ID associated
            # to these particles
            halo_id = np.min(haloids)
            dm_ihalo = halo_id

            # Include these particles in the halo
            halo_pid_dict[halo_id].update(parts)
            part_haloids[parts] = halo_id
            weights[halo_id] += weight
            qtime_dict[halo_id] = np.mean((q_time, qtime_dict[halo_id]))

            # Loop over the halo ids returned and delete redundant halos
            for other_halo in haloids:

                # Skip the halo we are stitching on to
                if other_halo == halo_id:
                    continue

                # Get the particles from other_halo
                other_parts = list(halo_pid_dict[other_halo])

                # Take the particles from other_halo and assign them to halo_id,
                # these halos are linked
                halo_pid_dict[halo_id].update(other_parts)
                part_haloids[other_parts] = halo_id
                weights[halo_id] += weights[other_halo]
                qtime_dict[halo_id] = np.mean((qtime_dict[other_halo],
                                               qtime_dict[halo_id]))

                # Delete the now redundant other_halo
                del halo_pid_dict[other_halo]
                del weights[other_halo]
                del qtime_dict[other_halo]

    else:

        # If there is 1 or less haloids we can assign them to the DM halo
        if haloids.size < 2:

            # Assign these non-DM particles
            halo_pid_dict[dm_ihalo].update(parts)
            part_haloids[parts] = dm_ihalo

        # We have multiple halos so need to link them
        else:

            # Get the minimum ID associated to these particles
            halo_id = np.min(haloids)

            # Overwrite the current dm halo id, it has changed
            dm_ihalo = halo_id

            # Include these particles in the halo
            halo_pid_dict[halo_id].update(parts)
            part_haloids[parts] = halo_id
            weights[halo_id] += weight
            qtime_dict[halo_id] = np.mean((q_time, qtime_dict[halo_id]))

            # Loop over the halo ids returned and delete redundant halos
            for other_halo in haloids:

                # Skip the halo we are stitching on to
                if other_halo == halo_id:
                    continue

                # Get the particles from other_halo
                other_parts = list(halo_pid_dict[other_halo])

                # Take the particles from other_halo and assign them to
                # halo_id, these halos are linked
                halo_pid_dict[halo_id].update(other_parts)
                part_haloids[other_parts] = halo_id
                weights[halo_id] += weights[other_halo]
                qtime_dict[halo_id] = np.mean((qtime_dict[other_halo],
                                               qtime_dict[halo_id]))

                # Delete the now redundant other_halo
                del halo_pid_dict[other_halo]
                del weights[other_halo]
                del qtime_dict[other_halo]

    return ihalo, dm_ihalo, part_haloids, halo_pid_dict, weights, qtime_dict


@timer("Stitching")
def combine_across_ranks(tictoc, meta, halo_pinds, rank_tree_parts, npart,
                         comm, rank_tree_bary_parts=None):
    """

    :param tictoc:
    :param meta:
    :param halo_pinds:
    :param rank_tree_parts:
    :param npart:
    :param comm:
    :param rank_tree_bary_parts:
    :return:
    """

    # Define baryon offset from rank linking
    bary_offset = rank_tree_parts.size

    # Shift particle IDs to their true index
    if meta.dmo:
        shifted_halo_pinds = {}
        ihalo = 0
        for old_ihalo in halo_pinds:
            parts = {rank_tree_parts[i]
                     for i in halo_pinds[old_ihalo]}
            shifted_halo_pinds[ihalo] = parts
            ihalo += 1
    else:
        shifted_halo_pinds = {}
        ihalo = 0
        for old_ihalo in halo_pinds:
            parts = set()
            for part in halo_pinds[old_ihalo]:
                if part < bary_offset:
                    parts.update(
                        {rank_tree_parts[part] + meta.part_ind_offset[1]}
                    )
                else:
                    parts.update(
                        {rank_tree_bary_parts[part - bary_offset]
                         + meta.part_ind_offset[0]}
                    )
            shifted_halo_pinds[ihalo] = parts
            ihalo += 1

    # Lets send these halos to master for combining
    combined_halos = comm.gather(shifted_halo_pinds)

    # Now we just need to remove duplicated halos getting the final
    # spatial halo catalogue
    if meta.rank == 0:

        # Define array to store particle halo ids
        spatial_part_haloids = np.full(npart, -2, dtype=int)

        # Set up dictionary to store halo tasks
        ihalo = 0
        halo_tasks = {}

        # Loop over what we have collected from all ranks
        for halos in combined_halos:
            for parts in halos.values():
                # Add this halo, stitching it if necessary
                (ihalo, spatial_part_haloids,
                 halo_tasks) = add_halo(ihalo, parts,
                                        spatial_part_haloids,
                                        halo_tasks)

        # Remove halos which fall below the minimum number of
        # particles threshold
        halo_keys = list(halo_tasks.keys())
        for ihalo in halo_keys:

            if len(halo_tasks[ihalo]) < meta.part_thresh:
                parts = list(halo_tasks[ihalo])
                spatial_part_haloids[parts] = -2

                # Delete dictionary entries
                del halo_tasks[ihalo]

        if meta.verbose:
            count_and_report_halos(spatial_part_haloids, meta,
                                   halo_type="Spatial Host Halos")

        if meta.debug:
            uni, counts = np.unique(spatial_part_haloids, return_counts=True)
            message(meta.rank,
                    "Have %d particles in halos according to halo id array"
                    % np.sum(counts[uni >= 0]))
            message(meta.rank,
                    "Have %d particles in halos according to "
                    "particle ind dictionary"
                    % np.sum([len(halo_tasks[i]) for i in halo_tasks]))

    else:
        halo_tasks = {}

    return halo_tasks


@timer("Stitching")
def combine_halo_types(tictoc, meta, halo_pinds, rank_tree_parts, pos,
                       bary_halo_pinds, rank_tree_bary_parts, bary_pos,
                       tree, bary_tree):
    """

    :param tictoc:
    :param meta:
    :param halo_pinds:
    :param rank_tree_parts:
    :param pos:
    :param bary_halo_pinds:
    :param rank_tree_bary_parts:
    :param bary_pos:
    :param tree:
    :param bary_tree:
    :return:
    """

    # Lets cross reference between dark matter and baryon halos
    bary_in_dm_halos = link_halo_species(tictoc, meta, halo_pinds, bary_tree,
                                         pos, meta.linkl[1])
    dm_in_bary_halos = link_halo_species(tictoc, meta, bary_halo_pinds, tree,
                                         bary_pos, meta.linkl[0])

    # Define the offset on this rank for particle types
    offset = rank_tree_parts.size

    # Combine these halos into a single dictionary and shift
    # particle indices to include their species offset
    shifted_halo_pinds = {}
    ihalo = 0
    for old_ihalo in halo_pinds:
        parts = halo_pinds[old_ihalo]
        parts.update({i + offset
                      for i in bary_in_dm_halos[old_ihalo]})
        shifted_halo_pinds[ihalo] = parts
        ihalo += 1
    for old_ihalo in bary_halo_pinds:
        parts = {i + offset
                 for i in bary_halo_pinds[old_ihalo]}
        parts.update(dm_in_bary_halos[old_ihalo])
        shifted_halo_pinds[ihalo] = parts
        ihalo += 1

    # Define array to store particle halo ids
    nparts_alltypes = rank_tree_parts.size + rank_tree_bary_parts.size
    rank_part_haloids = np.full(nparts_alltypes, -2, dtype=int)

    # Set up dictionary to store combined halos
    ihalo = 0
    halos = {}

    # Loop over the combined halos
    for parts in shifted_halo_pinds.values():

        # Add this halo, stitching it if necessary
        ihalo, rank_part_haloids, halos = add_halo(ihalo, parts,
                                                   rank_part_haloids,
                                                   halos)

    if meta.debug:

        # Report the number of halos found on this rank
        count_and_report_halos(rank_part_haloids, meta,
                               halo_type="Rank %d Combined Spatial Host Halos"
                                         % meta.rank)

    return halos
