import numpy as np

from domain_decomp import get_cell_rank
from timing import timer
import utilities


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


@timer("Stitching")
def combine_across_ranks(tictoc, meta, cell_ranks, pos, halo_pids,
                         rank_tree_parts, comm):

    # Define cell width
    cell_width = meta.boxsize / meta.cdim

    # Define a dictionary to hold the halos we need to communicate on each rank
    rank_halos = {k: [] for k in range(meta.nranks)}

    # Loop over halos we found
    for ihalo in halo_pids:

        # Define a set to hold the ranks we need to send this halo to
        halo_ranks = set()

        # Extract the particles
        parts = list(halo_pids[ihalo])

        # Get these positions
        this_pos = pos[parts, :]

        # Loop over these particles and get their ranks
        for xyz in this_pos:

            i, j, k = (int(xyz[0] / cell_width),
                       int(xyz[1] / cell_width),
                       int(xyz[2] / cell_width))

            # Get the rank for this particle
            other_rank, _ = get_cell_rank(cell_ranks, meta, i, j, k)
            halo_ranks.update({other_rank, })

            # If we have found all other ranks we don't need to
            # search any further
            if len(halo_ranks) == meta.nranks:
                break

        # Loop over ranks we need to send this halo to
        for i in halo_ranks:
            rank_halos[i].append(rank_tree_parts[parts])

    # Lets communicate the halos we need to send and receive
    for other_rank in range(meta.nranks):
        rank_halos[other_rank] = comm.gather(rank_halos[other_rank],
                                             root=other_rank)
        if rank_halos[other_rank] is None:
            rank_halos[other_rank] = set()

    # Define array to store particle halo ids
    rank_spatial_part_haloids = np.full(meta.npart[1], -2, dtype=int)

    # Loop over the halos we have received and stitch them together
    ihalo = 0
    combined_halo_pids = {}
    for other_rank_halos in rank_halos[meta.rank]:
        for parts in other_rank_halos:

            # Add this halo, stitching it if necessary
            res_tup = add_halo(ihalo, parts, rank_spatial_part_haloids,
                               combined_halo_pids)
            ihalo, rank_spatial_part_haloids, combined_halo_pids = res_tup

    # Communicate the halos combined on every rank to master
    combined_halos = comm.gather(combined_halo_pids)

    # Now we just need to remove duplicated halos getting the final
    # spatial halo catalogue
    if meta.rank == 0:

        # Define array to store particle halo ids
        spatial_part_haloids = np.full(meta.npart[1], -2, dtype=int)

        # Set up dictionary to store halo tasks
        ihalo = 0
        halo_tasks = {}
        # Loop over what we have collected from all ranks
        for halos in combined_halos:
            for parts in halos.values():

                if len(parts) < 10:
                    continue

                # Add this halo, stitching it if necessary
                res_tup = add_halo(ihalo, parts, spatial_part_haloids,
                                   halo_tasks)
                ihalo, spatial_part_haloids, halo_tasks = res_tup

        utilities.count_and_report_halos(spatial_part_haloids, meta,
                                         halo_type="Spatial Host Halos")

    else:
        halo_tasks = {}

    return halo_tasks

