import numpy as np

from core.domain_decomp import get_cell_rank
from core.timing import timer
from core.talking_utils import count_and_report_halos


def add_halo(ihalo, parts, part_haloids, halo_pid_dict, weights, weight, 
             qtime_dict, q_time):

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
        weights[ihalo] = weight
        qtime_dict[ihalo] = q_time

        # Increment the halo counter
        ihalo += 1

    elif haloids.size == 1:

        # This halo already exists, get the minimum ID associated
        # to these particles
        halo_id = haloids[0]

        # Include these particles in the halo
        halo_pid_dict[halo_id].update(parts)
        part_haloids[parts] = halo_id
        weights[halo_id] += weight
        qtime_dict[halo_id] = np.mean((q_time, qtime_dict[halo_id]))

    else:

        # This halo already exists, get the minimum ID associated
        # to these particles
        halo_id = np.min(haloids)

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

    return ihalo, part_haloids, halo_pid_dict, weights, qtime_dict


@timer("Stitching")
def combine_across_ranks(tictoc, meta, cell_ranks, pos, halo_pids,
                         rank_tree_parts, comm, weights, qtime_dict):

    # Define cell width
    cell_width = meta.boxsize / meta.cdim

    # Define a dictionary to hold the halos we need to communicate on each rank
    rank_halos = {k: [] for k in range(meta.nranks)}

    # Store time taking on each query for weighting
    rank_qtime = {k: [] for k in range(meta.nranks)}

    # Define dictionary storing the weighting information
    rank_weights = {k: [] for k in range(meta.nranks)}

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
            rank_qtime[i].append(qtime_dict[ihalo])
            rank_weights[i].append(weights[ihalo])

    # Lets communicate the halos we need to send and receive
    for other_rank in range(meta.nranks):
        rank_halos[other_rank] = comm.gather(rank_halos[other_rank],
                                             root=other_rank)
        rank_qtime[other_rank] = comm.gather(rank_qtime[other_rank],
                                             root=other_rank)
        rank_weights[other_rank] = comm.gather(rank_weights[other_rank],
                                             root=other_rank)
        if rank_halos[other_rank] is None:
            rank_halos[other_rank] = set()
            rank_qtime[other_rank] = set()
            rank_weights[other_rank] = set()

    # Define array to store particle halo ids
    rank_spatial_part_haloids = np.full(meta.npart[1], -2, dtype=int)

    # Loop over the halos we have received and stitch them together
    ihalo = 0
    combined_halo_pids = {}
    combined_weights = {}
    combined_qtime = {}
    for other_rank_halos, rweights, rqtimes in zip(rank_halos[meta.rank],
                                                   rank_weights[meta.rank],
                                                   rank_qtime[meta.rank]):
        for parts, weight, qtime in zip(other_rank_halos, rweights, rqtimes):

            # Add this halo, stitching it if necessary
            (ihalo, rank_spatial_part_haloids,
             combined_halo_pids, combined_weights,
             combined_qtime) = add_halo(ihalo, parts,
                                        rank_spatial_part_haloids,
                                        combined_halo_pids,
                                        combined_weights,
                                        weight,
                                        combined_qtime,
                                        qtime)

    # Communicate the halos combined on every rank to master
    combined_halos = comm.gather(combined_halo_pids)
    combined_weights = comm.gather(combined_weights)
    combined_qtime = comm.gather(combined_qtime)

    # Now we just need to remove duplicated halos getting the final
    # spatial halo catalogue
    if meta.rank == 0:

        # Define array to store particle halo ids
        spatial_part_haloids = np.full(meta.npart[1], -2, dtype=int)

        # Set up dictionary to store halo tasks
        ihalo = 0
        halo_tasks = {}
        master_weights = {}
        master_qtime = {}
        # Loop over what we have collected from all ranks
        for halos, weights, qtimes in zip(combined_halos, combined_weights,
                                          combined_qtime):
            for parts, weight, qtime in zip(halos.values(), weights, qtimes):

                # Add this halo, stitching it if necessary
                (ihalo, spatial_part_haloids,
                 halo_tasks, master_weights,
                 master_qtime) = add_halo(ihalo, parts,
                                            spatial_part_haloids,
                                            halo_tasks,
                                            master_weights,
                                            weight,
                                            master_qtime,
                                            qtime)

        # Remove halos which fall below the minimum number of
        # particles threshold
        halo_keys = list(halo_tasks.keys())
        for ihalo in halo_keys:

            if len(halo_tasks[ihalo]) < meta.part_thresh:
                parts = list(halo_tasks[ihalo])
                spatial_part_haloids[parts] = -2

                # Delete dictionary entries
                del halo_tasks[ihalo]
                del master_weights[ihalo]
                del master_qtime[ihalo]

        count_and_report_halos(spatial_part_haloids, meta,
                                         halo_type="Spatial Host Halos")

        # Complete the weighting by adding the query time to each weight
        for ihalo in halo_tasks:
            master_weights[ihalo] += master_qtime[ihalo]

    else:
        halo_tasks = {}
        master_weights = {}

    return halo_tasks, master_weights

