import numpy as np

from mega.core.talking_utils import count_and_report_halos
from mega.core.timing import timer


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
def combine_across_ranks(tictoc, meta, halo_pids,
                         rank_tree_parts, comm, weights, qtime_dict):

    # Shift particle IDs to their true index
    all_halos_pids = {}
    ihalo = 0
    for part_type in halo_pids:
        for old_ihalo in halo_pids[part_type]:
            parts = {rank_tree_parts[part_type][i]
                     for i in halo_pids[part_type][old_ihalo]}
            all_halos_pids[ihalo] = parts
            ihalo += 1

    # Lets send these halos to master for combining
    combined_halos = comm.gather(all_halos_pids)
    combined_weights = comm.gather(weights)
    combined_qtime = comm.gather(qtime_dict)

    # TODO: Work out best way to do other particle types when we have
    #  decided on implementation

    # Now we just need to remove duplicated halos getting the final
    # spatial halo catalogue
    if meta.rank == 0:

        # Define array to store particle halo ids
        spatial_part_haloids = np.full(np.sum(meta.npart), -2, dtype=int)

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
            # master_weights[ihalo] += master_qtime[ihalo]
            master_weights[ihalo] = len(halo_tasks[ihalo])

    else:
        halo_tasks = {}
        master_weights = {}

    return halo_tasks, master_weights
