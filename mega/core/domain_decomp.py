import numpy as np
import h5py

import mega.core.utilities as utils
from mega.core.partition import pick_vector, split_vector, get_parts_in_cell
from mega.core.talking_utils import message
from mega.core.timing import timer


def get_cell_rank(cell_ranks, meta, i, j, k):
    """

    :param cell_ranks:
    :param meta:
    :param i:
    :param j:
    :param k:
    :return:
    """
    # Get cell index
    ind = utils.get_cellid(meta.cdim, i, j, k)

    return cell_ranks[ind], ind


@timer("Domain-Decomp")
def cell_domain_decomp(tictoc, meta, comm, part_type, cell_ranks=None):
    """

    :param tictoc:
    :param meta:
    :param comm:
    :return:
    """

    # Get the number of particles we are working with
    npart = meta.npart[part_type]

    # Split the cells over the ranks
    if cell_ranks is None:
        samplecells = pick_vector(meta.nranks, meta.cdim)
        cell_ranks, cell_rank_dict = split_vector(meta.cdim, samplecells)

    # Find the cell each particle belongs to
    cells, npart_on_rank = get_parts_in_cell(npart, meta, part_type)

    # Ensure we haven't lost any particles
    if meta.debug:

        collected_cells = comm.gather(cells, root=0)

        if meta.rank == 0:

            count_parts = 0
            for cs in collected_cells:
                for c in cs:
                    count_parts += len(cs[c])

            assert count_parts == npart, \
                "Found an incompatible number of particles " \
                "in cells (found=%d, npart=%d)" % (count_parts,
                                                   npart)

    # Sort the cells and particles that I have
    # NOTE: we have to distinguish between particles in adjacent cells
    # and particles in my cells since we only need to query the
    # tree with the particles in my cells
    rank_parts = {r: set() for r in range(meta.nranks)}
    rank_tree_parts = {r: set() for r in range(meta.nranks)}
    cells_done = set()
    my_cells = list(cells.keys())
    for c_ijk in my_cells:

        # Get ijk coordinates
        i, j, k = c_ijk

        # Get the rank for this cell
        other_rank, ind = get_cell_rank(cell_ranks, meta, i, j, k)

        # Get the particles and store them in the corresponding place
        cells_done.update({ind})
        rank_parts[other_rank].update(set(cells[c_ijk]))

        # Loop over the adjacent cells
        for ii in range(-1, 2, 1):
            iii = i + ii
            iii %= meta.cdim
            for jj in range(-1, 2, 1):
                jjj = j + jj
                jjj %= meta.cdim
                for kk in range(-1, 2, 1):
                    kkk = k + kk
                    kkk %= meta.cdim

                    # Define ijk tuple
                    c_ijk = (iii, jjj, kkk)
                    ind = utils.get_cellid(meta.cdim, iii, jjj, kkk)

                    # Get the particles for the adjacent cells
                    if c_ijk in cells.keys():
                        cells_done.update({ind})
                        rank_tree_parts[other_rank].update(set(cells[c_ijk]))

    # Ensure we have got all particles allocated and we have done all cells
    if meta.debug:

        # All cells have been included
        assert len(cells_done) == len(set(cells.keys())), \
            "We have missed cells in the domain decompistion! " \
            "(found=%d, total=%d" % (len(cells_done),
                                     len(set(cells.keys())))
        found_parts_on_rank = set()
        for key in rank_tree_parts:
            found_parts_on_rank.update(rank_tree_parts[key])
        assert len(found_parts_on_rank) == npart_on_rank, \
            "Particles missing on rank %d (found=%d, " \
            "npart_on_rank=%d)" % (meta.rank, len(found_parts_on_rank),
                                   npart_on_rank)

    # We now need to exchange the particle indices
    for other_rank in range(meta.nranks):
        rank_parts[other_rank] = comm.gather(rank_parts[other_rank],
                                             root=other_rank)
        rank_tree_parts[other_rank] = comm.gather(rank_tree_parts[other_rank],
                                                  root=other_rank)
        if rank_parts[other_rank] is None:
            rank_parts[other_rank] = set()
            rank_tree_parts[other_rank] = set()

    my_particles = set()
    my_tree_particles = set()
    for s in rank_parts[meta.rank]:
        my_particles.update(s)
    for s in rank_tree_parts[meta.rank]:
        my_tree_particles.update(s)

    comm.Barrier()

    if meta.verbose:
        message(meta.rank, "I have %d particles and %d tree particle"
                % (len(my_particles), len(my_tree_particles)))

    if meta.debug:

        all_parts = comm.gather(my_particles, root=0)
        if meta.rank == 0:
            found_parts = len({i for s in all_parts for i in s})
            assert found_parts == npart, \
                "There are particles missing on rank %d " \
                "after exchange! (found=%d, npart=%d)" % (meta.rank,
                                                          found_parts,
                                                          npart)

    # Convert to lists and sort so the particles can index the hdf5 files
    my_particles = np.sort(np.array(list(my_particles), dtype=int))
    my_tree_particles = np.sort(np.array(list(my_tree_particles),
                                         dtype=int))

    return my_particles, my_tree_particles, cell_ranks


@timer("Domain-Decomp")
def hydro_cell_domain_decomp(tictoc, meta, comm, cell_ranks):
    """

    :param tictoc:
    :param meta:
    :param comm:
    :return:
    """

    # Define dictionary to store this ranks particles
    my_parts_dict = {}
    my_tree_parts_dict = {}

    # Loop over particle types present in the simulation
    # NOTE: Boundary particle types 2 and 3 are automatically ignored
    for part_type in meta.part_types:

        # Skip dark matter
        if part_type == 1:
            continue

        res = cell_domain_decomp(tictoc, meta, comm, part_type, cell_ranks)
        my_parts_dict[part_type], my_tree_parts_dict[part_type], _ = res

    return my_parts_dict, my_tree_parts_dict


@timer("Domain-Decomp")
def halo_decomp(tictoc, meta, halo_tasks, comm):
    """

    :param tictoc:
    :param meta:
    :param halo_tasks:
    :param weights:
    :return:
    """

    if meta.rank == 0:

        # Lets sort our halos by decreasing cost
        cost = [len(halo_tasks[key]) for key in halo_tasks]
        sinds = np.argsort(cost)[::-1]
        haloids = np.array(list(halo_tasks.keys()), dtype=int)
        sorted_halos = haloids[sinds]

        # Initialise tasks for each rank
        rank_halos_dict = {r: {} for r in range(meta.nranks)}

        # Initialise counter to track the weight allocated to each rank
        alloc_weights = [0, ] * meta.nranks

        # Allocate tasks, each task goes to the minimally weighted rank
        for ihalo in sorted_halos:

            # Get position of minimum current rank weighting
            r = np.argmin(alloc_weights)

            # Assign this halo and it's weight to r
            rank_halos_dict[r][ihalo] = halo_tasks[ihalo]
            alloc_weights[r] += len(halo_tasks[ihalo])

    else:
        rank_halos_dict = {r: None for r in range(meta.nranks)}

    # Convert dict to list for scattering
    rank_halos = [rank_halos_dict[r] for r in range(meta.nranks)]

    if meta.verbose:
        for r in range(meta.nranks):
            message(meta.rank, "Rank %d has %d halos" % (r,
                                                         len(rank_halos[r])))

    # Give everyone their tasks
    my_tasks = comm.scatter(rank_halos, root=0)

    # Get my task indices and particle index array
    my_halo_parts = []
    start_index = np.zeros(len(my_tasks), dtype=int)
    stride = np.zeros(len(my_tasks), dtype=int)
    current_ind = 0
    itask = 0
    for ihalo in my_tasks:

        # Extract this spatial halos particles
        parts = my_tasks[ihalo]
        npart = len(parts)

        # Store the index pointer and stride for these particles
        start_index[itask] = current_ind
        stride[itask] = npart

        # Increment counters
        current_ind += npart
        itask += 1

        # Store these particles indices
        my_halo_parts.extend(parts)

    # Convert particle index list to an array
    my_halo_parts = np.array(my_halo_parts, copy=False, dtype=int)

    return my_halo_parts, start_index, stride


@timer("Domain-Decomp")
def graph_halo_decomp(tictoc, nhalo, meta, comm, density_rank,
                      rank_pidbins):
    """

    :param tictoc:
    :param meta:
    :param halo_tasks:
    :param weights:
    :return:
    """

    if meta.rank == 0:

        # Open the current snapshot
        hdf = h5py.File(meta.halopath + meta.halo_basename
                        + meta.snap + '.hdf5', 'r')

        # Are we dealing with hosts or subhalos
        if density_rank == 0:
            root = hdf
        else:
            root = hdf["Subhalos"]

        # Initialise tasks for each rank
        rank_halos_dict = {r: {} for r in range(meta.nranks)}

        # Allocate tasks, each halo goes to the rank containing most
        # of it's particles
        for ihalo in range(nhalo):

            # Let's get this halos particles
            begin = root["start_index"][ihalo]
            end = begin + root["stride"][ihalo]
            parts = root["sim_part_ids"][begin: end]
            message(meta.rank, b, e, np.min(parts), np.min(rank_pidbins), np.max(parts), np.max(rank_pidbins))
            # Which rank holds the majority of this halo's particles?
            rs, counts = np.unique(
                np.digitize(parts, rank_pidbins),
                                   return_counts=True)

            # Binning returns the index of the right hand bin edge
            r = rs[np.argmax(counts)] - 1

            # Give that rank this halo
            rank_halos_dict[r][ihalo] = parts

        hdf.close()

    else:
        rank_halos_dict = {r: None for r in range(meta.nranks)}

    # Convert dict to list for scattering
    rank_halos = [rank_halos_dict[r] for r in range(meta.nranks)]

    if meta.verbose:
        for r in range(meta.nranks):
            message(meta.rank, "Rank %d has %d halos" % (r,
                                                         len(rank_halos[r])))

    # Give everyone their tasks
    my_tasks = comm.scatter(rank_halos, root=0)

    return my_tasks



