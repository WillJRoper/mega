import numpy as np
import h5py

import mega.core.utilities as utils
from mega.core.partition import stripe_cells, get_parts_in_cell
from mega.core.partition import get_halo_in_cell, initial_partition
from mega.core.partition import get_cell_from_pos
from mega.core.talking_utils import message
from mega.core.timing import timer
import mega.core.serial_io as io

from mpi4py import MPI


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

    # Stripe the cells over the ranks
    if cell_ranks is None:
        cell_ranks, cell_rank_dict = stripe_cells(meta)

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
def halo_cell_domain_decomp(tictoc, meta, comm, nhalo, nprog, ndesc,
                            density_rank):
    """

    :param tictoc:
    :param meta:
    :param comm:
    :return:
    """

    # If we're doing a zoom lets find the bounds
    if meta.zoom:
        meta = find_zoom_region(tictoc, meta, density_rank,
                                nhalo, nprog, ndesc, comm)

    # Get the cell structure
    cdim = meta.cdim

    # Find the cell each halo belongs to
    cells, nhalo_on_rank = get_halo_in_cell(nhalo, meta, density_rank,
                                            meta.snap)

    if not meta.isfirst:
        prog_cells, nprog_on_rank = get_halo_in_cell(nprog, meta, density_rank,
                                                     meta.prog_snap)
    else:
        prog_cells, nprog_on_rank = None, None
    if not meta.isfinal:
        desc_cells, ndesc_on_rank = get_halo_in_cell(ndesc, meta, density_rank,
                                                     meta.desc_snap)
    else:
        desc_cells, ndesc_on_rank = None, None

    # Stripe the cells over the ranks
    cell_ranks, cell_rank_dict = stripe_cells(meta)

    # Label non-empty cells

    # Ensure we haven't lost any halos (lets assume progenitors
    # and descendants are fine if halos are fine)
    if meta.debug:

        collected_cells = comm.gather(cells, root=0)

        if meta.rank == 0:

            count_halos = 0
            for cs in collected_cells:
                for c in cs:
                    count_halos += len(cs[c])

            assert count_halos == nhalo, \
                "Found an incompatible number of particles " \
                "in cells (found=%d, npart=%d)" % (count_halos,
                                                   nhalo)

    # Sort the cells and particles that I have
    # NOTE: we have to distinguish between particles in adjacent cells
    # and particles in my cells since we only need to query halos in my cells
    rank_halos = {r: set() for r in range(meta.nranks)}
    rank_progs = {r: set() for r in range(meta.nranks)}
    rank_descs = {r: set() for r in range(meta.nranks)}
    cells_done = set()
    for i in range(cdim):
        for j in range(cdim):
            for k in range(cdim):

                # Define cell coordinate tuple
                c_ijk = (i, j, k)

                # Get the rank for this cell
                other_rank, ind = get_cell_rank(cell_ranks, meta, i, j, k)

                # Include this cells current information if we have it
                if c_ijk in cells:

                    # Get the halos and store them in the corresponding
                    # place
                    cells_done.update({ind})
                    rank_halos[other_rank].update(set(cells[c_ijk]))

                # Loop over the adjacent cells
                for ii in range(-1, 2, 1):
                    iii = i + ii
                    if not meta.periodic and (iii < 0 or iii > meta.cdim):
                        continue
                    iii %= meta.cdim
                    for jj in range(-1, 2, 1):
                        jjj = j + jj
                        if not meta.periodic and (jjj < 0 or jjj > meta.cdim):
                            continue
                        jjj %= meta.cdim
                        for kk in range(-1, 2, 1):
                            kkk = k + kk
                            if not meta.periodic and (kkk < 0 or kkk > meta.cdim):
                                continue
                            kkk %= meta.cdim

                            # Define ijk tuple
                            adj_ijk = (iii, jjj, kkk)
                            ind = utils.get_cellid(meta.cdim, iii, jjj, kkk)

                            # Get the particles for the adjacent cells
                            if adj_ijk in prog_cells:
                                cells_done.update({ind})
                                if prog_cells is not None:
                                    rank_progs[other_rank].update(
                                        set(prog_cells[adj_ijk])
                                    )
                            if adj_ijk in desc_cells:
                                cells_done.update({ind})
                                if desc_cells is not None:
                                    rank_descs[other_rank].update(
                                        set(desc_cells[adj_ijk])
                                    )

    # Ensure we have got all halos allocated and we have done all cells
    if meta.debug:

        # All cells have been included
        assert len(cells_done) == len(set(cells.keys())), \
            "We have missed cells in the domain decompistion! " \
            "(found=%d, total=%d" % (len(cells_done),
                                     len(set(cells.keys())))
        found_halos_on_rank = set()
        for key in rank_halos:
            found_halos_on_rank.update(rank_halos[key])
        assert len(found_halos_on_rank) == nhalo_on_rank, \
            "Particles missing on rank %d (found=%d, " \
            "npart_on_rank=%d)" % (meta.rank, len(found_halos_on_rank),
                                   nhalo_on_rank)

    # We now need to exchange the particle indices
    for other_rank in range(meta.nranks):
        rank_halos[other_rank] = comm.gather(rank_halos[other_rank],
                                             root=other_rank)
        if not meta.isfirst:
            rank_progs[other_rank] = comm.gather(rank_progs[other_rank],
                                                 root=other_rank)
        if not meta.isfinal:
            rank_descs[other_rank] = comm.gather(rank_descs[other_rank],
                                                 root=other_rank)
        if rank_halos[other_rank] is None:
            rank_halos[other_rank] = set()
            rank_progs[other_rank] = set()
            rank_descs[other_rank] = set()

    my_halos = set()
    my_progs = set()
    my_descs = set()
    for s in rank_halos[meta.rank]:
        my_halos.update(s)
    if not meta.isfirst:
        for s in rank_progs[meta.rank]:
            my_progs.update(s)
    if not meta.isfinal:
        for s in rank_descs[meta.rank]:
            my_descs.update(s)

    comm.Barrier()

    message(meta.rank,
            "I have %d halos, %d progenitors, and %d descendants"
            % (len(my_halos), len(my_progs), len(my_descs)))

    if meta.debug:

        all_halos = comm.gather(my_halos, root=0)
        if meta.rank == 0:
            found_halos = len({i for s in all_halos for i in s})
            assert found_halos == nhalo, \
                "There are particles missing on rank %d " \
                "after exchange! (found=%d, npart=%d)" % (meta.rank,
                                                          found_halos,
                                                          nhalo)

    # Convert to lists and sort so the particles can index the hdf5 files
    my_halos = np.sort(np.array(list(my_halos), dtype=int))
    if not meta.isfirst:
        my_progs = np.sort(np.array(list(my_progs), dtype=int))
    if not meta.isfinal:
        my_descs = np.sort(np.array(list(my_descs), dtype=int))

    return my_halos, my_progs, my_descs, meta


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
def construct_cells(tictoc, halos, progs, descs, meta):

    # Define cell width
    l = np.max(meta.boxsize)
    cell_width = l / meta.cdim

    # Initialise cell dictionaries
    halo_cells = {(i, j, k): []
                  for i in range(meta.cdim)
                  for j in range(meta.cdim)
                  for k in range(meta.cdim)}
    prog_cells = {(i, j, k): []
                  for i in range(meta.cdim)
                  for j in range(meta.cdim)
                  for k in range(meta.cdim)}
    desc_cells = {(i, j, k): []
                  for i in range(meta.cdim)
                  for j in range(meta.cdim)
                  for k in range(meta.cdim)}

    # Loop over halos and store them in their cell
    for halo in halos:

        # Get position
        xyz = halo.mean_pos

        # Get cell indices
        i, j, k = get_cell_from_pos(xyz, cell_width, meta)

        # Store the result for this particle in the dictionary
        halo_cells[(i, j, k)].append(halo)

    # Loop over progenitors and store them in their cell
    for prog in progs:

        # Get position
        xyz = prog.mean_pos

        # Get cell indices
        i, j, k = get_cell_from_pos(xyz, cell_width, meta)

        # Store the result for this particle in the dictionary
        prog_cells[(i, j, k)].append(prog)

    # Loop over descendants and store them in their cell
    for desc in descs:

        # Get position
        xyz = desc.mean_pos

        # Get cell indices
        i, j, k = get_cell_from_pos(xyz, cell_width, meta)

        # Store the result for this particle in the dictionary
        desc_cells[(i, j, k)].append(desc)

    return halo_cells, prog_cells, desc_cells


def find_zoom_region(tictoc, meta, density_rank, nhalo, nprog, ndesc, comm):

    # Initialise bounds
    bounds = np.array([np.inf, 0, np.inf, 0, np.inf, 0], dtype=np.float64)

    # Loop over snapshots
    for snap, n in zip([meta.prog_snap, meta.snap, meta.desc_snap],
                       [nprog, nhalo, ndesc]):

        if snap is None:
            continue

        # Get the initial domain decomp
        low_lim, high_lim = initial_partition(n, meta.nranks, meta.rank)

        # Get the halo positions in the current snapshot
        pos = io.read_halo_range(meta.tictoc, meta,
                                 key="mean_positions",
                                 low=low_lim, high=high_lim,
                                 density_rank=density_rank, snap=snap)

        # Loop over particles and place them in their cell
        for pind in range(pos.shape[0]):

            # Get position
            xyz = pos[pind, :]

            # Update bounds
            for ijk in range(3):
                if bounds[ijk * 2] > xyz[ijk]:
                    bounds[ijk * 2] = xyz[ijk]
                if bounds[(ijk * 2) + 1] < xyz[ijk]:
                    bounds[(ijk * 2) + 1] = xyz[ijk]

    # Communicate what we have found
    min_buffer = bounds.copy()
    max_buffer = bounds.copy()
    comm.Allreduce(MPI.IN_PLACE, min_buffer, op=MPI.MIN)
    comm.Allreduce(MPI.IN_PLACE, max_buffer, op=MPI.MAX)

    # Reconsturct communicated bounds
    for ijk in range(3):
        bounds[ijk * 2] = min_buffer[ijk * 2]
        bounds[(ijk * 2) + 1] = max_buffer[(ijk * 2) + 1]

    # Get the dimension of the high resolution region
    dim = np.array([bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4]],
                   dtype=float)

    # Work out the maximum extent
    max_dim = np.max(dim)

    # Let add some padding for safetys sake (Yes, hardcoded magic number!)
    padded_dim = max_dim * 1.2

    # Find the mid point of the high resolution region
    mid_point = np.array([bounds[0] + (dim[0] / 2),
                          bounds[2] + (dim[0] / 2),
                          bounds[4] + (dim[0] / 2)],
                         dtype=float)

    # Update the bounds
    for ijk in range(3):
        bounds[ijk * 2] = mid_point[ijk] - (padded_dim / 2)
        bounds[(ijk * 2) + 1] = mid_point[ijk] + (padded_dim / 2)

    # Update the bounds of the cell structure
    meta.bounds = bounds

    return meta
