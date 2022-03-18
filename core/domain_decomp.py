import numpy as np

import core.utilities as utils 
from core.partition import pick_vector, split_vector, get_parts_in_cell
from core.talking_utils import message
from core.timing import timer


def get_cell_rank(cell_ranks, meta, i, j, k):
    # Get cell index
    ind = utils.get_cellid(meta.cdim, i, j, k)

    return cell_ranks[ind], ind


@timer("Domain-Decomp")
def cell_domain_decomp(tictoc, meta, comm, part_types=(1,)):
    npart = np.sum([meta.npart[i] for i in part_types])

    # Split the cells over the ranks
    samplecells = pick_vector(meta.nranks, meta.cdim)
    cell_ranks, cell_rank_dict = split_vector(meta.cdim, samplecells)

    # Find the cell each particle belongs to
    cells, npart_on_rank = get_parts_in_cell(npart, meta)

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
    my_particles = np.sort(list(my_particles))
    my_tree_particles = np.sort(list(my_tree_particles))

    return my_particles, my_tree_particles, cell_ranks
