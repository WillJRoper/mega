import numpy as np

from mega.core.serial_io import read_range
import mega.core.utilities as utils


def initial_partition(npart, nranks, rank):

    # Get limits in terms of indices for particles on each rank
    rank_bins = np.linspace(0, npart, nranks + 1, dtype=int)

    return rank_bins[rank], rank_bins[rank + 1]


def get_parts_in_cell(npart, meta, part_type):

    # Get the initial domain decomp
    low_lim, high_lim = initial_partition(npart, meta.nranks, meta.rank)

    # Define cell width
    l = np.max(meta.boxsize)
    cell_width = l / meta.cdim

    # Initialise cell dictionary
    cells = {}

    # Get the particles on this rank in the initial domain decomp
    pos = read_range(meta.tictoc, meta,
                     key="PartType%d/Coordinates" % part_type,
                     low=low_lim, high=high_lim)

    # Loop over particles and place them in their cell
    for pind in range(pos.shape[0]):

        # Get position
        xyz = pos[pind, :]

        # Get cell indices
        i, j, k = (int(xyz[0] / cell_width),
                   int(xyz[1] / cell_width),
                   int(xyz[2] / cell_width))

        # Store the result for this particle in the dictionary
        cells.setdefault((i, j, k), []).append(pind + low_lim)

    return cells, high_lim - low_lim


def stripe_cells(meta):

    # Get metadata values
    cdim = meta.cdim
    nranks = meta.nranks

    # Define dictionary to hold cells eventual ranks
    cell_ranks = np.full(cdim**3, -2, dtype=int)
    cell_rank_dict = {}

    # How many planes does each rank get?
    planes = np.linspace(0, cdim, nranks + 1, dtype=int)

    # Loop over cells
    for (rank, k_low), k_high in zip(enumerate(planes[:-1]), planes[1:]):
        for k in range(k_low, k_high):
            for i in range(cdim):
                for j in range(cdim):

                    # Get cell index
                    ind = utils.get_cellid(cdim, i, j, k)

                    # Set the rank for this cell
                    cell_ranks[ind] = rank
                    cell_rank_dict.setdefault(rank, []).append((i, j, k))

    # Ensure all cells have been partioned
    assert np.min(cell_ranks) == 0, "Not all cells assigned to a rank"

    return cell_ranks, cell_rank_dict
