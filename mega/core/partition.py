import numpy as np

from mega.core.serial_io import read_range, read_halo_range
import mega.core.utilities as utils


def initial_partition(npart, nranks, rank):

    # Get limits in terms of indices for particles on each rank
    rank_bins = np.linspace(0, npart, nranks + 1, dtype=int)

    return rank_bins[rank], rank_bins[rank + 1]


def get_cell_from_pos(xyz, cell_width, meta):

    # Get cell indices
    i, j, k = (int((xyz[0] - meta.bounds[0]) / cell_width),
               int((xyz[1] - meta.bounds[2]) / cell_width),
               int((xyz[2] - meta.bounds[4]) / cell_width))

    return i, j, k


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
        i, j, k = get_cell_from_pos(xyz, cell_width, meta)

        # Store the result for this particle in the dictionary
        cells.setdefault((i, j, k), []).append(pind + low_lim)

    return cells, high_lim - low_lim


def get_halo_in_cell(nhalo, meta, density_rank, snap):

    # Get the initial domain decomp
    low_lim, high_lim = initial_partition(nhalo, meta.nranks, meta.rank)

    # Define cell width
    l = np.max(meta.boxsize)
    cell_width = l / meta.cdim

    # Initialise cell dictionary
    cells = {}

    # Get the particles on this rank in the initial domain decomp
    pos = read_halo_range(meta.tictoc, meta,
                          key="mean_positions",
                          low=low_lim, high=high_lim,
                          density_rank=density_rank, snap=snap)

    # Loop over particles and place them in their cell
    for pind in range(pos.shape[0]):

        # Get position
        xyz = pos[pind, :]

        # Get cell indices
        i, j, k = get_cell_from_pos(xyz, cell_width, meta)

        # Store the result for this particle in the dictionary
        cells.setdefault((i, j, k), []).append(pind + low_lim)

    return cells, high_lim - low_lim


def stripe_cells(meta):

    # Get metadata values
    cdim = meta.cdim
    nranks = meta.nranks
    ncells = meta.ncells

    # Define dictionary to hold cells eventual ranks
    cell_rank_dict = {}

    assert nranks < ncells, "There are more ranks than cells!"

    # What cells are where?
    # NOTE: linspace will always put a single integer for the max value so
    #       we include it and subtract 1.
    cell_ranks = np.linspace(0, nranks, ncells, dtype=int)
    cell_ranks[cell_ranks == nranks] -= 1

    # Loop over cells and populate the dictionary
    for i in range(cdim):
        for j in range(cdim):
            for k in range(cdim):

                # Get cell index
                ind = utils.get_cellid(cdim, i, j, k)

                # Set the rank for this cell
                cell_rank_dict.setdefault(cell_ranks[ind],
                                          []).append((i, j, k))

    # Ensure all cells have been partioned
    assert np.min(cell_ranks) >= 0, "Not all cells assigned to a rank"
    assert np.max(
        cell_ranks) == nranks - 1, "Cells sorted into too many ranks: %d but should be %d" % (np.max(cell_ranks), nranks - 1)

    return cell_ranks, cell_rank_dict
