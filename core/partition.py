import numpy as np
import h5py


def initial_partition(npart, nranks, rank):

    # Get limits in terms of indices for particles on each rank
    rank_bins = np.linspace(0, npart, nranks + 1, dtype=int)

    return rank_bins[rank], rank_bins[rank + 1]


def get_parts_in_cell(npart, meta):

    # Get the initial domain decomp
    low_lim, high_lim = initial_partition(npart, meta.nranks, meta.rank)

    # Define cell width
    cell_width = meta.boxsize / meta.cdim

    # Initialise cell dictionary
    cells = {}

    # Get the particles on this rank in the initial domain decomp
    hdf = h5py.File(meta.inputpath + meta.snap + ".hdf5", "r")
    if npart == meta.npart[1]:
        pos = hdf["PartType1"]["part_pos"][low_lim: high_lim, :]
    else:
        pos = hdf["All"]["part_pos"][low_lim: high_lim, :]
    hdf.close()

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


def pick_vector(nranks, cdim):

    # Define the number of cells
    ncells = cdim ** 3

    # Set up array to hold the sample cell selections
    samplecells = []

    # Set up the loop variables
    step = ncells / nranks
    l = 0

    # Loop over cells
    ii = 0
    for i in range(cdim):
        for j in range(cdim):
            for k in range(cdim):
                if ii % step == 0 and l < nranks:
                    samplecells.append((i, j, k))
                    l += 1
                ii += 1

    return samplecells


def split_vector(cdim, samplecells):

    # Define dictionary to hold cells eventual ranks
    cell_ranks = np.full(cdim**3, -2, dtype=int)
    cell_rank_dict = {}

    # Loop over cells
    ind = 0
    for i in range(cdim):
        for j in range(cdim):
            for k in range(cdim):
                select = -1
                rsqmax = 10**10
                for l in range(len(samplecells)):
                    dx = samplecells[l][0] - i
                    dy = samplecells[l][1] - j
                    dz = samplecells[l][2] - k
                    rsq = (dx * dx + dy * dy + dz * dz)
                    if rsq < rsqmax:
                        rsqmax = rsq
                        select = l

                # Set the closest seed rank as the rank for this cell
                cell_ranks[ind] = select
                cell_rank_dict.setdefault(select, []).append((i, j, k))
                ind += 1

    # Ensure all cells have been partioned
    assert np.min(cell_ranks) == 0, "Not all cells assigned to a rank"
    assert np.max(cell_ranks) == len(samplecells) - 1, "More ranks than exist"

    return cell_ranks, cell_rank_dict
