import h5py
import networkx
import numpy as np
from halo import Halo
# import readgadgetdata
import yaml
from networkx.algorithms.components.connected import connected_components


def hdf5_write_dataset(grp, key, data, compression="gzip"):
    grp.create_dataset(key, shape=data.shape, dtype=data.dtype, data=data,
                       compression=compression)

def read_param(paramfile):
    # Read in the param file
    with open(paramfile) as yfile:
        parsed_yaml_file = yaml.load(yfile, Loader=yaml.FullLoader)

    # Extract individual dictionaries
    inputs = parsed_yaml_file['inputs']
    flags = parsed_yaml_file['flags']
    params = parsed_yaml_file['parameters']
    cosmology = parsed_yaml_file['cosmology']
    simulation = parsed_yaml_file['simulation']

    return inputs, flags, params, cosmology, simulation


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def to_graph(l):
    """ https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements """

    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    """ https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def SWIFT_to_MEGA_hdf5(snapshot, PATH, basename, inputpath='input/'):
    """ Reads in SWIFT format simulation data from standalone HDF5 files
    and writes the necessary fields into the expected MEGA format.
    NOTE: hless units are expected!

    :param snapshot: The snapshot ID as a string (e.g. '061', "00001", etc)
    :param PATH: The filepath to the directory containing the simulation data.
    :param basename: The name of the snapshot files WIHTOUT the snapshot string.
    :param inputpath: The file path for storing the MEGA format
                      input data read from the simulation files.

    :return:

    """

    # =============== Load Simulation Data ===============

    if PATH[-1] != "/":
        PATH = PATH + "/"

    # Load snapshot data from SWIFT hdf5
    hdf = h5py.File(PATH + basename + snapshot + ".hdf5", "r")

    # Get metadata about the simulation
    boxsize = hdf["Header"].attrs["BoxSize"][0]
    redshift = hdf["Header"].attrs["Redshift"]
    npart = hdf["Header"].attrs["NumPart_Total"]
    nparts_this_file = hdf["Header"].attrs["NumPart_ThisFile"][1]

    assert npart[1] == nparts_this_file, "Opening a file split " \
                                      "across multiple files as if " \
                                      "it is a standalone"

    # Define part_types based on how many particle species are
    # recorded in the file
    part_types = [i for i in range(len(npart)) if npart[i] > 0]

    print("Found %d particle species:" % len(part_types))
    print(npart)

    # Open hdf5 file
    hdf_out = h5py.File(inputpath + basename + "mega_inputs_" + snapshot + ".hdf5", 'w')

    # Write out the inputs
    hdf_out.attrs['boxsize'] = boxsize
    hdf_out.attrs['npart'] = np.int64(npart)
    hdf_out.attrs['redshift'] = redshift

    # Create lists to combine arrays
    all = {}
    mean_seps = np.zeros(len(npart))

    for ind, i in enumerate(part_types):

        print("PartType%d" % i)

        # Get the particle data
        pid = hdf["PartType%d" % i]["ParticleIDs"][...]
        pos = hdf["PartType%d" % i]["Coordinates"][...]
        vel = hdf["PartType%d" % i]["Velocities"][...]
        masses = hdf["PartType%d" % i]["Masses"][...]

        # Define part_type array
        part_type = np.full(masses.size, i, dtype=int)

        if len(part_types) > 0:
            # Assign arrays
            all.setdefault('part_pid', []).extend(pid)
            all.setdefault('part_types', []).extend(part_type)
            all.setdefault('part_pos', []).extend(pos)
            all.setdefault('part_vel', []).extend(vel)
            all.setdefault('part_masses', []).extend(masses)

        # =============== Sort particles ===============

        # Sort the simulation data arrays by the particle ID
        sinds = pid.argsort()
        pid = pid[sinds]
        pos = pos[sinds, :]
        vel = vel[sinds, :]
        masses = masses[sinds]
        part_type = part_type[sinds]

        # =============== Write this particle type ===============

        partgrp = hdf_out.create_group("PartType%d" % i)

        # Compute the mean separation
        mean_seps[i] = boxsize / npart[i] ** (1. / 3.)

        # Write out the inputs
        partgrp.attrs['mean_sep'] = mean_seps[i]
        partgrp.attrs['tot_mass'] = np.sum(masses)
        hdf5_write_dataset(partgrp, 'part_pid', pid)
        hdf5_write_dataset(partgrp, 'sort_inds', sinds)
        hdf5_write_dataset(partgrp, 'part_types', part_type)
        hdf5_write_dataset(partgrp, 'part_pos', pos)
        hdf5_write_dataset(partgrp, 'part_vel', vel)
        hdf5_write_dataset(partgrp, 'part_masses', masses)

    hdf.close()

    if len(part_types) > 0:

        # Convert to arrays
        for key in all:
            all[key] = np.array(all[key])

        # =============== Sort particles ===============

        # Sort the simulation data arrays by the particle ID
        sinds = all['part_pid'].argsort()
        pid = all['part_pid'][sinds]
        pos = all['part_pos'][sinds, :]
        vel = all['part_vel'][sinds, :]
        masses = all['part_masses'][sinds]
        part_type = all['part_types'][sinds]

        assert pid.size == np.sum(npart), "Somehow we have more array entries " \
                                          "than particles"

        # ============== Compute and store the necessary values ==============

        allgrp = hdf_out.create_group("All")

        # Write out the inputs
        allgrp.attrs['tot_npart'] = pid.size
        allgrp.attrs['mean_sep'] = mean_seps
        allgrp.attrs['tot_mass'] = np.sum(masses)
        hdf5_write_dataset(allgrp, 'part_pid', pid)
        hdf5_write_dataset(allgrp, 'sort_inds', sinds)
        hdf5_write_dataset(allgrp, 'part_types', part_type)
        hdf5_write_dataset(allgrp, 'part_pos', pos)
        hdf5_write_dataset(allgrp, 'part_vel', vel)
        hdf5_write_dataset(allgrp, 'part_masses', masses)

    hdf_out.close()


def SWIFT_to_MEGA_hdf5_allsnaps(snaplist_path, PATH, basename,
                                inputpath='input/'):
    """ Reads in SWIFT format simulation data from standalone HDF5 files
    and writes the necessary fields into the expected MEGA format.
    NOTE: hless units are expected!

    :param snapshot: The snapshot ID as a string (e.g. '061', "00001", etc)
    :param PATH: The filepath to the directory containing the simulation data.
    :param basename: The name of the snapshot files WIHTOUT the snapshot string.
    :param inputpath: The file path for storing the MEGA format
                      input data read from the simulation files.

    :return:

    """

    # Ensure paths are as expected
    if PATH[-1] != "/":
        PATH = PATH + "/"
    if inputpath[-1] != "/":
        inputpath = inputpath + "/"

    # Load the snapshot list
    snaplist = list(np.loadtxt(snaplist_path, dtype=str))

    # Loop over all snapshots
    for snapshot in snaplist:

        print("Writing snapshot:", snapshot)

        SWIFT_to_MEGA_hdf5(snapshot, PATH, basename, inputpath)


# def GADGET_binary_to_hdf5(snapshot, PATH, inputpath='input/'):
#     """ Reads in GADGET binary format simulation data
#     and writes the necessary fields into the expected MEGA format.
#
#     :param snapshot: The snapshot ID as a string (e.g. '061')
#     :param PATH: The filepath to the directory containing the simulation data.
#     :param inputpath: The file path for storing the MEGA format
#                       input data read from the simulation files.
#
#     :return:
#     """
#
#     # =============== Load Simulation Data ===============
#
#     # Load snapshot data from gadget-2 file *** Note: will need to be
#     # changed for use with other simulations data ***
#     snap = readgadgetdata.readsnapshot(snapshot, PATH)
#     pid, pos, vel = snap[
#                     0:3]  # pid=particle ID, pos=all particle's position,
#                           # vel=all particle's velocity
#     head = snap[3:]  # header values
#     npart = head[0]  # number of particles in simulation
#     boxsize = head[3]  # simulation box length(/size) along each axis
#     redshift = head[1]
#     t = head[2]  # elapsed time of the snapshot
#     rhocrit = head[4]  # Critical density
#     pmass = head[5]  # Particle mass
#     h = head[6]  # 'little h' (hubble parameter parametrisation)
#
#     # =============== Sort particles ===============
#
#     # Sort the simulation data arrays by the particle ID
#     sinds = pid.argsort()
#     pid = pid[sinds]
#     pos = pos[sinds, :]
#     vel = vel[sinds, :]
#
#     # =============== Compute Linking Length ===============
#
#     # Compute the mean separation
#     mean_sep = boxsize / npart ** (1. / 3.)
#
#     # Open hdf5 file
#     hdf = h5py.File(inputpath + "mega_inputs_" + snapshot + ".hdf5", 'w')
#
#     # Write out the inputs
#     hdf.attrs['mean_sep'] = mean_sep
#     hdf.attrs['boxsize'] = boxsize
#     hdf.attrs['npart'] = npart
#     hdf.attrs['redshift'] = redshift
#     hdf.attrs['t'] = t
#     hdf.attrs['rhocrit'] = rhocrit
#     hdf.attrs['pmass'] = pmass
#     hdf.attrs['h'] = h
#     hdf.create_dataset('part_pid', shape=pid.shape, dtype=float, data=pid,
#                        compression="gzip")
#     hdf.create_dataset('sort_inds', shape=sinds.shape, dtype=int, data=sinds,
#                        compression="gzip")
#     hdf.create_dataset('part_pos', shape=pos.shape, dtype=float, data=pos,
#                        compression="gzip")
#     hdf.create_dataset('part_vel', shape=vel.shape, dtype=float, data=vel,
#                        compression="gzip")
#
#     hdf.close()


def wrap_halo(halo_poss, boxsize, domean=False):
    # Define the comparison particle as the maximum
    # position in the current dimension
    max_part_pos = halo_poss.max(axis=0)

    # Compute all the halo particle separations from the maximum position
    sep = max_part_pos - halo_poss

    # If any separations are greater than 50% the boxsize
    # (i.e. the halo is split over the boundary)
    # bring the particles at the lower boundary together
    # with the particles at the upper boundary
    # (ignores halos where constituent particles aren't
    # separated by at least 50% of the boxsize)
    # *** Note: fails if halo's extent is greater than 50%
    # of the boxsize in any dimension ***
    halo_poss[np.where(sep > 0.5 * boxsize)] += boxsize

    if domean:
        # Compute the shifted mean position in the dimension ixyz
        mean_halo_pos = halo_poss.mean(axis=0)

        # Centre the halos about the mean in the dimension ixyz
        halo_poss -= mean_halo_pos

        return halo_poss, mean_halo_pos

    else:

        return halo_poss


def bin_nodes(pos, ncells, minmax, ranks):
    r0, r1 = minmax

    npart = pos.shape[0]

    cell_size = (r1 - r0) / ncells
    cell_volume = cell_size ** 3

    bin_inds = (float(ncells) / (r1 - r0) * (pos - r0)).astype(int)
    bin_edges = np.linspace(r0, r1, ncells + 1)

    nodes = get_initasks(bin_inds)

    initasks = [nodes[key] for key in nodes]
    weights = [len(task) / cell_volume for task in initasks]
    target = np.sum(weights) / ranks
    print(target)

    work = {}
    for i in range(ranks):
        work_weights = 0
        for task, w in zip(initasks, weights):
            work.setdefault(i, []).extend(task)
            work_weights += weights
            if work_weights > target:
                break

    return work


def get_initasks(bin_inds):
    nodes = {}
    for ind, key in enumerate(bin_inds):
        nodes.setdefault(tuple(key), set()).update({ind, })

    return nodes


def decomp_nodes(npart, ranks, cells_per_rank, rank):
    # Define the limits for particles on all ranks
    rank_edges = np.linspace(0, npart, ranks + 1, dtype=int)
    rank_cell_edges = np.linspace(rank_edges[rank], rank_edges[rank + 1],
                                  cells_per_rank + 1, dtype=int)

    # Define the nodes
    tasks = []
    for low, high in zip(rank_cell_edges[:-1], rank_cell_edges[1:]):
        tasks.append(np.arange(low, high, dtype=int))

    nnodes = cells_per_rank * ranks

    # Get the particles in this rank
    parts_in_rank = np.arange(rank_edges[rank], rank_edges[rank + 1],
                              dtype=int)

    return tasks, parts_in_rank, nnodes, rank_edges


def combine_tasks_networkx(results, ranks, halos_to_combine, npart):
    results_to_combine = {frozenset(results.pop(halo)) for halo in
                          halos_to_combine}

    # Convert results to a networkx graph for linking
    G = to_graph(results_to_combine)

    # Create list of unique lists of particles
    combined_results = {frozenset(parts) for parts in connected_components(G)
                        if len(parts) >= 10}
    results = set(results.values())
    results.update(combined_results)

    spatial_part_haloids = np.full(npart, -2, dtype=np.int32)

    # Split into a list containing a list of halos for each rank
    newSpatialID = 0
    tasks = {}
    while len(results) > 0:
        res = results.pop()
        if len(res) >= 10:
            tasks[newSpatialID] = np.array(list(res), dtype=np.int64)
            spatial_part_haloids[list(res)] = newSpatialID
            newSpatialID += 1

    # Find the halos with 10 or more particles by finding the unique IDs in the particle
    # halo ids array and finding those IDs that are assigned to 10 or more particles
    unique, counts = np.unique(spatial_part_haloids, return_counts=True)
    unique_haloids = unique[np.where(counts >= 10)]

    # Remove the null -2 value for single particle halos
    unique_haloids = unique_haloids[np.where(unique_haloids != -2)]

    # Print the number of halos found by the halo finder in >10, >100, >1000, >10000 criteria
    print(
        "=========================== Spatial halos ===========================")
    print(unique_haloids.size, 'halos found with 10 or more particles')
    print(unique[np.where(counts >= 15)].size - 1,
          'halos found with 15 or more particles')
    print(unique[np.where(counts >= 20)].size - 1,
          'halos found with 20 or more particles')
    print(unique[np.where(counts >= 50)].size - 1,
          'halos found with 50 or more particles')
    print(unique[np.where(counts >= 100)].size - 1,
          'halos found with 100 or more particles')
    print(unique[np.where(counts >= 500)].size - 1,
          'halos found with 500 or more particles')
    print(unique[np.where(counts >= 1000)].size - 1,
          'halos found with 1000 or more particles')
    print(unique[np.where(counts >= 10000)].size - 1,
          'halos found with 10000 or more particles')

    return tasks


def decomp_halos(results, nnodes):
    # Initialise halo dictionaries read for the phase space test
    halo_pids = {}
    ini_parts_in_rank = []

    # Store halo ids and halo data for the halos found out in the spatial search
    newtaskID = nnodes + 1
    while len(results) > 0:
        # Extract particle IDs
        parts_arr = np.array(list(results.pop()))

        # Assign the particles to the main dictionary
        halo_pids[(1, newtaskID)] = parts_arr

        # Assign particles
        ini_parts_in_rank.extend(parts_arr)

        newtaskID += 1

    # Convert parts in rank to list for use with numpy
    parts_in_rank = np.sort(ini_parts_in_rank)

    return halo_pids, parts_in_rank, newtaskID


def decomp_subhalos(subhalo_pids_per_rank, ranks):
    # Split into a list containing a list of halos for each rank
    newTaskID = 0
    chunked_part_load = np.zeros(ranks)
    chunked_halo_load = np.zeros(ranks)
    chunked_pids = [{} for i in range(ranks)]

    # Initialise the key to store all particles on a rank
    for i in range(ranks):
        chunked_pids[i]["parts_on_rank"] = set()

    for ind, subhalo_pids_dict in enumerate(subhalo_pids_per_rank):
        while len(subhalo_pids_dict) > 0:
            task = (3, newTaskID)
            _, pids = subhalo_pids_dict.popitem()
            if len(pids) >= 10:
                i = np.argmin(chunked_part_load)
                chunked_part_load[i] += len(pids)
                chunked_halo_load[i] += 1
                chunked_pids[i][task] = pids
                chunked_pids[i]["parts_on_rank"].update(pids)
                newTaskID += 1

    # Convert sets of included particles on a rank to sorted arrays
    for i in range(ranks):
        chunked_pids[i]["parts_on_rank"] = np.sort(
            list(chunked_pids[i]["parts_on_rank"]))

    return chunked_pids


def combine_tasks_per_thread(results, rank, thisRank_parts):
    # Initialise halo dictionaries read for the phase space test
    halo_pids = {}

    results = {parts for d in results.values() for parts in d.values()}

    G = to_graph(results)

    results = list(connected_components(G))

    # Store halo ids and halo data for the halos found out in the spatial search
    newtaskID = 0
    halos_in_other_ranks = set()
    while len(results) > 0:
        parts = results.pop()

        parts_in_other_ranks = parts - thisRank_parts

        if len(parts) < 10:
            if len(parts_in_other_ranks) == 0:
                continue

        if len(parts_in_other_ranks) > 0:
            halos_in_other_ranks.update({(rank, newtaskID)})

        halo_pids[(rank, newtaskID)] = frozenset(parts)
        newtaskID += 1

    return halo_pids, halos_in_other_ranks


def read_halo_data(part_inds, inputpath, snapshot, hdf_part_key,
                   ini_vlcoeff, boxsize, soft, redshift, G, cosmo):

    # Open hdf5 file
    hdf = h5py.File(inputpath + snapshot + ".hdf5", 'r')

    # Get the position and velocity of
    # each particle in this rank
    sim_pids = hdf[hdf_part_key]['part_pid'][part_inds]
    pos = hdf[hdf_part_key]['part_pos'][part_inds, :]
    vel = hdf[hdf_part_key]['part_vel'][part_inds, :]
    masses = hdf[hdf_part_key]['part_masses'][part_inds] * 10 ** 10
    part_types = hdf[hdf_part_key]['part_types'][part_inds]

    hdf.close()

    # Instantiate halo object
    halo = Halo(part_inds, sim_pids, pos, vel, part_types,
                masses, ini_vlcoeff, boxsize, soft,
                redshift, G, cosmo)

    return halo


def get_linked_halo_data(all_linked_halos, start_ind, nlinked_halos):
    """ A helper function for extracting a halo's linked halos
        (i.e. progenitors and descendants)

    :param all_linked_halos: Array containing all progenitors and descendants.
    :type all_linked_halos: float[N_linked halos]
    :param start_ind: The start index for this halos progenitors or
    descendents elements in all_linked_halos
    :type start_ind: int
    :param nlinked_halos: The number of progenitors or descendents
    (linked halos) the halo in question has
    :type nlinked_halos: int
    :return:
    """

    return all_linked_halos[start_ind: start_ind + nlinked_halos]
