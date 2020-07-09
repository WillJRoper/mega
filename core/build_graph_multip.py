import numpy as np
import h5py
import multiprocessing
from multiprocessing import Lock
from functools import partial
from guppy import hpy; hp = hpy()
import pickle
from utilities import get_linked_halo_data


# Initialise lock object
lock = Lock()


def get_graph(z0halo, snaplist, data_dict):
    """ A funciton which traverses a graph including all linked halos.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param z0halo: The halo ID of a z=0 halo for which the graph is desired.

    :return: graph_dict: The dictionary containing the graph. Each key is the snapshot ID and
             the value is a list of halos in this snapshot of the graph.
             massgrowth: The mass history of the graph.
             tree: The dictionary containing the tree. Each key is the snapshot ID and
             the value is a list of halos in this snapshot of the tree.
             main_growth: The mass history of the main branch.
    """

    # Initialise dictionary instances
    graph_dict = {}
    mass_dict = {}

    # Initialise the halo's set for tree walking
    halos = {(z0halo, snaplist[0])}

    # Initialise the graph dictionary with the present day halo as the first entry
    graph_dict[snaplist[0]] = halos

    # Initialise the set of new found halos used to loop until no new halos are found
    new_halos = halos

    # Initialise the set of found halos used to check if the halo has been found or not
    found_halos = set()

    # =============== Progenitors ===============

    count = 0

    # Loop until no new halos are found
    while len(new_halos) != 0:

        print(count)
        count += 1

        # Overwrite the last set of new_halos
        new_halos = set()

        # =============== Progenitors ===============

        # Loop over snapshots and progenitor snapshots
        for prog_snap, snap in zip(snaplist[1:], snaplist[:-1]):

            # Assign the halos variable for the next stage of the tree
            halos = graph_dict[snap]

            # Loop over halos in this snapshot
            for halo in halos:

                # Get the progenitors
                these_progs = get_linked_halo_data(data_dict['progs'], data_dict['prog_start_index'][halo[0]],
                                                   data_dict['nprogs'][halo[0]])

                # Assign progenitors using a tuple to keep track of the snapshot ID
                # in addition to the halo ID
                graph_dict.setdefault(prog_snap, set()).update({(p, prog_snap) for p in these_progs})

            # Add any new halos not found in found halos to the new halos set
            new_halos.update(graph_dict[prog_snap] - found_halos)

        # =============== Descendants ===============

        # Loop over halos found during the progenitor step
        snapshots = list(reversed(list(graph_dict.keys())))
        for desc_snap, snap in zip(snapshots[1:], snapshots[:-1]):

            # Assign the halos variable for the next stage of the tree
            halos = graph_dict[snap]

            # Loop over the progenitor halos
            for halo in halos:

                # Get the descendants
                these_descs = get_linked_halo_data(data_dict['descs'], data_dict['desc_start_index'][halo[0]],
                                                   data_dict['ndescs'][halo[0]])

                # Load descendants adding the snapshot * 100000 to keep track of the snapshot ID
                # in addition to the halo ID
                graph_dict.setdefault(desc_snap, set()).update({(d, desc_snap) for d in these_descs})

            # Redefine the new halos set to have any new halos not found in found halos
            new_halos.update(graph_dict[desc_snap] - found_halos)

        # Add the new_halos to the found halos set
        found_halos.update(new_halos)

    # Get the number of particle in each halo and sort based on mass
    for snap in graph_dict:

        if len(graph_dict[snap]) == 0:
            continue

        # Convert entry to an array for sorting
        graph_dict[snap] = np.array([int(halo[0]) for halo in graph_dict[snap]])

        # Get the halo masses
        mass_dict[snap] = data_dict['nparts'][graph_dict[snap]]

        # Sort by mass
        sinds = np.argsort(mass_dict[snap])[::-1]
        mass_dict[snap] = mass_dict[snap][sinds]
        graph_dict[snap] = graph_dict[snap][sinds]

    return graph_dict, mass_dict


def graph_worker(root_halo, snaplist, verbose):

    lock.acquire()
    with open('utilityfiles/roothalos.pck', 'rb') as pfile:
        roots = pickle.load(pfile)
        if verbose:
            print(len(roots) - 1, 'Roots')
    with open('utilityfiles/link_datadict.pck', 'rb') as pfile:
        data_dict = pickle.load(pfile)
    lock.release()
    if int(root_halo) in roots:
        if verbose:
            print('Halo ' + str(root_halo) + '\'s Forest exists...')
        return {}

    # Get the graph with this halo at it's root
    graph_dict, mass_dict = get_graph(root_halo, snaplist, data_dict)

    print('Halo ' + str(root_halo) + '\'s Forest extracted...')

    lock.acquire()
    with open('utilityfiles/roothalos.pck', 'rb') as file:
        roots = pickle.load(file)
    for root in graph_dict['061']:
        roots.update([root])
    with open('utilityfiles/roothalos.pck', 'wb') as file:
        pickle.dump(roots, file)
    lock.release()

    return graph_dict, mass_dict


def graph_writer(graphs, graphpath, snaplist, verbose):

    # Reverse the snapshot list such that the present day is the first element

    hdf = h5py.File(graphpath + '.hdf5', 'w')

    for graph in graphs:

        # Initialise list to store this graphs data and start index array
        this_graph = []
        generation_start_index = np.full(len(snaplist), -2)
        generation_length = np.zeros(len(snaplist))

        if len(graph) == 0:
            continue

        graph_dict, mass_dict = graph

        # Get highest mass root halo
        root_halos = graph_dict[snaplist[0]]
        z0masses = mass_dict[snaplist[0]]

        # IDs in this generation of this graph
        root_halo = root_halos[np.argmax(z0masses)]

        if str(root_halo) in hdf.keys():
            if verbose:
                print(str(root_halo) + '\'s Forest already exists')
            continue

        if verbose:
            print('Creating Group', str(root_halo) + '...', end='\r')

        graph = hdf.create_group(str(root_halo))  # create halo group

        for snap in range(snaplist):

            # Extract this generation
            this_gen = graph_dict.pop(snaplist[snap])

            if len(this_gen) == 0:
                continue
            else:

                # Assign the start index for this generation and the generations lengths
                generation_start_index[snap] = len(this_graph)
                generation_length[snap] = len(this_gen)

                # Get the halos in this generation
                this_graph.extend(this_gen)

        # IDs in this generation of this graph
        graph.create_dataset('halo_ids', data=np.array(this_graph), dtype=int, compression='gzip')
        graph.create_dataset('generation_length', data=np.array(this_graph), dtype=int, compression='gzip')

    hdf.close()


def main_get_graph_members(treepath, graphpath, snaplist, density_rank, verbose):

    # Get the root snapshot
    snaplist.reverse()
    root_snap = snaplist[0]

    # Create file to store this snapshots graph results
    if density_rank == 0:

        hdf = h5py.File(treepath + 'Mgraph_' + root_snap + '.hdf5', 'r')

    else:

        hdf = h5py.File(treepath + 'SubMgraph_' + root_snap + '.hdf5', 'r')

    # Extract the halo IDs (group names/keys) contained within this snapshot and the realness flag
    halo_ids = hdf['halo_IDs'][...]
    reals = hdf['real_flag'][...]

    hdf.close()

    # Get the start indices, progs, and descs and store them in dictionaries
    progs = {}
    descs = {}
    nprogs = {}
    ndescs = {}
    prog_start_index = {}
    desc_start_index = {}
    nparts = {}
    for snap in snaplist:

        # Open this graph file
        if density_rank == 0:

            hdf = h5py.File(treepath + 'Mgraph_' + snap + '.hdf5', 'r')

        else:

            hdf = h5py.File(treepath + 'SubMgraph_' + snap + '.hdf5', 'r')

        # Assign
        progs[snap] = hdf['Prog_haloIDs'][...]
        descs[snap] = hdf['Desc_haloIDs'][...]
        nprogs[snap] = hdf['nProgs'][...]
        ndescs[snap] = hdf['nDescs'][...]
        prog_start_index[snap] = hdf['prog_start_index'][...]
        desc_start_index[snap] = hdf['desc_start_index'][...]
        nparts[snap] = hdf['nparts'][...]

        hdf.close()

    data_dict = {'progs': progs, 'descs': descs, 'nprogs': nprogs, 'ndescs': ndescs,
                 'prog_start_index': prog_start_index, 'desc_start_index': desc_start_index, 'nparts': nparts}

    with open('utilityfiles/link_datadict.pck', 'wb') as file:
        pickle.dump(data_dict, file)

    roots = {-999}

    with open('utilityfiles/roothalos.pck', 'wb') as file:
        pickle.dump(roots, file)

    # Extract only the real roots
    roots = halo_ids[reals]

    p = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2)
    graphs = p.map(partial(graph_worker, snaplist=snaplist, verbose=verbose), iter(roots))
    p.close()
    p.join()

    # Write out the result
    graph_writer(graphs, graphpath, snaplist, verbose)
