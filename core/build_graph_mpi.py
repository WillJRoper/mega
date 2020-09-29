import numpy as np
import h5py
from multiprocessing import Lock
from guppy import hpy; hp = hpy()
from mpi4py import MPI
from functools import partial
import time
import pickle
import gc
import utilities


# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # density_rank of this process
status = MPI.Status()   # get MPI status object

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

    for snap in snaplist:

        graph_dict.setdefault(snap, set())

    # Loop until no new halos are found
    while len(new_halos) != 0:

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
                these_progs = utilities.get_linked_halo_data(data_dict['progs'][snap], data_dict['prog_start_index'][snap][halo[0]],
                                                             data_dict['nprogs'][snap][halo[0]])

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
                these_descs = utilities.get_linked_halo_data(data_dict['descs'][snap],
                                                             data_dict['desc_start_index'][snap][halo[0]],
                                                             data_dict['ndescs'][snap][halo[0]])

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

            # Convert entry to an array for sorting
            graph_dict[snap] = np.array([])

            # Get the halo masses
            mass_dict[snap] = np.array([])

            continue

        # Convert entry to an array for sorting
        graph_dict[snap] = np.array([int(halo[0]) for halo in graph_dict[snap]])

        # Get the halo masses
        mass_dict[snap] = data_dict['nparts'][snap][graph_dict[snap]]

        # Sort by mass
        sinds = np.argsort(mass_dict[snap])[::-1]
        mass_dict[snap] = mass_dict[snap][sinds]
        graph_dict[snap] = graph_dict[snap][sinds]

    return graph_dict, mass_dict


def graph_worker(root_halo, snaplist, verbose, data_dict):

    # lock.acquire()
    # with open('utilityfiles/roothalos.pck', 'rb') as pfile:
    #     roots = pickle.load(pfile)
    #     if verbose:
    #         print(len(roots) - 1, 'Roots')
    # lock.release()
    # if int(root_halo) in roots:
    #     if verbose:
    #         print('Halo ' + str(root_halo) + '\'s Forest exists...')
    #     return {}

    # Get the graph with this halo at it's root
    graph_dict, mass_dict = get_graph(root_halo, snaplist, data_dict)

    # print('Halo ' + str(root_halo) + '\'s Forest extracted...')

    # lock.acquire()
    # with open('utilityfiles/roothalos.pck', 'rb') as file:
    #     roots = pickle.load(file)
    # for root in graph_dict['061']:
    #     roots.update([root])
    # with open('utilityfiles/roothalos.pck', 'wb') as file:
    #     pickle.dump(roots, file)
    # lock.release()

    return graph_dict, mass_dict


def graph_writer(graphs, sub_graphs, graphpath, treepath, snaplist, data_dict):

    # Initialise roots set to ensure there isn't duplication
    done_roots = set()

    prog_conts = {}
    desc_conts = {}
    sub_prog_conts = {}
    sub_desc_conts = {}
    host_in_graph = np.full(len(data_dict['nparts'][snaplist[-1]]), 2**30)

    for snap in snaplist:

        # Open this graph file
        hdf = h5py.File(treepath + 'Mgraph_' + snap + '.hdf5', 'r')
        sub_hdf = h5py.File(treepath + 'SubMgraph_' + snap + '.hdf5', 'r')

        # Assign
        prog_conts[snap] = hdf['Prog_Mass_Contribution'][...]
        desc_conts[snap] = hdf['Desc_Mass_Contribution'][...]
        sub_prog_conts[snap] = sub_hdf['Prog_Mass_Contribution'][...]
        sub_desc_conts[snap] = sub_hdf['Desc_Mass_Contribution'][...]

        hdf.close()

    data_dict["prog_conts"] = prog_conts
    data_dict["desc_conts"] = desc_conts
    data_dict["sub"]["prog_conts"] = sub_prog_conts
    data_dict["sub"]["desc_conts"] = sub_desc_conts

    # Define lists for graph level data
    nhalo_in_graph = []
    root_mass = []
    graph_length = []

    # Initialise graph id
    graph_id = 0

    # Reverse the snapshot list such that the present day is the first element
    hdf = h5py.File(graphpath + '.hdf5', 'w')

    for graph in graphs:

        # Initialise internal graph ids
        graph_internal_id = 0

        # print(graph_internal_id)

        # Create dictionaries for ID matching
        internal2halocat = {}
        halocat2internal = {}

        # Initialise list data structures to store results
        this_graph = []
        this_graph_halo_cat_ids = []
        snaps = []
        snaps_str = []
        generations = []
        nparts = []
        generation_start_index = np.full(len(snaplist), 2**30)
        generation_length = np.full(len(snaplist), 2**30)

        if len(graph) == 0:
            continue

        graph_dict, mass_dict = graph

        # Get highest mass root halo
        root_halos = graph_dict[snaplist[0]]
        z0masses = mass_dict[snaplist[0]]

        # IDs in this generation of this graph
        root_mass.append(np.max(z0masses))

        if len(done_roots.intersection(set(root_halos))) > 0:
            continue

        done_roots.update(root_halos)

        # Assign roots to array
        host_in_graph[root_halos] = graph_id

        graph = hdf.create_group(str(graph_id))  # create halo group

        # Loop over snapshots
        generation = 0
        for snap_ind in range(len(snaplist)):

            # Extract this generation
            snap = snaplist[snap_ind]
            this_gen = graph_dict.pop(snap)
            this_gen_masses = mass_dict.pop(snap)

            if len(this_gen) == 0:
                continue
            else:

                # Assign the start index for this generation and the generations lengths
                generation_start_index[snap_ind] = len(this_graph)
                generation_length[snap_ind] = len(this_gen)

                # Assign this generations number
                generations.append(generation)

                for halo, m in zip(this_gen, this_gen_masses):

                    # Store these halos
                    this_graph.append(graph_internal_id)
                    this_graph_halo_cat_ids.append(halo)
                    snaps.append(int(snap))
                    snaps_str.append(snap)
                    nparts.append(m)

                    # Keep tracks of IDs
                    internal2halocat[graph_internal_id] = halo
                    halocat2internal[halo] = graph_internal_id

                    graph_internal_id += 1

            generation += 1

        # Assign the numbers of halos in this graph
        nhalo_in_graph.append(len(this_graph))

        # Set up direct progenitor and descendant arrays for data
        nprogs = np.full(len(this_graph), 2**30)
        ndescs = np.full(len(this_graph), 2**30)
        prog_start_index = np.full(len(this_graph), 2**30)
        desc_start_index = np.full(len(this_graph), 2**30)
        progs = []
        descs = []
        prog_mass_conts = []
        desc_mass_conts = []

        for snap, haloID in zip(snaps_str, this_graph):

            # Get halo catalog ID
            halo_cat_id = internal2halocat[haloID]

            # Get data for this halo
            this_nprog = data_dict['nprogs'][snap][halo_cat_id]
            this_ndesc = data_dict['ndescs'][snap][halo_cat_id]
            this_prog_start = data_dict['prog_start_index'][snap][halo_cat_id]
            this_desc_start = data_dict['desc_start_index'][snap][halo_cat_id]
            this_progs = utilities.get_linked_halo_data(data_dict['progs'][snap], this_prog_start, this_nprog)
            this_descs = utilities.get_linked_halo_data(data_dict['descs'][snap], this_desc_start, this_ndesc)
            this_prog_conts = utilities.get_linked_halo_data(data_dict['prog_conts'][snap], this_prog_start, this_nprog)
            this_desc_conts = utilities.get_linked_halo_data(data_dict['desc_conts'][snap], this_desc_start, this_ndesc)

            this_prog_graph_ids = [halocat2internal[i] for i in this_progs]
            this_desc_graph_ids = [halocat2internal[i] for i in this_descs]

            nprogs[haloID] = this_nprog  # number of progenitors
            ndescs[haloID] = this_ndesc  # number of descendants

            if this_nprog > 0:
                prog_start_index[haloID] = len(progs)
                progs.extend(this_prog_graph_ids)
                prog_mass_conts.extend(this_prog_conts)
            else:
                prog_start_index[haloID] = 2 ** 30

            if this_ndesc > 0:
                desc_start_index[haloID] = len(descs)
                descs.extend(this_desc_graph_ids)
                desc_mass_conts.extend(this_desc_conts)
            else:
                desc_start_index[haloID] = 2 ** 30

        # Get the length of this graph
        length = len(np.unique(snaps))
        graph_length.append(length)

        # IDs in this generation of this graph
        graph.attrs["length"] = length
        graph.attrs["root_mass"] = np.max(z0masses)
        graph.attrs["nhalos_in_graph"] = len(this_graph)
        graph.create_dataset('graph_halo_ids', data=np.array(this_graph), dtype=int, compression='gzip')
        graph.create_dataset('halo_catalog_halo_ids', data=np.array(this_graph_halo_cat_ids), dtype=int,
                             compression='gzip')
        graph.create_dataset('snapshots', data=np.array(snaps), dtype=int, compression='gzip')
        graph.create_dataset('generation_id', data=np.array(generations), dtype=int, compression='gzip')
        graph.create_dataset('nparts', data=np.array(nparts), dtype=int, compression='gzip')
        graph.create_dataset('generation_start_index', data=generation_start_index, dtype=int, compression='gzip')
        graph.create_dataset('generation_length', data=generation_length, dtype=int, compression='gzip')
        graph.create_dataset('nprog', data=nprogs, dtype=int, compression='gzip')
        graph.create_dataset('ndesc', data=ndescs, dtype=int, compression='gzip')
        graph.create_dataset('prog_start_index', data=prog_start_index, dtype=int, compression='gzip')
        graph.create_dataset('desc_start_index', data=desc_start_index, dtype=int, compression='gzip')
        graph.create_dataset('direct_prog_ids', data=np.array(progs), dtype=int, compression='gzip')
        graph.create_dataset('direct_desc_ids', data=np.array(descs), dtype=int, compression='gzip')
        graph.create_dataset('direct_prog_contribution', data=np.array(this_prog_conts), dtype=int, compression='gzip')
        graph.create_dataset('direct_desc_contribution', data=np.array(this_desc_conts), dtype=int, compression='gzip')

        graph_id += 1

    hdf.create_dataset('graph_lengths', data=np.array(graph_length), dtype=int, compression='gzip')
    hdf.create_dataset('root_nparts', data=np.array(root_mass), dtype=int, compression='gzip')
    hdf.create_dataset('nhalos_in_graph', data=np.array(nhalo_in_graph), dtype=int, compression='gzip')

    hdf.close()

    # ==================================== Subhalo graph ====================================

    # Reverse the snapshot list such that the present day is the first element
    hdf = h5py.File(graphpath + '.hdf5', 'r+')

    for graph in sub_graphs:

        # Initialise internal graph ids
        graph_internal_id = 0

        # Create dictionaries for ID matching
        internal2halocat = {}
        halocat2internal = {}

        # Initialise list data structures to store results
        this_graph = []
        this_graph_halo_cat_ids = []
        snaps = []
        snaps_str = []
        generations = []
        nparts = []
        generation_start_index = np.full(len(snaplist), 2 ** 30)
        generation_length = np.full(len(snaplist), 2 ** 30)

        if len(graph) == 0:
            continue

        graph_dict, mass_dict = graph

        # Get highest mass root halo
        root_halos = graph_dict[snaplist[0]]
        z0masses = mass_dict[snaplist[0]]

        # IDs in this generation of this graph
        root_mass.append(np.max(z0masses))

        if len(done_roots.intersection(set(root_halos))) > 0:
            continue

        done_roots.update(root_halos)

        # Get host id
        host_ids = data_dict["sub"]["hosts"][snaplist[-1]][root_halos]

        # Assign roots to array
        graph_id = np.unique(host_in_graph[host_ids])

        assert len(graph_id) == 1, "Subhalos populate multiple host level graphs, something is VERY wrong"

        graph = hdf[str(graph_id)]

        # Loop over snapshots
        generation = 0
        for snap_ind in range(len(snaplist)):

            # Extract this generation
            snap = snaplist[snap_ind]
            this_gen = graph_dict.pop(snap)
            this_gen_masses = mass_dict.pop(snap)

            if len(this_gen) == 0:
                continue
            else:

                # Assign the start index for this generation and the generations lengths
                generation_start_index[snap_ind] = len(this_graph)
                generation_length[snap_ind] = len(this_gen)

                # Assign this generations number
                generations.append(generation)

                for halo, m in zip(this_gen, this_gen_masses):
                    # Store these halos
                    this_graph.append(graph_internal_id)
                    this_graph_halo_cat_ids.append(halo)
                    snaps.append(int(snap))
                    snaps_str.append(snap)
                    nparts.append(m)

                    # Keep tracks of IDs
                    internal2halocat[graph_internal_id] = halo
                    halocat2internal[halo] = graph_internal_id

                    graph_internal_id += 1

            generation += 1

        # Assign the numbers of halos in this graph
        nhalo_in_graph.append(len(this_graph))

        # Set up direct progenitor and descendant arrays for data
        nprogs = np.full(len(this_graph), 2 ** 30)
        ndescs = np.full(len(this_graph), 2 ** 30)
        prog_start_index = np.full(len(this_graph), 2 ** 30)
        desc_start_index = np.full(len(this_graph), 2 ** 30)
        progs = []
        descs = []
        prog_mass_conts = []
        desc_mass_conts = []

        for snap, haloID in zip(snaps_str, this_graph):

            # Get halo catalog ID
            halo_cat_id = internal2halocat[haloID]

            # Get data for this halo
            this_nprog = data_dict['nprogs'][snap][halo_cat_id]
            this_ndesc = data_dict['ndescs'][snap][halo_cat_id]
            this_prog_start = data_dict['prog_start_index'][snap][halo_cat_id]
            this_desc_start = data_dict['desc_start_index'][snap][halo_cat_id]
            this_progs = utilities.get_linked_halo_data(data_dict['progs'][snap], this_prog_start, this_nprog)
            this_descs = utilities.get_linked_halo_data(data_dict['descs'][snap], this_desc_start, this_ndesc)
            this_prog_conts = utilities.get_linked_halo_data(data_dict['prog_conts'][snap], this_prog_start, this_nprog)
            this_desc_conts = utilities.get_linked_halo_data(data_dict['desc_conts'][snap], this_desc_start, this_ndesc)

            this_prog_graph_ids = [halocat2internal[i] for i in this_progs]
            this_desc_graph_ids = [halocat2internal[i] for i in this_descs]

            nprogs[haloID] = this_nprog  # number of progenitors
            ndescs[haloID] = this_ndesc  # number of descendants

            if this_nprog > 0:
                prog_start_index[haloID] = len(progs)
                progs.extend(this_prog_graph_ids)
                prog_mass_conts.extend(this_prog_conts)
            else:
                prog_start_index[haloID] = 2 ** 30

            if this_ndesc > 0:
                desc_start_index[haloID] = len(descs)
                descs.extend(this_desc_graph_ids)
                desc_mass_conts.extend(this_desc_conts)
            else:
                desc_start_index[haloID] = 2 ** 30

        # Get the length of this graph
        length = len(np.unique(snaps))
        graph_length.append(length)

        # IDs in this generation of this graph
        graph.attrs["sub_length"] = length
        graph.attrs["sub_root_mass"] = np.max(z0masses)
        graph.attrs["sub_nhalos_in_graph"] = len(this_graph)
        graph.create_dataset('sub_graph_halo_ids', data=np.array(this_graph), dtype=int, compression='gzip')
        graph.create_dataset('subhalo_catalog_halo_ids', data=np.array(this_graph_halo_cat_ids), dtype=int,
                             compression='gzip')
        graph.create_dataset('sub_snapshots', data=np.array(snaps), dtype=int, compression='gzip')
        graph.create_dataset('sub_generation_id', data=np.array(generations), dtype=int, compression='gzip')
        graph.create_dataset('sub_nparts', data=np.array(nparts), dtype=int, compression='gzip')
        graph.create_dataset('sub_generation_start_index', data=generation_start_index, dtype=int, compression='gzip')
        graph.create_dataset('sub_generation_length', data=generation_length, dtype=int, compression='gzip')
        graph.create_dataset('sub_nprog', data=nprogs, dtype=int, compression='gzip')
        graph.create_dataset('sub_ndesc', data=ndescs, dtype=int, compression='gzip')
        graph.create_dataset('sub_prog_start_index', data=prog_start_index, dtype=int, compression='gzip')
        graph.create_dataset('sub_desc_start_index', data=desc_start_index, dtype=int, compression='gzip')
        graph.create_dataset('sub_direct_prog_ids', data=np.array(progs), dtype=int, compression='gzip')
        graph.create_dataset('sub_direct_desc_ids', data=np.array(descs), dtype=int, compression='gzip')
        graph.create_dataset('sub_direct_prog_contribution', data=np.array(this_prog_conts), dtype=int, compression='gzip')
        graph.create_dataset('sub_direct_desc_contribution', data=np.array(this_desc_conts), dtype=int, compression='gzip')

    hdf.create_dataset('sub_graph_lengths', data=np.array(graph_length), dtype=int, compression='gzip')
    hdf.create_dataset('sub_root_nparts', data=np.array(root_mass), dtype=int, compression='gzip')
    hdf.create_dataset('sub_nhalos_in_graph', data=np.array(nhalo_in_graph), dtype=int, compression='gzip')

    hdf.close()


def main_get_graph_members(treepath, graphpath, snaplist, verbose, halopath):

    # Get the root snapshot
    snaplist.reverse()
    root_snap = snaplist[0]
    past2present_snaplist = list(reversed(snaplist))

    # Create file to store this snapshots graph results
    hdf = h5py.File(treepath + 'Mgraph_' + root_snap + '.hdf5', 'r')

    # Extract the halo IDs (group names/keys) contained within this snapshot and the realness flag
    halo_ids = hdf['halo_IDs'][...]
    reals = hdf['real_flag'][...]

    hdf.close()

    # Extract only the real roots
    roots = halo_ids[reals]

    # Get the work for each task
    lims = np.linspace(0, roots.size, size + 1, dtype=int)
    low_lims = lims[:-1]
    up_lims = lims[1:]

    # Define the roots for this rank
    myroots = roots[low_lims[rank]: up_lims[rank]]
    print(roots.size, rank, low_lims[rank], up_lims[rank], len(myroots))

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
        hdf = h5py.File(treepath + 'Mgraph_' + snap + '.hdf5', 'r')

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

    # Initialise the tuple for storing results
    graphs = []

    for thisTask in myroots:

        result = graph_worker(thisTask, snaplist, verbose, data_dict)
        graphs.append(result)

    collected_results = comm.gather(graphs, root=0)

    # Create file to store this snapshots graph results
    hdf = h5py.File(treepath + 'SubMgraph_' + root_snap + '.hdf5', 'r')

    # Extract the halo IDs (group names/keys) contained within this snapshot and the realness flag
    subhalo_ids = hdf['halo_IDs'][...]
    sub_reals = hdf['real_flag'][...]

    hdf.close()

    # Extract only the real roots
    sub_roots = subhalo_ids[sub_reals]

    # Get the work for each task
    sub_lims = np.linspace(0, sub_roots.size, size + 1, dtype=int)
    sub_low_lims = sub_lims[:-1]
    sub_up_lims = sub_lims[1:]

    # Define the roots for this rank
    sub_myroots = sub_roots[sub_low_lims[rank]: sub_up_lims[rank]]
    print(sub_roots.size, rank, sub_low_lims[rank], sub_up_lims[rank], len(sub_myroots))

    # Get the start indices, progs, and descs and store them in dictionaries
    progs = {}
    descs = {}
    nprogs = {}
    ndescs = {}
    prog_start_index = {}
    desc_start_index = {}
    nparts = {}
    hosts = {}
    for snap in snaplist:

        # Open this graph file
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

        hdf = h5py.File(halopath + 'halos_' + str(snap) + '.hdf5', 'r')

        hosts[snap] = hdf['Subhalos']['host_IDs'][...]

        hdf.close()

    data_dict["sub"] = {'progs': progs, 'descs': descs, 'nprogs': nprogs, 'ndescs': ndescs,
                        'prog_start_index': prog_start_index, 'desc_start_index': desc_start_index, 'nparts': nparts,
                        "hosts": hosts}

    # Initialise the tuple for storing results
    sub_graphs = []

    for thisTask in myroots:

        result = graph_worker(thisTask, snaplist, verbose, data_dict["sub"])
        sub_graphs.append(result)

    sub_collected_results = comm.gather(sub_graphs, root=0)

    if rank == 0:

        results = []
        for col_res in collected_results:
            results.extend(col_res)

        sub_results = []
        for col_res in sub_collected_results:
            sub_results.extend(col_res)

        # Write out the result
        graph_writer(results, sub_results, graphpath, treepath, past2present_snaplist, data_dict)
