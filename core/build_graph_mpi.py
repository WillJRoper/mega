from multiprocessing import Lock

import h5py
import numpy as np
# from guppy import hpy;

hp = hpy()
from mpi4py import MPI
import utilities

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # density_rank of this process
status = MPI.Status()  # get MPI status object

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
    if type(z0halo) == list:
        halos = {(i, snaplist[0]) for i in z0halo}
    else:
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
        for ind, snap in enumerate(snaplist):

            if ind - 1 >= 0:
                desc_snap = snaplist[ind - 1]
            else:
                desc_snap = None

            if ind + 1 < len(snaplist):
                prog_snap = snaplist[ind + 1]
            else:
                prog_snap = None

            # Assign the halos variable for the next stage of the tree
            halos = graph_dict[snap]

            # Loop over halos in this snapshot
            for halo in halos:

                if prog_snap != None:
                    # Get the progenitors
                    these_progs = utilities.get_linked_halo_data(
                        data_dict['progs'][snap],
                        data_dict['prog_start_index'][snap][halo[0]],
                        data_dict['nprogs'][snap][halo[0]])

                    # Assign progenitors using a tuple to keep track of the snapshot ID
                    # in addition to the halo ID
                    graph_dict.setdefault(prog_snap, set()).update(
                        {(p, prog_snap) for p in these_progs})

                if desc_snap != None:
                    # Get the descendants
                    these_descs = utilities.get_linked_halo_data(
                        data_dict['descs'][snap],
                        data_dict['desc_start_index'][snap][halo[0]],
                        data_dict['ndescs'][snap][halo[0]])

                    # Load descendants adding the snapshot * 100000 to keep track of the snapshot ID
                    # in addition to the halo ID
                    graph_dict.setdefault(desc_snap, set()).update(
                        {(d, desc_snap) for d in these_descs})

            # Add any new halos not found in found halos to the new halos set
            if prog_snap != None:
                new_halos.update(graph_dict[prog_snap] - found_halos)
            if desc_snap != None:
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
        graph_dict[snap] = np.array(
            [int(halo[0]) for halo in graph_dict[snap]])

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
    host_in_graph = np.full(len(data_dict['nparts'][snaplist[-1]]), 2 ** 30)

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

    # Add metadata
    header = hdf.create_group("Header")
    header.attrs["part_mass"] = data_dict["pmass"]
    header.attrs["NO_DATA_INT"] = 2 ** 30
    header.attrs["NO_DATA_FLOAT"] = np.nan

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
        prog_snaps_str = []
        desc_snaps_str = []
        generations = np.full(len(snaplist), 2 ** 30, dtype=np.int32)
        nparts = []
        generation_start_index = np.full(len(snaplist), 2 ** 30)
        generation_length = np.full(len(snaplist), 2 ** 30)

        if len(graph) == 0:
            continue

        graph_dict, mass_dict = graph

        # Get highest mass root halo
        root_halos = graph_dict[snaplist[-1]]
        z0masses = mass_dict[snaplist[-1]]

        # IDs in this generation of this graph
        root_mass.append(np.max(z0masses))

        if len(done_roots.intersection(set(root_halos))) > 0:
            continue

        done_roots.update(root_halos)

        # Assign roots to array
        host_in_graph[root_halos] = graph_id

        graph = hdf.create_group(str(graph_id))  # create halo group

        # Loop over snapshots
        for snap_ind, snap in enumerate(snaplist):

            if snap_ind - 1 >= 0:
                prog_snap = snaplist[snap_ind - 1]
            else:
                prog_snap = None

            if snap_ind + 1 < len(snaplist):
                desc_snap = snaplist[snap_ind + 1]
            else:
                desc_snap = None

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
                generations[snap_ind] = snap_ind

                for halo, m in zip(this_gen, this_gen_masses):
                    # Store these halos
                    this_graph.append(graph_internal_id)
                    this_graph_halo_cat_ids.append(halo)
                    snaps.append(int(snap))
                    snaps_str.append(snap)
                    prog_snaps_str.append(prog_snap)
                    desc_snaps_str.append(desc_snap)
                    nparts.append(m)

                    # Keep tracks of IDs
                    internal2halocat[graph_internal_id] = halo
                    halocat2internal[(snap, halo)] = graph_internal_id

                    graph_internal_id += 1

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
        mean_pos = np.full((len(this_graph), 3), np.nan, dtype=float)
        mean_vel = np.full((len(this_graph), 3), np.nan, dtype=float)
        rms_rad = np.full(len(this_graph), np.nan, dtype=float)
        zs = np.full(len(this_graph), np.nan, dtype=float)
        vdisp = np.full(len(this_graph), np.nan, dtype=float)
        vmax = np.full(len(this_graph), np.nan, dtype=float)
        hmrs = np.full(len(this_graph), np.nan, dtype=float)
        hmvrs = np.full(len(this_graph), np.nan, dtype=float)

        for snap, psnap, dsnap, haloID in zip(snaps_str, prog_snaps_str,
                                              desc_snaps_str, this_graph):

            # Get halo catalog ID
            halo_cat_id = internal2halocat[haloID]

            # Get data for this halo
            this_nprog = data_dict['nprogs'][snap][halo_cat_id]
            this_ndesc = data_dict['ndescs'][snap][halo_cat_id]
            this_prog_start = data_dict['prog_start_index'][snap][halo_cat_id]
            this_desc_start = data_dict['desc_start_index'][snap][halo_cat_id]
            this_progs = utilities.get_linked_halo_data(
                data_dict['progs'][snap], this_prog_start, this_nprog)
            this_descs = utilities.get_linked_halo_data(
                data_dict['descs'][snap], this_desc_start, this_ndesc)
            this_prog_conts = utilities.get_linked_halo_data(
                data_dict['prog_conts'][snap], this_prog_start, this_nprog)
            this_desc_conts = utilities.get_linked_halo_data(
                data_dict['desc_conts'][snap], this_desc_start, this_ndesc)

            mean_pos[haloID, :] = data_dict["mean_pos"][snap][halo_cat_id, :]
            mean_vel[haloID, :] = data_dict["mean_vel"][snap][halo_cat_id, :]
            rms_rad[haloID] = data_dict["rms_rad"][snap][halo_cat_id]
            vdisp[haloID] = data_dict["vdisp"][snap][halo_cat_id]
            vmax[haloID] = data_dict["vmax"][snap][halo_cat_id]
            hmrs[haloID] = data_dict["hmrs"][snap][halo_cat_id]
            hmvrs[haloID] = data_dict["hmvrs"][snap][halo_cat_id]

            zs[haloID] = data_dict["redshift"][snap]

            this_prog_graph_ids = [halocat2internal[(psnap, i)] for i in
                                   this_progs]
            this_desc_graph_ids = [halocat2internal[(dsnap, i)] for i in
                                   this_descs]

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
        # graph.create_dataset('graph_halo_ids', data=np.array(this_graph), dtype=np.int32, compression='gzip')
        graph.create_dataset('halo_catalog_halo_ids',
                             data=np.array(this_graph_halo_cat_ids),
                             dtype=np.int64,
                             compression='gzip')
        graph.create_dataset('snapshots', data=np.array(snaps), dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('redshifts', data=zs, dtype=float,
                             compression='gzip')
        graph.create_dataset('generation_id', data=np.array(generations),
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('nparts', data=np.array(nparts), dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('mean_pos', data=mean_pos, dtype=float,
                             compression='gzip')
        graph.create_dataset('mean_vel', data=mean_vel, dtype=float,
                             compression='gzip')
        graph.create_dataset('rms_radius', data=rms_rad, dtype=float,
                             compression='gzip')
        graph.create_dataset('3D_velocity_dispersion', data=vdisp, dtype=float,
                             compression='gzip')
        graph.create_dataset('v_max', data=vmax, dtype=float,
                             compression='gzip')
        graph.create_dataset('half_mass_radius', data=hmrs, dtype=float,
                             compression='gzip')
        graph.create_dataset('half_mass_velocity_radius', data=hmvrs,
                             dtype=float,
                             compression='gzip')
        graph.create_dataset('generation_start_index',
                             data=generation_start_index, dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('generation_length', data=generation_length,
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('nprog', data=nprogs, dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('ndesc', data=ndescs, dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('prog_start_index', data=prog_start_index,
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('desc_start_index', data=desc_start_index,
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('direct_prog_ids', data=np.array(progs),
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('direct_desc_ids', data=np.array(descs),
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('direct_prog_contribution',
                             data=np.array(prog_mass_conts), dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('direct_desc_contribution',
                             data=np.array(desc_mass_conts), dtype=np.int32,
                             compression='gzip')

        graph_id += 1

    hdf.create_dataset('graph_lengths', data=np.array(graph_length),
                       dtype=np.int32, compression='gzip')
    hdf.create_dataset('root_nparts', data=np.array(root_mass), dtype=np.int32,
                       compression='gzip')
    hdf.create_dataset('nhalos_in_graph', data=np.array(nhalo_in_graph),
                       dtype=np.int32, compression='gzip')

    hdf.close()

    sub_graph_length = np.zeros_like(graph_length)
    sub_nhalo_in_graph = np.zeros_like(nhalo_in_graph)

    # ==================================== Subhalo graph ====================================

    # Reinitialise the done root set to avoid duplication of a graph
    done_roots = set()

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
        hosts = []
        snaps = []
        snaps_str = []
        prog_snaps_str = []
        desc_snaps_str = []
        generations = np.full(len(snaplist), 2 ** 30)
        nparts = []
        generation_start_index = np.full(len(snaplist), 2 ** 30)
        generation_length = np.full(len(snaplist), 2 ** 30)

        if len(graph) == 0:
            continue

        graph_dict, mass_dict = graph

        # Get highest mass root halo
        root_halos = graph_dict[snaplist[-1]]
        z0masses = mass_dict[snaplist[-1]]

        if len(done_roots.intersection(set(root_halos))) > 0:
            continue

        done_roots.update(root_halos)

        # Get host id
        host_ids = data_dict["sub"]["hosts"][snaplist[-1]][root_halos]

        # Assign roots to array
        graph_id = np.unique(host_in_graph[host_ids])[0]

        assert len(np.unique(host_in_graph[host_ids])) == 1, \
            "Subhalos populate multiple host level graphs, " \
            "something is VERY wrong"

        graph = hdf[str(graph_id)]

        host_cat_ids = graph["halo_catalog_halo_ids"][...]
        host_snap_ids = graph["snapshots"][...]
        host_and_snap = np.array([str(h) + "_" + str(s)
                                  for h, s in zip(host_cat_ids,
                                                  host_snap_ids)])

        # Create array to store start pointer and lengths for subhalos
        subhalo_start_index = np.full(len(host_cat_ids), 2 ** 30,
                                      dtype=np.int32)
        nsubhalos = np.zeros(len(host_cat_ids), dtype=np.int32)

        # Loop over snapshots
        for snap_ind, snap in enumerate(snaplist):

            if snap_ind - 1 >= 0:
                prog_snap = snaplist[snap_ind - 1]
            else:
                prog_snap = None

            if snap_ind + 1 < len(snaplist):
                desc_snap = snaplist[snap_ind + 1]
            else:
                desc_snap = None

            # Extract this generation
            snap = snaplist[snap_ind]
            this_gen = graph_dict.pop(snap)
            this_gen_masses = mass_dict.pop(snap)

            if len(this_gen) == 0:
                continue
            else:

                # Order this generation by the host halo ID
                this_hosts = data_dict["sub"]["hosts"][snap][this_gen]
                sinds = np.argsort(this_hosts)
                this_hosts = this_hosts[sinds]
                this_gen = this_gen[sinds]
                this_gen_masses = this_gen_masses[sinds]

                # Sort each host halo's subhalos by mass
                for host in np.unique(this_hosts):
                    okinds = np.where(this_hosts == host)
                    sinds = np.argsort(this_gen_masses[okinds])[::-1]
                    this_gen[okinds] = this_gen[okinds][sinds]
                    this_gen_masses[okinds] = this_gen_masses[okinds][sinds]

                # Assign the start index for this generation and the
                # generations lengths
                generation_start_index[snap_ind] = len(this_graph)
                generation_length[snap_ind] = len(this_gen)

                # Assign this generations number
                generations[snap_ind] = snap_ind

                # Initialise prev_halo pointer
                prev_host = -1

                for halo, m, host in zip(this_gen,
                                         this_gen_masses,
                                         this_hosts):

                    # Get the internal host halo ID associated to the host
                    # of this subhalo
                    this_host_intern = np.where(host_and_snap
                                                == str(host) + "_"
                                                + str(int(snap)))[0]

                    # If we have moved on to a new halo set it's start pointer
                    if host != prev_host:
                        subhalo_start_index[this_host_intern] = len(this_graph)

                    nsubhalos[this_host_intern] += 1

                    prev_host = host

                    # Store these halos
                    this_graph.append(graph_internal_id)
                    this_graph_halo_cat_ids.append(halo)
                    hosts.append(this_host_intern[0])
                    snaps.append(int(snap))
                    snaps_str.append(snap)
                    prog_snaps_str.append(prog_snap)
                    desc_snaps_str.append(desc_snap)
                    nparts.append(m)

                    # Keep tracks of IDs
                    internal2halocat[graph_internal_id] = halo
                    halocat2internal[(snap, halo)] = graph_internal_id

                    graph_internal_id += 1

        # Assign the numbers of halos in this graph
        sub_nhalo_in_graph[graph_id] = len(this_graph)

        # Set up direct progenitor and descendant arrays for data
        nprogs = np.full(len(this_graph), 2 ** 30)
        ndescs = np.full(len(this_graph), 2 ** 30)
        prog_start_index = np.full(len(this_graph), 2 ** 30)
        desc_start_index = np.full(len(this_graph), 2 ** 30)
        progs = []
        descs = []
        prog_mass_conts = []
        desc_mass_conts = []
        mean_pos = np.full((len(this_graph), 3), np.nan, dtype=float)
        mean_vel = np.full((len(this_graph), 3), np.nan, dtype=float)
        rms_rad = np.full(len(this_graph), np.nan, dtype=float)
        zs = np.full(len(this_graph), 2 ** 30, dtype=float)
        vdisp = np.full(len(this_graph), 2 ** 30, dtype=float)
        vmax = np.full(len(this_graph), 2 ** 30, dtype=float)
        hmrs = np.full(len(this_graph), 2 ** 30, dtype=float)
        hmvrs = np.full(len(this_graph), 2 ** 30, dtype=float)

        for snap, psnap, dsnap, haloID in zip(snaps_str, prog_snaps_str,
                                              desc_snaps_str, this_graph):

            # Get halo catalog ID
            halo_cat_id = internal2halocat[haloID]

            # Get data for this halo
            this_nprog = data_dict["sub"]['nprogs'][snap][halo_cat_id]
            this_ndesc = data_dict["sub"]['ndescs'][snap][halo_cat_id]
            this_prog_start = data_dict["sub"]['prog_start_index'][snap][
                halo_cat_id]
            this_desc_start = data_dict["sub"]['desc_start_index'][snap][
                halo_cat_id]
            this_progs = utilities.get_linked_halo_data(
                data_dict["sub"]['progs'][snap], this_prog_start, this_nprog)
            this_descs = utilities.get_linked_halo_data(
                data_dict["sub"]['descs'][snap], this_desc_start, this_ndesc)
            this_prog_conts = utilities.get_linked_halo_data(
                data_dict["sub"]['prog_conts'][snap], this_prog_start,
                this_nprog)
            this_desc_conts = utilities.get_linked_halo_data(
                data_dict["sub"]['desc_conts'][snap], this_desc_start,
                this_ndesc)

            mean_pos[haloID, :] = data_dict["sub"]["mean_pos"][snap][
                                  halo_cat_id, :]
            mean_vel[haloID, :] = data_dict["sub"]["mean_vel"][snap][
                                  halo_cat_id, :]
            rms_rad[haloID] = data_dict["sub"]["rms_rad"][snap][halo_cat_id]
            vdisp[haloID] = data_dict["sub"]["vdisp"][snap][halo_cat_id]
            vmax[haloID] = data_dict["sub"]["vmax"][snap][halo_cat_id]
            hmrs[haloID] = data_dict["sub"]["hmrs"][snap][halo_cat_id]
            hmvrs[haloID] = data_dict["sub"]["hmvrs"][snap][halo_cat_id]

            zs[haloID] = data_dict["redshift"][snap]

            this_prog_graph_ids = [halocat2internal[(psnap, i)] for i in
                                   this_progs]
            this_desc_graph_ids = [halocat2internal[(dsnap, i)] for i in
                                   this_descs]

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
        sub_graph_length[graph_id] = length

        # Write out Subhalo attributes
        graph.attrs["sub_length"] = length
        graph.attrs["sub_root_mass"] = np.max(z0masses)
        graph.attrs["sub_nhalos_in_graph"] = len(this_graph)

        # Write out subhalo pointer and nsubhalo arrays for the hosts
        graph.create_dataset('subhalo_start_index', data=subhalo_start_index,
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('nsubhalos', data=nsubhalos, dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('host_halos', data=hosts, dtype=np.int32,
                             compression='gzip')

        # Write out the subhalo properties
        # graph.create_dataset('sub_graph_halo_ids', data=np.array(this_graph),
        #                      dtype=np.int32, compression='gzip')
        graph.create_dataset('subhalo_catalog_halo_ids',
                             data=np.array(this_graph_halo_cat_ids),
                             dtype=np.int64,
                             compression='gzip')
        graph.create_dataset('sub_snapshots', data=np.array(snaps),
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('sub_redshifts', data=zs, dtype=float,
                             compression='gzip')
        graph.create_dataset('sub_generation_id', data=np.array(generations),
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('sub_nparts', data=np.array(nparts),
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('sub_mean_pos', data=mean_pos, dtype=float,
                             compression='gzip')
        graph.create_dataset('sub_mean_vel', data=mean_vel, dtype=float,
                             compression='gzip')
        graph.create_dataset('sub_rms_radius', data=rms_rad, dtype=float,
                             compression='gzip')
        graph.create_dataset('sub_3D_velocity_dispersion', data=vdisp,
                             dtype=float,
                             compression='gzip')
        graph.create_dataset('sub_v_max', data=vmax, dtype=float,
                             compression='gzip')
        graph.create_dataset('sub_half_mass_radius', data=hmrs, dtype=float,
                             compression='gzip')
        graph.create_dataset('sub_half_mass_velocity_radius', data=hmvrs,
                             dtype=float,
                             compression='gzip')
        graph.create_dataset('sub_generation_start_index',
                             data=generation_start_index, dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('sub_generation_length', data=generation_length,
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('sub_nprog', data=nprogs, dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('sub_ndesc', data=ndescs, dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('sub_prog_start_index', data=prog_start_index,
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('sub_desc_start_index', data=desc_start_index,
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('sub_direct_prog_ids', data=np.array(progs),
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('sub_direct_desc_ids', data=np.array(descs),
                             dtype=np.int32, compression='gzip')
        graph.create_dataset('sub_direct_prog_contribution',
                             data=np.array(prog_mass_conts), dtype=np.int32,
                             compression='gzip')
        graph.create_dataset('sub_direct_desc_contribution',
                             data=np.array(desc_mass_conts), dtype=np.int32,
                             compression='gzip')

    hdf.create_dataset('sub_graph_lengths', data=np.array(sub_graph_length),
                       dtype=np.int32, compression='gzip')
    hdf.create_dataset('sub_nhalos_in_graph',
                       data=np.array(sub_nhalo_in_graph), dtype=np.int32,
                       compression='gzip')

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
    roots = halo_ids[reals][0:100]

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
    hosts = {}
    mean_pos = {}
    mean_vel = {}
    rms_rad = {}
    zs = {}
    vdisp = {}
    vmax = {}
    hmrs = {}
    hmvrs = {}
    pmass = 0
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

        hdf = h5py.File(halopath + 'halos_' + str(snap) + '.hdf5', 'r')

        zs[snap] = hdf.attrs["redshift"]
        pmass = hdf.attrs["part_mass"]

        hosts[snap] = hdf['Subhalos']['host_IDs'][...]
        mean_pos[snap] = hdf['mean_positions'][...]
        mean_vel[snap] = hdf['mean_velocities'][...]
        rms_rad[snap] = hdf["rms_spatial_radius"][...]
        vdisp[snap] = hdf["3D_velocity_dispersion"][...]
        vmax[snap] = hdf["v_max"][...]
        hmrs[snap] = hdf["half_mass_radius"][...]
        hmvrs[snap] = hdf["half_mass_velocity_radius"][...]

        hdf.close()

    data_dict = {'progs': progs, 'descs': descs, 'nprogs': nprogs,
                 'ndescs': ndescs, 'prog_start_index': prog_start_index,
                 'desc_start_index': desc_start_index, 'nparts': nparts,
                 "hosts": hosts, "mean_pos": mean_pos, "mean_vel": mean_vel,
                 "redshift": zs, "pmass": pmass, "rms_rad": rms_rad,
                 "vdisp": vdisp, "vmax": vmax, "hmrs": hmrs, "hmvrs": hmvrs}

    # Initialise the tuple for storing results
    graphs = []

    for thisTask in myroots:
        result = graph_worker(thisTask, snaplist, verbose, data_dict)
        graphs.append(result)

    collected_results = comm.gather(graphs, root=0)

    if rank == 0:

        graph_dicts = [[] for i in range(size)]
        mass_dicts = [[] for i in range(size)]
        found_roots = set()
        chunked_halo_load = np.zeros(size)

        for lst in collected_results:
            for graph in lst:

                graph_dict, mass_dict = graph

                if graph_dict[root_snap][0] not in found_roots:
                    found_roots.update(graph_dict[root_snap])
                    i = np.argmin(chunked_halo_load)
                    chunked_halo_load[i] += len(graph_dict[root_snap])
                    graph_dicts[i].append(graph_dict)
                    mass_dicts[i].append(mass_dict)

    else:

        graph_dicts = None
        mass_dicts = None

    graph_dicts = comm.scatter(graph_dicts, root=0)

    print("Rank", rank, "has", len(graph_dicts), "graphs")

    # Create file to store this snapshots graph results
    hdf = h5py.File(treepath + 'SubMgraph_' + root_snap + '.hdf5', 'r')

    # Extract the halo IDs (group names/keys) contained within this snapshot and the realness flag
    subhalo_ids = hdf['halo_IDs'][...]
    sub_reals = hdf['real_flag'][...]

    hdf.close()

    # Extract only the real roots
    sub_roots = subhalo_ids[sub_reals]
    sub_hosts = data_dict["hosts"][root_snap][sub_roots]

    # Get the subhalos in roots on this rank
    sub_myroots = []
    for graph in graph_dicts:

        subs = []
        for host in graph[root_snap]:
            subs.extend(sub_roots[sub_hosts == host])

        if len(subs) > 0:
            sub_myroots.append(subs)

    print(sub_roots.size, rank, len(sub_myroots),
          np.sum([len(i) for i in sub_myroots]))

    # Get the start indices, progs, and descs and store them in dictionaries
    progs = {}
    descs = {}
    nprogs = {}
    ndescs = {}
    prog_start_index = {}
    desc_start_index = {}
    nparts = {}
    hosts = {}
    mean_pos = {}
    mean_vel = {}
    rms_rad = {}
    zs = {}
    vdisp = {}
    vmax = {}
    hmrs = {}
    hmvrs = {}
    pmass = 0
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

        zs[snap] = hdf.attrs["redshift"]
        pmass = hdf.attrs["part_mass"]

        hosts[snap] = hdf['Subhalos']['host_IDs'][...]
        mean_pos[snap] = hdf['Subhalos']['mean_positions'][...]
        mean_vel[snap] = hdf['Subhalos']['mean_velocities'][...]
        rms_rad[snap] = hdf['Subhalos']["rms_spatial_radius"][...]
        vdisp[snap] = hdf['Subhalos']["3D_velocity_dispersion"][...]
        vmax[snap] = hdf['Subhalos']["v_max"][...]
        hmrs[snap] = hdf['Subhalos']["half_mass_radius"][...]
        hmvrs[snap] = hdf['Subhalos']["half_mass_velocity_radius"][...]

        hdf.close()

    data_dict["sub"] = {'progs': progs, 'descs': descs, 'nprogs': nprogs,
                        'ndescs': ndescs, 'prog_start_index': prog_start_index,
                        'desc_start_index': desc_start_index, 'nparts': nparts,
                        "hosts": hosts, "mean_pos": mean_pos,
                        "mean_vel": mean_vel, "redshift": zs, "pmass": pmass,
                        "rms_rad": rms_rad, "vdisp": vdisp, "vmax": vmax,
                        "hmrs": hmrs, "hmvrs": hmvrs}

    # Initialise the tuple for storing results
    sub_graphs = []

    for thisTask in sub_myroots:
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
        graph_writer(results, sub_results, graphpath, treepath,
                     past2present_snaplist, data_dict)
