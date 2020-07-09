import numpy as np
from collections import defaultdict
import h5py
import multiprocessing
from multiprocessing import Lock
import multiprocessing as mp
import functools
from functools import partial
from copy import deepcopy
import time
import sys
import pprint
import pickle
import random
import os
import gc
from blessings import Terminal


term = Terminal()


lock = Lock()


def directProgDescFinder(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
                         prog_counts, desc_counts, part_threshold, prog_reals):
    """

    :param current_halo_pids:
    :param prog_snap_haloIDs:
    :param desc_snap_haloIDs:
    :param prog_counts:
    :param desc_counts:
    :param part_threshold:
    :return:
    """

    # =============== Find Progenitor IDs ===============

    # If any progenitor halos exist (i.e. The current snapshot ID is not 000, enforced in the main function)
    if prog_snap_haloIDs.size != 0:

        # Find the halo IDs of the current halo's particles in the progenitor snapshot by indexing the
        # progenitor snapshot's particle halo IDs array with the halo's particle IDs, this can be done
        # since the particle halo IDs array is sorted by particle ID.
        prog_haloids = prog_snap_haloIDs[current_halo_pids]

        # Find the unique halo IDs and the number of times each appears
        uniprog_haloids, uniprog_counts = np.unique(prog_haloids, return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value.
        if uniprog_haloids[0] == -2:
            uniprog_haloids = uniprog_haloids[1:]
            uniprog_counts = uniprog_counts[1:]

        uniprog_haloids = uniprog_haloids[np.where(uniprog_counts >= 10)]
        uniprog_counts = uniprog_counts[np.where(uniprog_counts >= 10)]

        # Remove halos below the Progenitor/Descendant mass threshold (unnecessary with a part_threshold
        # of 10 since the halo finder only returns halos with 10 or more particles)
        if part_threshold > 10:
            uniprog_haloids = uniprog_haloids[np.where(prog_counts[uniprog_haloids] >= part_threshold)]
            uniprog_counts = uniprog_counts[np.where(prog_counts[uniprog_haloids] >= part_threshold)]

        # Get only real halos
        preals = np.array([prog_reals[str(prog)] for prog in uniprog_haloids], copy=False, dtype=bool)
        uniprog_haloids = uniprog_haloids[preals]
        uniprog_counts = uniprog_counts[preals]

        # Find the number of progenitor halos from the size of the unique array
        nprog = uniprog_haloids.size

        # Assign the corresponding number of particles in each progenitor for sorting and storing
        # This can be done simply by using the ID of the progenitor since again np.unique returns
        # sorted results.
        prog_npart = prog_counts[uniprog_haloids]

        # Sort the halo IDs and number of particles in each progenitor halo by their contribution to the
        # current halo (number of particles from the current halo in the progenitor or descendant)
        sorting_inds = uniprog_counts.argsort()[::-1]
        prog_npart = prog_npart[sorting_inds]
        prog_haloids = uniprog_haloids[sorting_inds]
        prog_mass_contribution = uniprog_counts[sorting_inds]

    # If there is no progenitor store Null values
    else:
        nprog = -1
        prog_npart = np.array([-1], copy=False, dtype=int)
        prog_haloids = np.array([-1], copy=False, dtype=int)
        prog_mass_contribution = np.array([-1], copy=False, dtype=int)
        preals = np.array([False], copy=False, dtype=bool)

    # =============== Find Descendant IDs ===============

    # If descendant halos exist (i.e. The current snapshot ID is not 061, enforced in the main function)
    if desc_snap_haloIDs.size != 0:

        # Find the halo IDs of the current halo's particles in the descendant snapshot by indexing the
        # descendant snapshot's particle halo IDs array with the halo's particle IDs, this can be done
        # since the particle halo IDs array is sorted by particle ID.
        desc_haloids = desc_snap_haloIDs[current_halo_pids]

        # Find the unique halo IDs and the number of times each appears
        unidesc_haloids, unidesc_counts = np.unique(desc_haloids, return_counts=True)

        # Remove single particle halos (ID=-2) for the counts, since np.unique returns a sorted array this can be
        # done by removing the first value.
        if unidesc_haloids[0] == -2:
            unidesc_haloids = unidesc_haloids[1:]
            unidesc_counts = unidesc_counts[1:]

        unidesc_haloids = unidesc_haloids[np.where(unidesc_counts >= 10)]
        unidesc_counts = unidesc_counts[np.where(unidesc_counts >= 10)]

        # Remove halos below the Progenitor/Descendant mass threshold (unnecessary with a part_threshold
        # of 10 since the halo finder only returns halos with 10 or more particles)
        if part_threshold > 10:
            unidesc_haloids = unidesc_haloids[np.where(desc_counts[unidesc_haloids] >= part_threshold)]
            unidesc_counts = unidesc_counts[np.where(desc_counts[unidesc_haloids] >= part_threshold)]

        # Find the number of descendant halos from the size of the unique array
        ndesc = unidesc_haloids.size

        # Assign the corresponding number of particles in each descendant for storing.
        # Could be extracted later from halo data but make analysis faster to save it here.
        # This can be done simply by using the ID of the descendant since again np.unique returns
        # sorted results.
        desc_npart = desc_counts[unidesc_haloids]

        # Sort the halo IDs and number of particles in each progenitor halo by their contribution to the
        # current halo (number of particles from the current halo in the progenitor or descendant)
        sorting_inds = unidesc_counts.argsort()[::-1]
        desc_npart = desc_npart[sorting_inds]
        desc_haloids = unidesc_haloids[sorting_inds]
        desc_mass_contribution = unidesc_counts[sorting_inds]

    # If there is no descendant snapshot store Null values
    else:
        ndesc = -1
        desc_npart = np.array([-1], copy=False, dtype=int)
        desc_haloids = np.array([-1], copy=False, dtype=int)
        desc_mass_contribution = np.array([-1], copy=False, dtype=int)

    return (nprog, prog_haloids, prog_npart, prog_mass_contribution,
            ndesc, desc_haloids, desc_npart, desc_mass_contribution,
            preals, current_halo_pids)


def directProgDescWriter(snapshot, halopath='halo_snapshots/', savepath='MergerGraphs/', part_threshold=10, rank=0):
    """ A function which cycles through all halos in a snapshot finding and writing out the
    direct progenitor and descendant data.

    :param snapshot: The snapshot ID.
    :param halopath: The filepath to the halo finder HDF5 file.
    :param savepath: The filepath to the directory where the Merger Graph should be written out to.
    :param part_threshold: The mass (number of particles) threshold defining a halo.

    :return: None
    """

    # =============== Current Snapshot ===============

    # Load the current snapshot data
    hdf_current = h5py.File(halopath + 'halos_' + snapshot + '.hdf5', 'r')

    # Extract the halo IDs (group names/keys) contained within this snapshot
    halo_ids = np.unique(hdf_current['Halo_IDs'][...][:, rank])
    halo_ids = np.array(halo_ids[np.where(halo_ids >= 0)], dtype=str)

    hdf_current.close()  # close the root group to reduce overhead when looping

    # =============== Progenitor Snapshot ===============

    # Define the progenitor snapshot ID (current ID - 1)
    if int(snapshot) > 10:
        prog_snap = '0' + str(int(snapshot) - 1)
    else:
        prog_snap = '00' + str(int(snapshot) - 1)

    # Only look for progenitor data if there is a progenitor snapshot (subtracting 1 from the
    # earliest snapshot results in a snapshot ID of '00-1')
    if prog_snap != '00-1':

        # Load the progenitor snapshot
        hdf_prog = h5py.File(halopath + 'halos_' + prog_snap + '.hdf5', 'r')

        # Extract the particle halo ID array and particle ID array
        prog_snap_haloIDs = hdf_prog['Halo_IDs'][...][:, rank]

        # Get all the unique halo IDs in this snapshot and the number of times they appear
        prog_unique, prog_counts = np.unique(prog_snap_haloIDs, return_counts=True)

        # Make sure all halos at this point in the data have more than ten particles
        assert all(prog_counts >= 10), 'Not all halos are large than the minimum mass threshold'

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        prog_unique = prog_unique[1:]
        prog_counts = prog_counts[1:]

        # Get realness of each halo
        prog_reals = {}
        for halo in prog_unique:
            prog_reals[str(halo)] = hdf_prog[str(halo)].attrs['Real']
        hdf_prog.close()

    else:  # Assign an empty array if the snapshot is less than the earliest (000)
        prog_snap_haloIDs = np.array([], copy=False)
        prog_counts = []
        prog_reals = {}

    # =============== Descendant Snapshot ===============

    # Define the descendant snapshot ID (current ID + 1)
    if int(snapshot) > 8:
        desc_snap = '0' + str(int(snapshot) + 1)
    else:
        desc_snap = '00' + str(int(snapshot) + 1)

    # Only look for descendant data if there is a descendant snapshot (last snapshot has the ID '061')
    if int(desc_snap) <= 61:

        # Load the descendant snapshot
        hdf_desc = h5py.File(halopath + 'halos_' + desc_snap + '.hdf5', 'r')

        # Extract the particle -> halo ID array and particle ID array
        desc_snap_haloIDs = hdf_desc['Halo_IDs'][...][:, rank]
        hdf_desc.close()

        # Get all unique halos in this snapshot
        desc_unique, desc_counts = np.unique(desc_snap_haloIDs, return_counts=True)

        # Make sure all halos at this point in the data have more than ten particles
        assert all(desc_counts >= 10), 'Not all halos are large than the mass threshold'

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        desc_unique = desc_unique[1:]
        desc_counts = desc_counts[1:]

    else:  # Assign an empty array if the snapshot is less than the earliest (000)
        desc_snap_haloIDs = np.array([], copy=False)
        desc_counts = []

    # =============== Find all Direct Progenitors And Descendant Of Halos In This Snapshot ===============

    # Initialise the progress
    progress = -1

    # Assign the number of halos for progress reporting
    size = len(halo_ids)
    imreal = 0
    results = {}

    # Loop through all the halos in this snapshot
    for num, haloID in enumerate(halo_ids):

        # Print progress
        previous_progress = progress
        progress = int(num / size * 100)
        if progress != previous_progress:
            print('Graph progress: ', progress, '%', haloID, end='\r')

        # =============== Current Halo ===============

        # Load the snapshot data
        hdf_current = h5py.File(halopath + 'halos_' + snapshot + '.hdf5', 'r')

        # Assign the particle IDs contained in the current halo
        current_halo_pids = hdf_current[str(haloID)]['Halo_Part_IDs'][...]
        real = hdf_current[str(haloID)].attrs['Real']

        hdf_current.close()  # close the root group to reduce overhead when looping

        # If halo falls below the 20 cutoff threshold do not look for progenitors or descendants
        if current_halo_pids.size >= part_threshold and real:

            # =============== Run The Direct Progenitor and Descendant Finder ===============

            # Run the progenitor/descendant finder

            result = directProgDescFinder(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
                                          prog_counts, desc_counts, part_threshold, prog_reals)
            (nprog, prog_haloids, prog_npart, prog_mass_contribution,
             ndesc, desc_haloids, desc_npart, desc_mass_contribution, preals, current_halo_pids) = result

            # If the halo has neither descendants or progenitors we do not need to store it
            if nprog == ndesc == -1 or nprog == ndesc == 0:

                # Load the snapshot data
                hdf_current = h5py.File(halopath + 'halos_' + snapshot + '.hdf5', 'r+')

                real = False
                hdf_current[str(haloID)].attrs['Real'] = real

                hdf_current.close()  # close the root group to reduce overhead when looping
                continue

            # If progenitor is real and this halo is not then it is real
            elif True in preals and not real:

                # Load the snapshot data
                hdf_current = h5py.File(halopath + 'halos_' + snapshot + '.hdf5', 'r+')

                real = True
                hdf_current[str(haloID)].attrs['Real'] = real

                hdf_current.close()  # close the root group to reduce overhead when looping

            # If this halo has no real progenitors and is less than 20 particle it is by definition not
            # a halo
            elif nprog == 0 and current_halo_pids.size < 20:

                # Load the snapshot data
                hdf_current = h5py.File(halopath + 'halos_' + snapshot + '.hdf5', 'r+')

                real = False
                hdf_current[str(haloID)].attrs['Real'] = real

                hdf_current.close()  # close the root group to reduce overhead when looping
                continue

            # If this halo is real then it's descendents are real
            if real and int(desc_snap) < 62:

                # Load the descendant snapshot
                hdf_desc = h5py.File(halopath + 'halos_' + desc_snap + '.hdf5', 'r+')

                # Loop over decendents
                for desc in desc_haloids:
                    hdf_desc[str(desc)].attrs['Real'] = True

                hdf_desc.close()

            # =============== Write Out Data ===============

            imreal += 1

            results[str(haloID)] = result
    if rank == 0:

        hdf = h5py.File(savepath + 'Mgraph_' + snapshot + '.hdf5', 'w')

    else:

        hdf = h5py.File(savepath + 'SubMgraph_' + snapshot + '.hdf5', 'w')

    for num, haloID in enumerate(results.keys()):

        (nprog, prog_haloids, prog_npart, prog_mass_contribution,
         ndesc, desc_haloids, desc_npart, desc_mass_contribution, preals, current_halo_pids) = results[haloID]

        # Print progress
        previous_progress = progress
        progress = int(num / size * 100)
        if progress != previous_progress:
            print('Write progress: ', progress, '%', haloID, end='\r')

        # Write out the data produced
        halo = hdf.create_group(haloID)  # create halo group
        halo.attrs['nProg'] = nprog  # number of progenitors
        halo.attrs['nDesc'] = ndesc  # number of descendants
        halo.attrs['current_halo_nPart'] = current_halo_pids.size  # mass of the halo
        halo.create_dataset('current_halo_partIDs', data=current_halo_pids, dtype=int,
                            compression='gzip')  # particle ids in this halo
        halo.create_dataset('prog_mass_contribution', data=prog_mass_contribution, dtype=int,
                            compression='gzip')  # Mass contribution
        halo.create_dataset('desc_mass_contribution', data=desc_mass_contribution, dtype=int,
                            compression='gzip')  # Mass contribution
        halo.create_dataset('Prog_nPart', data=prog_npart, dtype=int,
                            compression='gzip')  # number of particles in each progenitor
        halo.create_dataset('Desc_nPart', data=desc_npart, dtype=int,
                            compression='gzip')  # number of particles in each descendant
        halo.create_dataset('Prog_haloIDs', data=prog_haloids, dtype=int,
                            compression='gzip')  # progenitor IDs
        halo.create_dataset('Desc_haloIDs', data=desc_haloids, dtype=int,
                            compression='gzip')  # descendant IDs

    hdf.close()

    print(snapshot, 'Real=', imreal, 'All=', len(halo_ids), 'Not Real=', len(halo_ids) - imreal)

    return


def get_forest(z0halo, treepath):
    """ A funciton which traverses a tree including all halos which have interacted with the tree
    in a 'forest/mangrove'.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param z0halo: The halo ID of a z=0 halo for which the forest/mangrove plot is desired.

    :return: forest_dict: The dictionary containing the forest. Each key is the snapshot ID and
             the value is a list of halos in this snapshot of the forest.
             massgrowth: The mass history of the forest.
             tree: The dictionary containing the tree. Each key is the snapshot ID and
             the value is a list of halos in this snapshot of the tree.
             main_growth: The mass history of the main branch.
    """

    # Initialise dictionary instances
    forest_dict = defaultdict(set)
    mass_dict = {}

    # Create snapshot list in reverse order (present day to past) for the progenitor searching loop
    snaplist = []
    for snap in range(61, 0, -1):
        if snap < 10:
            snaplist.append('00' + str(snap))
        elif snap >= 10:
            snaplist.append('0' + str(snap))

    # Initialise the halo's set for tree walking
    halos = {int(z0halo) + (61 * 1000000)}

    # Initialise the forest dictionary with the present day halo as the first entry
    forest_dict['061'] = halos

    # Initialise the set of new found halos used to loop until no new halos are found
    new_halos = halos

    # Initialise the set of found halos used to check if the halo has been found or not
    found_halos = set()

    # Loop until no new halos are found
    while len(new_halos) != 0:

        # Overwrite the last set of new_halos
        new_halos = set()

        # =============== Progenitors ===============

        # Loop over snapshots and progenitor snapshots
        for prog_snap, snap in zip(snaplist[1:], snaplist[:-1]):

            # Assign the halos variable for the next stage of the tree
            halos = forest_dict[snap]

            # Loop over halos in this snapshot
            for halo in halos:

                # Remove snapshot ID from halo ID
                halo -= (int(snap) * 1000000)

                # Open this snapshots root group
                snap_tree_data = h5py.File(treepath + snap + '.hdf5', 'r')

                # Assign progenitors adding the snapshot * 100000 to the ID to keep track of the snapshot ID
                # in addition to the halo ID
                forest_dict.setdefault(prog_snap, set()).update(set((int(prog_snap) * 1000000) +
                                                                    snap_tree_data[str(halo)]['Prog_haloIDs'][...]))
                snap_tree_data.close()

            # Add any new halos not found in found halos to the new halos set
            new_halos.update(forest_dict[prog_snap] - found_halos)

        # =============== Descendants ===============

        # Loop over halos found during the progenitor step
        snapshots = list(reversed(list(forest_dict.keys())))
        for desc_snap, snap in zip(snapshots[1:], snapshots[:-1]):

            # Assign the halos variable for the next stage of the tree
            halos = forest_dict[snap]

            # Loop over the progenitor halos
            for halo in halos:

                # Remove snapshot ID from halo ID
                halo -= (int(snap) * 1000000)

                # Open this snapshots root group
                snap_tree_data = h5py.File(treepath + snap + '.hdf5', 'r')

                # Load descendants adding the snapshot * 100000 to keep track of the snapshot ID
                # in addition to the halo ID
                forest_dict.setdefault(desc_snap, set()).update(set((int(desc_snap) * 1000000) +
                                                                    snap_tree_data[str(halo)]['Desc_haloIDs'][...]))

                snap_tree_data.close()

            # Redefine the new halos set to have any new halos not found in found halos
            new_halos.update(forest_dict[desc_snap] - found_halos)

        # Add the new_halos to the found halos set
        found_halos.update(new_halos)

    forest_snaps = list(forest_dict.keys())

    for snap in forest_snaps:

        try:
            # Open this snapshots root group
            snap_tree_data = h5py.File(treepath + snap + '.hdf5', 'r')
        except OSError:
            del forest_dict[snap]
            continue

        if len(forest_dict[snap]) == 0:
            del forest_dict[snap]
            continue

        forest_dict[snap] = np.array(list(forest_dict[snap])) - (int(snap) * 1000000)
        mass_dict[snap] = np.array([snap_tree_data[str(halo)].attrs['current_halo_nPart']
                                    for halo in forest_dict[snap]])

        snap_tree_data.close()

    return forest_dict,  mass_dict


def forest_worker(z0halo, treepath):

    lock.acquire()
    with open('roothalos.pck', 'rb') as file:
        roots = pickle.load(file)
        print(len(roots) - 1, 'Roots')
    lock.release()
    if int(z0halo) in roots:
        print('Halo ' + str(z0halo) + '\'s Forest exists...')
        return {}

    # Get the forest with this halo at it's root
    forest_dict, mass_dict = get_forest(z0halo, treepath)

    print('Halo ' + str(z0halo) + '\'s Forest extracted...')

    lock.acquire()
    with open('roothalos.pck', 'rb') as file:
        roots = pickle.load(file)
    for root in forest_dict['061']:
        roots.update([root])
    with open('roothalos.pck', 'wb') as file:
        pickle.dump(roots, file)
    lock.release()

    return forest_dict, mass_dict


def forest_writer(forests, graphpath):

    snap_tree_data = h5py.File(graphpath + '.hdf5', 'w')

    for forest in forests:

        if len(forest) == 0:
            continue

        forest_dict, mass_dict = forest

        # Get highest mass root halo
        z0halos = np.array(list(forest_dict['061']))
        z0masses = mass_dict['061']

        # IDs in this generation of this graph
        z0halo = z0halos[np.argmax(z0masses)]

        if str(z0halo) in snap_tree_data.keys():
            print(str(z0halo) + '\'s Forest already exists')
            continue

        print('Creating Group', str(z0halo) + '...', end='\r')

        graph = snap_tree_data.create_group(str(z0halo))  # create halo group

        for snap in forest_dict.keys():

            # Get the halos in this generation
            this_gen_halos = np.array(list(forest_dict[snap]))

            if this_gen_halos.size == 0:
                continue

            # Write out the data to the halo
            this_gen_halos_lst = []
            for halo in this_gen_halos:

                this_gen_halos_lst.append(halo)

            # IDs in this generation of this graph
            graph.create_dataset(snap, data=np.array(this_gen_halos_lst), dtype=int, compression='gzip')

    snap_tree_data.close()


def get_graph_members(treepath='MergerGraphs/Mgraph_', graphpath='MergerGraphs/FullMgraphs',
                      halopath='halo_snapshots/halos_'):

    # Open the present day snapshot
    snap_tree_data = h5py.File(treepath + '061.hdf5', 'r', driver='core')

    # Get the haloIDs for the halos at the present day
    z0halos = list(snap_tree_data.keys())

    snap_tree_data.close()

    roots = {-999}

    with open('roothalos.pck', 'wb') as file:
        pickle.dump(roots, file)

    # Remove not real roots
    halo_hdf = h5py.File(halopath + '061.hdf5', 'r')
    roots = []
    for halo in z0halos:

        if not halo_hdf[halo].attrs['Real'] or halo_hdf[halo].attrs['halo_nPart'] < 20:
            continue

        print('Including root', halo, end='\r')

        roots.append(halo)

    halo_hdf.close()

    p = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2)
    graphs = p.map(partial(forest_worker, treepath=treepath), iter(roots))
    p.close()
    p.join()

    # Write out the result
    forest_writer(graphs, graphpath)


def notreal_extract_writer(arg, halopath, treepath):

    prog_snap, snap, desc_snap = arg

    try:
        tree_hdf = h5py.File(treepath + snap + '.hdf5', 'r+')
    except OSError:
        print(f'No halos in {snap}')
        return
    try:
        desc_hdf = h5py.File(halopath + desc_snap + '.hdf5', 'r')
    except OSError:
        desc_hdf = None

    prog_hdf = h5py.File(halopath + prog_snap + '.hdf5', 'r')
    hdf = h5py.File(halopath + snap + '.hdf5', 'r')

    # Loop over halos
    for halo in tree_hdf.keys():

        print(f'Correcting {snap} {halo}...', end='\r')

        # Get progenitor and descendent data
        progs = tree_hdf[halo]['Prog_haloIDs'][...]
        old_nprog = tree_hdf[halo].attrs['nProg']
        prog_reals = np.array([prog_hdf[str(prog)].attrs['Real'] for prog in progs], copy=False, dtype=bool)
        progs = progs[prog_reals]
        prog_masses = tree_hdf[halo]['Prog_nPart'][...][prog_reals]
        prog_conts = tree_hdf[halo]['prog_mass_contribution'][...][prog_reals]
        nprog = len(progs)
        if desc_hdf != None:
            descs = tree_hdf[halo]['Desc_haloIDs'][...]
            old_ndesc = tree_hdf[halo].attrs['nDesc']
            desc_reals = np.array([desc_hdf[str(desc)].attrs['Real'] for desc in descs], copy=False, dtype=bool)
            descs = descs[desc_reals]
            desc_masses = tree_hdf[halo]['Desc_nPart'][...][desc_reals]
            desc_conts = tree_hdf[halo]['desc_mass_contribution'][...][desc_reals]
            ndesc = len(descs)
        else:
            ndesc = -1
            old_ndesc = -1
            desc_masses = np.array([-1], copy=False, dtype=int)
            descs = np.array([-1], copy=False, dtype=int)
            desc_conts = np.array([-1], copy=False, dtype=int)

        if old_ndesc == ndesc and old_nprog == nprog:
            continue

        current_halo_pids = hdf[halo]['Halo_Part_IDs'][...]
        hmass = hdf[halo].attrs['halo_nPart']

        del tree_hdf[halo]

        # Write out the data produced
        ghalo = tree_hdf.create_group(halo)  # create halo group
        ghalo.attrs['nProg'] = nprog  # number of progenitors
        ghalo.attrs['nDesc'] = ndesc  # number of descendants
        ghalo.attrs['current_halo_nPart'] = hmass  # mass of the halo
        ghalo.create_dataset('current_halo_partIDs', data=current_halo_pids, dtype=int,
                             compression='gzip')  # particle ids in this halo
        ghalo.create_dataset('prog_mass_contribution', data=prog_conts, dtype=int,
                             compression='gzip')  # Mass contribution
        ghalo.create_dataset('desc_mass_contribution', data=desc_conts, dtype=int,
                             compression='gzip')  # Mass contribution
        ghalo.create_dataset('Prog_nPart', data=prog_masses, dtype=int,
                             compression='gzip')  # number of particles in each progenitor
        ghalo.create_dataset('Desc_nPart', data=desc_masses, dtype=int,
                             compression='gzip')  # number of particles in each descendant
        ghalo.create_dataset('Prog_haloIDs', data=progs, dtype=int,
                             compression='gzip')  # progenitor IDs
        ghalo.create_dataset('Desc_haloIDs', data=descs, dtype=int,
                             compression='gzip')  # descendant IDs

    prog_hdf.close()
    tree_hdf.close()
    hdf.close()
    if desc_hdf != None:
        desc_hdf.close()


def notreal_extract(treepath='MergerGraphs/Mgraph_', halopath='halo_snapshots/halos_'):

    # Create a snapshot list (past to present day) for looping
    snaplist = []
    for snap in range(0, 62):
        if snap < 10:
            snaplist.append('00' + str(snap))
        elif snap >= 10:
            snaplist.append('0' + str(snap))

    # Loop over snapshots correcting the reality flag based on whether a halo has genuine progs
    for prog_snap, snap in zip(snaplist[:-1], snaplist[1:]):

        try:
            tree_hdf = h5py.File(treepath + snap + '.hdf5', 'r')
        except OSError:
            print(f'No halos in {snap}')
            continue

        prog_hdf = h5py.File(halopath + prog_snap + '.hdf5', 'r')
        hdf = h5py.File(halopath + snap + '.hdf5', 'r+')

        # Loop over halos
        for halo in tree_hdf.keys():

            print(f'Checking halo {snap} {halo}', end='\r')

            if hdf[halo].attrs['halo_energy'] < 0:
                continue

            print(snap, halo, end='\r')

            # Get progenitor and descendent data
            nprog = tree_hdf[halo].attrs['nProg']

            if nprog == 0 or nprog == -1:
                real = False
            else:
                progs = tree_hdf[halo]['Prog_haloIDs'][...]
                prog_reals = {prog_hdf[str(prog)].attrs['Real'] for prog in progs}

                # If there is a real progenitor then this halo is real
                if True in prog_reals:
                    real = True
                else:
                    real = False

            print(snap, halo, 'is real:', real)

            hdf[halo].attrs['Real'] = real

        prog_hdf.close()
        tree_hdf.close()
        hdf.close()

    # # Create a snapshot list (past to present day) for looping
    snaplist = []
    for snap in range(0, 63):
        if snap < 10:
            snaplist.append('00' + str(snap))
        elif snap >= 10:
            snaplist.append('0' + str(snap))

    pool = mp.Pool(int(mp.cpu_count() - 2))
    pool.map(partial(notreal_extract_writer, treepath=treepath, halopath=halopath),
             zip(snaplist[:-2], snaplist[1:-1], snaplist[2:]))
    pool.close()
    pool.join()


def notreal_machete_worker(snap, treepath, halopath, real_halos):

    try:
        tree_hdf = h5py.File(treepath + snap + '.hdf5', 'r+')
    except OSError:
        print(f'No halos in {snap}')
        return 0

    tree_ids = real_halos[snap]

    hdf = h5py.File(halopath + snap + '.hdf5', 'r+', driver='core')

    halo_ids = np.unique(hdf['Halo_IDs'][...][:, 0])
    halo_ids = set(halo_ids[np.where(halo_ids >= 0)])

    halos_to_delete = halo_ids - tree_ids

    print(len(halos_to_delete), 'to remove in snapshot', snap)

    # Loop over halos
    for halo in halos_to_delete:
        print(f'Correcting halo catalogue {snap} {halo}...', end='\r')
        try:
            del hdf[str(halo)]
            del tree_hdf[str(halo)]
        except KeyError:
            continue

    gc.collect()

    hdf.close()
    tree_hdf.close()

    return len(halos_to_delete)


def notreal_machete(halopath, treepath, graphpath):

    # # Create a snapshot list (past to present day) for looping
    snaplist = []
    for snap in range(0, 62):
        if snap < 10:
            snaplist.append('00' + str(snap))
        elif snap >= 10:
            snaplist.append('0' + str(snap))

    graphs = h5py.File(graphpath + '.hdf5', 'r')
    real_halos = {}
    for root in graphs.keys():
        for snap in graphs[root].keys():
            print(snap, root, end='\r')
            real_halos.setdefault(snap, set()).update(graphs[root][snap][...])

    # pool = mp.Pool(int(mp.cpu_count() - 2))
    # results = pool.map(partial(notreal_machete_worker, treepath=treepath, halopath=halopath,
    #                            real_halos=real_halos), snaplist)
    # pool.close()
    # pool.join()
    results = []
    for snap in real_halos.keys():
        results.append(notreal_machete_worker(snap, treepath, halopath, real_halos))

    print('The total number of halos deleteed is:', np.sum(results))


# def remove_unlinked_halos_worker(snap, treepath, halopath):
#
#     try:
#         tree_hdf = h5py.File(treepath + snap + '.hdf5', 'r+', driver='core')
#     except OSError:
#         print(treepath + snap + '.hdf5', 'does not exist...')
#         return
#     halo_hdf = h5py.File(halopath + snap + '.hdf5', 'r+', driver='core')
#
#     count = 0
#     previous_progress = -1
#     size = len(tree_hdf.keys())
#
#     for halo in tree_hdf.keys():
#
#         # Compute and print the progress
#         progress = int(count / size * 100)
#         if progress != previous_progress:  # only print if the integer progress differs from the last printed value
#             print('Removing Unnecessary Not Real Halos... ' + snap + ' %.2f' % progress + '%')
#         previous_progress = progress
#         count += 1
#
#         # Delete this halo if it is not real and has no progenitors or descendents
#         if not halo_hdf[halo].attrs['Real'] and tree_hdf[halo].attrs['nProg'] < 1 \
#                 and tree_hdf[halo].attrs['nDesc'] < 1:
#             del tree_hdf[halo]
#
#     tree_hdf.close()
#     halo_hdf.close()
#
#
# def remove_unlinked_halos(treepath='MergerGraphs/Mgraph_', halopath='halo_snapshots/halos_'):
#
#     # Create a snapshot list (past to present day) for looping
#     snaplist = []
#     for snap in range(61, -1, -1):
#         # if snap % 2 != 0: continue
#         if snap < 10:
#             snaplist.append('00' + str(snap))
#         elif snap >= 10:
#             snaplist.append('0' + str(snap))
#
#     pool = mp.Pool(int(mp.cpu_count() - 2))
#     pool.map(partial(remove_unlinked_halos_worker, treepath=treepath, halopath=halopath), snaplist)
#     pool.close()
#     pool.join()


def link_cutter_worker(snap, treepath):

    # Compute the progenitor snapshot ID
    if int(snap) > 10:
        prog_snap = '0' + str(int(snap) - 1)
    else:
        prog_snap = '00' + str(int(snap) - 1)

    # Compute the progenitor snapshot ID
    if int(snap) + 1 > 10:
        desc_snap = '0' + str(int(snap) + 1)
    else:
        desc_snap = '00' + str(int(snap) + 1)

    try:
        # Open the snapshot root group
        prog_hdf = h5py.File(treepath + prog_snap + '.hdf5', 'r')
    except OSError:
        print(prog_snap, 'does not exist...')
        return

    try:
        # Open the snapshot root group
        desc_hdf = h5py.File(treepath + desc_snap + '.hdf5', 'r')
        desc_hdf.close()
    except OSError:
        print(desc_snap, 'does not exist...')
        desc_hdf = None
    
    # Open the snapshot root group
    hdf = h5py.File(treepath + snap + '.hdf5', 'r')

    previous_progress = -1
    count = 0
    size = len(hdf.keys())
    halos = list(hdf.keys())

    hdf.close()
    prog_hdf.close()

    # Loop over halos in this snapshot
    for halo in halos:

        # Compute and print the progress
        progress = int(count/size*100)
        if progress != previous_progress:  # only print if the integer progress differs from the last printed value
            print('Breaking Incorrect Links... ' + snap + ' %.2f' % progress + '%')
        previous_progress = progress
        count += 1

        hdf = h5py.File(treepath + snap + '.hdf5', 'r')

        # Get the old progs
        progs = hdf[halo]['Prog_haloIDs'][...]
        prog_masses = hdf[halo]['Prog_nPart'][...]
        prog_conts = hdf[halo]['prog_mass_contribution'][...]
        nProgs = len(progs)

        hdf.close()

        if nProgs != 0 and np.min(prog_masses) < 20:

            prog_hdf = h5py.File(treepath + prog_snap + '.hdf5', 'r')

            # Loop over progenitors
            new_progs = np.zeros_like(progs)
            new_prog_masses = np.zeros_like(progs)
            new_progs_cont = np.zeros_like(progs)
            for ind, (prog, mass, cont) in enumerate(zip(progs, prog_masses, prog_conts)):

                if halo in list(prog_hdf.keys()):
                    new_progs[ind] = prog
                    new_prog_masses[ind] = mass
                    new_progs_cont[ind] = cont
                else:
                    new_progs[ind] = -999
                    new_prog_masses[ind] = -999
                    new_progs_cont[ind] = -999

            prog_hdf.close()

            if -999 in new_progs:

                hdf = h5py.File(treepath + snap + '.hdf5', 'r+')

                del hdf[halo]['Prog_haloIDs']
                del hdf[halo]['Prog_nPart']
                del hdf[halo]['prog_mass_contribution']

                prog_haloids = new_progs[np.where(new_progs != -999)]
                prog_npart = new_prog_masses[np.where(new_progs != -999)]
                prog_mass_contribution = new_progs_cont[np.where(new_progs != -999)]
                new_nprogs = len(prog_haloids)

                halohdf = hdf[halo]
                halohdf.create_dataset('prog_mass_contribution', data=prog_mass_contribution, dtype=int,
                                       compression='gzip')  # Mass contribution
                halohdf.create_dataset('Prog_nPart', data=prog_npart, dtype=int,
                                       compression='gzip')  # number of particles in each progenitor
                halohdf.create_dataset('Prog_haloIDs', data=prog_haloids, dtype=int,
                                       compression='gzip')  # progenitor IDs
                halohdf.attrs['nProg'] = new_nprogs

                hdf.close()

        if desc_hdf != None:

            hdf = h5py.File(treepath + snap + '.hdf5', 'r')

            # Get the old descs
            descs = hdf[halo]['Desc_haloIDs'][...]
            desc_masses = hdf[halo]['Desc_nPart'][...]
            desc_conts = hdf[halo]['desc_mass_contribution'][...]
            nDescs = len(descs)

            hdf.close()

            if nDescs != 0 and np.min(desc_masses) < 20:

                desc_hdf = h5py.File(treepath + desc_snap + '.hdf5', 'r')

                # Loop over descenitors
                new_descs = np.zeros_like(descs)
                new_desc_masses = np.zeros_like(descs)
                new_descs_cont = np.zeros_like(descs)
                for ind, (desc, mass, cont) in enumerate(zip(descs, desc_masses, desc_conts)):

                    if halo in list(desc_hdf.keys()):
                        new_descs[ind] = desc
                        new_desc_masses[ind] = mass
                        new_descs_cont[ind] = cont
                    else:
                        new_descs[ind] = -999
                        new_desc_masses[ind] = -999
                        new_descs_cont[ind] = -999

                desc_hdf.close()

                if -999 in new_descs:

                    hdf = h5py.File(treepath + snap + '.hdf5', 'r+')

                    del hdf[halo]['Desc_haloIDs']
                    del hdf[halo]['Desc_nPart']
                    del hdf[halo]['desc_mass_contribution']

                    desc_haloids = new_descs[np.where(new_descs != -999)]
                    desc_npart = new_desc_masses[np.where(new_descs != -999)]
                    desc_mass_contribution = new_descs_cont[np.where(new_descs != -999)]
                    new_ndescs = len(desc_haloids)

                    halohdf = hdf[halo]
                    halohdf.create_dataset('desc_mass_contribution', data=desc_mass_contribution, dtype=int,
                                           compression='gzip')  # Mass contribution
                    halohdf.create_dataset('Desc_nPart', data=desc_npart, dtype=int,
                                           compression='gzip')  # number of particles in each descenitor
                    halohdf.create_dataset('Desc_haloIDs', data=desc_haloids, dtype=int,
                                           compression='gzip')  # descenitor IDs
                    halohdf.attrs['nDesc'] = new_ndescs

                    hdf.close()


def link_cutter(treepath='MergerGraphs/Mgraph_'):

    # Create a snapshot list (past to present day) for looping
    even_snaplist = []
    for snap in range(0, 62):
        if snap % 2 != 0:
            continue
        if snap < 10:
            even_snaplist.append('00' + str(snap))
        elif snap >= 10:
            even_snaplist.append('0' + str(snap))

    pool = mp.Pool(int(mp.cpu_count() - 5))
    pool.map(partial(link_cutter_worker, treepath=treepath), even_snaplist)
    pool.close()
    pool.join()

    # Create a snapshot list (past to present day) for looping
    odd_snaplist = []
    for snap in range(0, 62):
        if snap % 2 == 0:
            continue
        if snap < 10:
            odd_snaplist.append('00' + str(snap))
        elif snap >= 10:
            odd_snaplist.append('0' + str(snap))

    pool = mp.Pool(int(mp.cpu_count() - 5))
    pool.map(partial(link_cutter_worker, treepath=treepath), odd_snaplist)
    pool.close()
    pool.join()


# if __name__ == '__main__':
#     count = 0
#     while 'halos_061.hdf5' not in os.listdir('halo_snapshots/'):
#         time.sleep(10)
#         print(count, end='\r')
#         count += 1


def main(snap):
     directProgDescWriter(snap, halopath='halo_snapshots/', savepath='MergerGraphs/', part_threshold=10,
                          rank=1)


# Create a snapshot list (past to present day) for looping
snaplist = []
for snap in range(0, 62):
    # if snap % 2 != 0: continue
    if snap < 10:
        snaplist.append('00' + str(snap))
    elif snap >= 10:
        snaplist.append('0' + str(snap))

if __name__ == '__main__':
    start = time.time()
    for snap in snaplist:
        print(snap)
        main(snap)
    print('Total:', time.time() - start)

# if __name__ == '__main__':
    # notreal_extract(treepath='MergerGraphs/Mgraph_', halopath='halo_snapshots_spatial/halos_')
    # link_cutter(treepath='MergerGraphs/Mgraph_')
    # remove_unlinked_halos(treepath='MergerGraphs/Mgraph_', halopath='halo_snapshots_spatial/halos_')
    # get_graph_members(treepath='MergerGraphs_sub/Mgraph_', graphpath='MergerGraphs_sub/FullMgraphs',
    #                   halopath='halo_snapshots_sub/halos_')
