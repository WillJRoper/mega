import numpy as np
import h5py
import pprint
from collections import defaultdict
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Lock
import multiprocessing as mp
from functools import partial
import multiprocessing as mp
import time
import sys
import pickle
import os
import seaborn as sns


sns.set_style('whitegrid')


def directProgDescFinder(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
                         prog_counts, desc_counts, part_threshold, prog_reals, current_npart, persist_npart):
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

        # Remove single particle halos (ID=-2) for the counts, since np.unique returns a sorted array this can be
        # done by removing the first value.
        uniprog_counts = uniprog_counts[np.where(uniprog_haloids >= 0)]
        uniprog_haloids = uniprog_haloids[np.where(uniprog_haloids >= 0)]

        uniprog_haloids = uniprog_haloids[np.where(uniprog_counts >= 10)]
        uniprog_counts = uniprog_counts[np.where(uniprog_counts >= 10)]

        # Remove halos below the Progenitor/Descendant mass threshold (unnecessary with a part_threshold
        # of 10 since the halo finder only returns halos with 10 or more particles)
        if part_threshold > 10:
            uniprog_counts = uniprog_counts[np.where(prog_counts[uniprog_haloids] >= part_threshold)]
            uniprog_haloids = uniprog_haloids[np.where(prog_counts[uniprog_haloids] >= part_threshold)]

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
        unidesc_counts = unidesc_counts[np.where(unidesc_haloids >= 0)]
        unidesc_haloids = unidesc_haloids[np.where(unidesc_haloids >= 0)]

        unidesc_haloids = unidesc_haloids[np.where(unidesc_counts >= 10)]
        unidesc_counts = unidesc_counts[np.where(unidesc_counts >= 10)]

        # Remove halos below the Progenitor/Descendant mass threshold (unnecessary with a part_threshold
        # of 10 since the halo finder only returns halos with 10 or more particles)
        if part_threshold > 10:
            unidesc_counts = unidesc_counts[np.where(desc_counts[unidesc_haloids] >= part_threshold)]
            unidesc_haloids = unidesc_haloids[np.where(desc_counts[unidesc_haloids] >= part_threshold)]

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

        # Extract the most massive descendant in the same way SMT13 algorithms do

        # Use the merit function to denote the main descendant defined as the maximum of the merit function
        if len(unidesc_haloids) > 1:
            merit = desc_mass_contribution
            genuine_descs = np.argmax(merit)
            desc_npart = np.array([desc_npart[genuine_descs]], copy=False, dtype=int)
            desc_haloids = np.array([desc_haloids[genuine_descs]], copy=False, dtype=int)
            desc_mass_contribution = np.array([desc_mass_contribution[genuine_descs]], copy=False, dtype=int)
            ndesc = 1

        elif len(unidesc_haloids) == 1:
            pass

            # This is effectively doing this:
            # desc_npart = desc_npart
            # desc_haloids = desc_haloids
            # desc_mass_contribution = desc_mass_contribution
            # ndesc = 1

        else:
            ndesc = 0
            desc_npart = np.array([], copy=False, dtype=int)
            desc_haloids = np.array([], copy=False, dtype=int)
            desc_mass_contribution = np.array([], copy=False, dtype=int)

    # If there are no descendant store Null values
    else:
        ndesc = -1
        desc_npart = np.array([-1], copy=False, dtype=int)
        desc_haloids = np.array([-1], copy=False, dtype=int)
        desc_mass_contribution = np.array([-1], copy=False, dtype=int)

    return (nprog, prog_haloids, prog_npart, prog_mass_contribution,
            ndesc, desc_haloids, desc_npart, desc_mass_contribution,
            preals, current_halo_pids, current_npart, persist_npart)


def directProgDescWriter(snapshot, halopath='halo_snapshots/', savepath='MergerTrees/', part_threshold=10):
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
    if len(hdf_current['Halo_IDs'][...].shape) == 2:
        halo_ids = np.unique(hdf_current['Halo_IDs'][...])
    else:
        halo_ids = np.unique(hdf_current['Halo_IDs'][...])
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
        prog_snap_haloIDs = hdf_prog['Halo_IDs'][...]

        # Get all the unique halo IDs in this snapshot and the number of times they appear
        prog_unique, prog_counts = np.unique(prog_snap_haloIDs, return_counts=True)

        # # Make sure all halos at this point in the data have more than ten particles
        # assert all(prog_counts >= 10), 'Not all halos are large than the minimum mass threshold'

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
        if len(hdf_desc['Halo_IDs'][...].shape) == 2:
            desc_snap_haloIDs = hdf_desc['Halo_IDs'][...]
        else:
            desc_snap_haloIDs = hdf_desc['Halo_IDs'][...]
        hdf_desc.close()

        # Get all unique halos in this snapshot
        desc_unique, desc_counts = np.unique(desc_snap_haloIDs, return_counts=True)

        # # Make sure all halos at this point in the data have more than ten particles
        # assert all(desc_counts >= 10), 'Not all halos are large than the mass threshold'

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        desc_unique = desc_unique[1:]
        desc_counts = desc_counts[1:]

    else:  # Assign an empty array if the snapshot is less than the earliest (061)
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
        current_npart = hdf_current[str(haloID)].attrs['halo_nPart']
        try:
            persist_npart = hdf_current[str(haloID)].attrs['halo_nPart_persist']
        except  KeyError:
            persist_npart = current_npart

        hdf_current.close()  # close the root group to reduce overhead when looping

        # If halo falls below the 20 cutoff threshold do not look for progenitors or descendants
        if current_halo_pids.size >= part_threshold and real:

            # =============== Run The Direct Progenitor and Descendant Finder ===============

            # Run the progenitor/descendant finder

            result = directProgDescFinder(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
                                          prog_counts, desc_counts, part_threshold, prog_reals, current_npart,
                                          persist_npart)
            (nprog, prog_haloids, prog_npart, prog_mass_contribution,
             ndesc, desc_haloids, desc_npart, desc_mass_contribution,
             preals, current_halo_pids, current_npart, persist_npart) = result

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

                hdf_desc[str(desc_haloids[0])].attrs['Real'] = True

                hdf_desc.close()

            # =============== Write Out Data ===============

            imreal += 1

            results[str(haloID)] = result

    hdf = h5py.File(savepath + 'Mtree_' + snapshot + '.hdf5', 'w')

    for num, haloID in enumerate(results.keys()):

        (nprog, prog_haloids, prog_npart, prog_mass_contribution,
         ndesc, desc_haloids, desc_npart, desc_mass_contribution,
         preals, current_halo_pids, current_npart, persist_npart) = results[haloID]

        # Print progress
        previous_progress = progress
        progress = int(num / size * 100)
        if progress != previous_progress:
            print('Write progress: ', progress, '%', haloID, end='\r')

        # Write out the data produced
        halo = hdf.create_group(haloID)  # create halo group
        halo.attrs['nProg'] = nprog  # number of progenitors
        halo.attrs['nDesc'] = ndesc  # number of descendants
        halo.attrs['current_halo_nPart'] = current_npart  # mass of the halo
        halo.attrs['current_halo_nPart_persist'] = persist_npart  # mass of the halo
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


def get_tree(z0halo, treepath):
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

    # =============== Progenitors ===============

    # Loop over snapshots and progenitor snapshots
    for prog_snap, snap in zip(snaplist[1:], snaplist[:-1]):

        # Loop over halos in this snapshot
        for halo in halos:

            # Remove snapshot ID from halo ID
            halo -= (int(snap) * 1000000)

            # Open this snapshots root group
            snap_tree_data = h5py.File(treepath + snap + '.hdf5', 'r')

            # Assign progenitors adding the snapshot * 100000 to the ID to keep track of the snapshot ID
            # in addition to the halo ID
            forest_dict.setdefault(prog_snap, set()).update(set((int(prog_snap) * 1000000) +
                                                                snap_tree_data[str(halo)]['Prog_haloIDs'][()]))
            snap_tree_data.close()

        # Assign the halos variable for the next stage of the tree
        halos = forest_dict[prog_snap]

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

    return forest_dict, mass_dict


def forest_worker(z0halo, treepath):

    # Get the forest with this halo at it's root
    forest_dict, mass_dict = get_tree(z0halo, treepath)

    print('Halo ' + str(z0halo) + '\'s Forest extracted...')

    return forest_dict, mass_dict


def forest_writer(forests, graphpath):

    snap_tree_data = h5py.File(graphpath + '.hdf5', 'w')

    for forest in forests:

        if len(forest) == 0:
            continue

        forest_dict, mass_dict = forest

        # Get highest mass root halo
        z0halo = forest_dict['061'][0]

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


def get_graph_members(treepath='MergerTrees/Mtree_', graphpath='MergerTrees/FullMtrees',
                      halopath='split_halos/halos_'):

    # Open the present day snapshot
    snap_tree_data = h5py.File(treepath + '061.hdf5', 'r', driver='core')

    # Get the haloIDs for the halos at the present day
    z0halos = list(snap_tree_data.keys())

    snap_tree_data.close()

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


def main_teaser(graphpath='MergerTrees/FullMtrees', treepath='MergerTrees/Mtree_'):

    # Open graph hdf5 file
    graph_hdf = h5py.File(graphpath + '.hdf5', 'r+')

    # Extract roots
    roots = list(graph_hdf.keys())

    graph_hdf.close()

    # Initialise dictionaries to store results
    halo_app = {}

    # Loop over roots
    for count, root in enumerate(roots):

        print('%.2f' % float(count/len(roots) * 100), root)

        # Open graph file and extract this roots graph
        graph_hdf = h5py.File(graphpath + '.hdf5', 'r+')
        graph = graph_hdf[root]
        graph_roots = list(graph['061'])
        graph_hdf.close()

        if len(graph_roots) > 1:

            # Walk down from each root
            trees = {}
            for graph_root in graph_roots:

                # Open this snapshots root group
                snap_tree_data = h5py.File(treepath + '061.hdf5', 'r')
                if snap_tree_data[str(graph_root)].attrs['current_halo_nPart'] < 20:
                    continue
                snap_tree_data.close()

                trees[graph_root] = root_walker(graph_root, treepath)

            for root1 in trees:
                tree1 = trees[root1]
                for snap in tree1.keys():
                    halo_app.setdefault(snap, {})
                    for halo in tree1[snap]:
                        if halo in halo_app[snap].keys():
                            halo_app[snap][halo] += 1
                        else:
                            halo_app[snap][halo] = 1
                # for root2 in trees:
                #     tree2 = trees[root2]
                #     if root1 == root2:
                #         continue
                #     com_halos = []
                #     for snap1, snap2 in zip(tree1.keys(), tree2.keys()):
                #
                #         common_halos = tree1[snap1] & tree2[snap2]
                #         if len(common_halos) > 0:
                #             com_halos.extend(common_halos)

        else:
            graph_hdf = h5py.File(graphpath + '.hdf5', 'r+')
            for snap in graph_hdf[root].keys():
                halo_app.setdefault(snap, {})
                for halo in graph_hdf[root][snap][...]:
                    if halo in halo_app[snap].keys():
                        halo_app[snap][halo] += 1
                    else:
                        halo_app[snap][halo] = 1

            graph_hdf.close()

        if count > 1000:
            # pprint.pprint(halo_app)
            break

    hist = {}
    for snap in halo_app.keys():
        for halo in halo_app[snap].keys():
            num = halo_app[snap][halo]
            if num in hist.keys():
                hist[num] += 1
            else:
                hist[num] = 1

    xs = []
    ys = []
    for key, val in hist.items():
        print(key, val)
        xs.append(key)
        ys.append(val)

    xs = np.array(xs)
    ys = np.array(ys)

    # Set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(xs, ys, width=1)

    ax.set_xlabel(r'Halo Appearences')
    ax.set_ylabel(r'$N$')

    ax.set_yscale('log')

    fig.savefig('halo_appearances.png', dpi=200, bbox_inches='tight')

    print(ys[np.where(xs == 1.)]/np.sum(ys[np.where(xs != 1.)]),
          'times as many halos appearing in multiple trees as appear in a single tree')


def root_walker(root, treepath):

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
    forest_dict = {}

    # Create snapshot list in reverse order (present day to past) for the progenitor searching loop
    snaplist = []
    for snap in range(61, 0, -1):
        if snap < 10:
            snaplist.append('00' + str(snap))
        elif snap >= 10:
            snaplist.append('0' + str(snap))

    # Initialise the halo's set for tree walking
    halos = {root, }

    # Initialise the forest dictionary with the present day halo as the first entry
    forest_dict['061'] = halos

    # =============== Progenitors ===============

    # Loop over snapshots and progenitor snapshots
    for prog_snap, snap in zip(snaplist[1:], snaplist[:-1]):

        # Loop over halos in this snapshot
        if len(halos) > 0:
            for halo in halos:

                # Open this snapshots root group
                snap_tree_data = h5py.File(treepath + snap + '.hdf5', 'r')

                # progs = snap_tree_data[str(halo)]['Prog_haloIDs'][...]
                # mass_cont = snap_tree_data[str(halo)]['prog_mass_contribution'][...]
                # prog_npart = snap_tree_data[str(halo)]['Prog_nPart'][...]
                # halo_npart = snap_tree_data[str(halo)].attrs['current_halo_nPart']
                #
                # merit = mass_cont ** 2 / (prog_npart * halo_npart)
                # genuine_progs = np.where(merit >= 0.015 ** 2)
                # progs = progs[genuine_progs]

                # Assign progenitors adding the snapshot * 100000 to the ID to keep track of the snapshot ID
                # in addition to the halo ID
                forest_dict.setdefault(prog_snap, set()).update(set(progs))
                snap_tree_data.close()

        else:
            forest_dict.setdefault(prog_snap, set())

        # Assign the halos variable for the next stage of the tree
        halos = forest_dict[prog_snap]

    return forest_dict


def link_cutter_worker(snap, treepath):

    # Compute the progenitor snapshot ID
    if int(snap) > 10:
        prog_snap = '0' + str(int(snap) - 1)
    else:
        prog_snap = '00' + str(int(snap) - 1)

    # Open the snapshot root group
    try:
        hdf = h5py.File(treepath + snap + '.hdf5', 'r')
    except OSError:
        print(snap, 'does not exist...')
        return

    try:
        # Open the snapshot root group
        prog_hdf = h5py.File(treepath + prog_snap + '.hdf5', 'r')
    except OSError:
        print(prog_snap, 'does not exist...')
        return

    previous_progress = -1
    count = 0
    size = len(hdf.keys())
    halos = list(hdf.keys())

    # Loop over halos in this snapshot
    for halo in halos:

        # Compute and print the progress
        progress = int(count/size*100)
        if progress != previous_progress:  # only print if the integer progress differs from the last printed value
            print('Breaking Incorrect Links... ' + snap + ' %.2f' % progress)
        previous_progress = progress
        count += 1

        hdf = h5py.File(treepath + snap + '.hdf5', 'r')

        # Get the old progs
        progs = hdf[halo]['Prog_haloIDs'][...]
        prog_masses = hdf[halo]['Prog_nPart'][...]
        prog_conts = hdf[halo]['prog_mass_contribution'][...]
        nProgs = len(progs)

        hdf.close()

        prog_hdf = h5py.File(treepath + prog_snap + '.hdf5', 'r')

        # Loop over progenitors
        new_progs = np.zeros_like(progs)
        new_prog_masses = np.zeros_like(progs)
        new_progs_cont = np.zeros_like(progs)
        for ind, (prog, mass, cont) in enumerate(zip(progs, prog_masses, prog_conts)):

            descs = prog_hdf[str(prog)]['Desc_haloIDs'][...]

            if int(descs[0]) == int(halo):
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
            new_nprogs = len(new_progs)

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

    pool = mp.Pool(int(mp.cpu_count() - 2))
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

    pool = mp.Pool(int(mp.cpu_count() - 2))
    pool.map(partial(link_cutter_worker, treepath=treepath), odd_snaplist)
    pool.close()
    pool.join()


def notreal_extract_tree(treepath='MergerGraphs/Mgraph_', halopath='halo_snapshots/halos_'):

    # # Create a snapshot list (past to present day) for looping
    snaplist = []
    for snap in range(0, 63):
        if snap < 10:
            snaplist.append('00' + str(snap))
        elif snap >= 10:
            snaplist.append('0' + str(snap))

    # Loop over snapshots
    for prog_snap, snap, desc_snap in zip(snaplist[:-2], snaplist[1:-1], snaplist[2:]):

        try:
            prog_hdf = h5py.File(halopath + prog_snap + '.hdf5', 'r+', driver='core')
        except OSError:
            prog_hdf = None
            print('No Progenitor Halo Snapshot')
        try:
            desc_hdf = h5py.File(halopath + desc_snap + '.hdf5', 'r+', driver='core')
        except OSError:
            print('No Descendent Halo Snapshot')
            desc_hdf = None
        try:
            tree_hdf = h5py.File(treepath + snap + '.hdf5', 'r+', driver='core')
        except OSError:
            continue
        try:
            desc_treehdf = h5py.File(treepath + desc_snap + '.hdf5', 'r+', driver='core')
        except OSError:
            print('No Descendent Tree Snapshot')
            desc_treehdf = None
        try:
            prog_treehdf = h5py.File(treepath + prog_snap + '.hdf5', 'r+', driver='core')
        except OSError:
            print('No Progenitor Tree Snapshot')
            prog_treehdf = None

        # Loop over halos
        for halo in tree_hdf.keys():

            print(snap, halo, end='\r')

            if tree_hdf[halo].attrs['nProg'] <= 1:
                continue

            # Get progenitor and descendent data
            progs = tree_hdf[halo]['Prog_haloIDs'][...]
            prog_masses = tree_hdf[halo]['Prog_nPart'][...]
            prog_conts = tree_hdf[halo]['prog_mass_contribution'][...]
            if prog_hdf != None:
                prog_reals = np.array([prog_hdf[str(prog)].attrs['Real'] for prog in progs], copy=False)
            else:
                prog_reals = [True, ]

            # If all progenitors are real move on
            if not np.all(prog_reals):

                # If the most massive halo is 'not real' included it, i.e. set it's realness to true
                if not prog_reals[0]:
                    prog_reals[0] = True

                # Get the not real halos
                not_reals = progs[np.invert(prog_reals)]
                real_progs = progs[prog_reals]
                real_masses = prog_masses[prog_reals]
                real_conts = prog_conts[prog_reals]

                if len(progs) != len(real_progs):
                    print(len(progs) - len(real_progs), 'Progenitors removed from', halo, 'in snapshot', snap + ':')

                    # Remove the previous entries
                    del tree_hdf[halo]['Prog_haloIDs']
                    del tree_hdf[halo]['Prog_nPart']
                    del tree_hdf[halo]['prog_mass_contribution']

                    tree_hdf[halo].create_dataset('prog_mass_contribution', data=real_conts, dtype=int,
                                        compression='gzip')  # Mass contribution
                    tree_hdf[halo].create_dataset('Prog_nPart', data=real_masses, dtype=int,
                                        compression='gzip')  # number of particles in each progenitor
                    tree_hdf[halo].create_dataset('Prog_haloIDs', data=real_progs, dtype=int,
                                        compression='gzip')  # progenitor IDs
                    tree_hdf[halo].attrs['nProg'] = len(real_progs)

        prog_hdf.close()
        tree_hdf.close()
        try:
            desc_hdf.close()
        except AttributeError:
            pass
        try:
            prog_treehdf.close()
        except AttributeError:
            pass
        try:
            desc_treehdf.close()
        except AttributeError:
            pass


def remove_unlinked_halos_worker(snap, treepath, halopath):

    try:
        tree_hdf = h5py.File(treepath + snap + '.hdf5', 'r+', driver='core')
    except OSError:
        print(treepath + snap + '.hdf5', 'does not exist...')
        return
    halo_hdf = h5py.File(halopath + snap + '.hdf5', 'r+', driver='core')

    for halo in tree_hdf.keys():

        print(snap, halo)

        # Delete this halo if it is not real and has no progenitors or descendents
        if not halo_hdf[halo].attrs['Real'] and tree_hdf[halo].attrs['nProg'] < 1 \
                and tree_hdf[halo].attrs['nDesc'] < 1:
            del tree_hdf[halo]

    tree_hdf.close()
    halo_hdf.close()


def remove_unlinked_halos(treepath='MergerGraphs/Mgraph_', halopath='halo_snapshots/halos_'):

    # Create a snapshot list (past to present day) for looping
    snaplist = []
    for snap in range(61, -1, -1):
        # if snap % 2 != 0: continue
        if snap < 10:
            snaplist.append('00' + str(snap))
        elif snap >= 10:
            snaplist.append('0' + str(snap))

    pool = mp.Pool(int(mp.cpu_count() - 2))
    pool.map(partial(remove_unlinked_halos_worker, treepath=treepath, halopath=halopath), snaplist)
    pool.close()
    pool.join()


# if __name__ == '__main__':
#     count = 0
#     while 'halos_001.hdf5' not in os.listdir('split_halos/'):
#         time.sleep(10)
#         print(count, end='\r')
#         count += 1
#
#
# def main(snap):
#     directProgDescWriter(snap, part_threshold=10, halopath='split_halos/', savepath='MergerTrees/')


# # Create a snapshot list (past to present day) for looping
# snaplist = []
# for snap in range(0, 62):
#     # if snap % 2 != 0: continue
#     if snap < 10:
#         snaplist.append('00' + str(snap))
#     elif snap >= 10:
#         snaplist.append('0' + str(snap))

# main_teaser()

# if __name__ == '__main__':
#     main('021')
    # pool = mp.Pool(int(mp.cpu_count() - 2))
    # pool.map(main, snaplist)
    # pool.close()
    # pool.join()

    # for snap in snaplist:
    #     main(snap)

    # link_cutter(treepath='MergerTrees/Mtree_')
    # notreal_extract_tree(treepath='MergerTrees/Mtree_', halopath='split_halos/halos_')
    # link_cutter(treepath='MergerTrees/Mtree_')
    # remove_unlinked_halos(treepath='MergerTrees/Mtree_', halopath='split_halos/halos_')
    # get_graph_members(treepath='MergerTrees/Mtree_', graphpath='MergerTrees/FullMtrees')
