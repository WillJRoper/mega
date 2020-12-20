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
        preals = prog_reals[uniprog_haloids]
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
    hdf_current = h5py.File(halopath + 'halos_' + snap + '.hdf5', 'r')

    # Extract the halo IDs (group names/keys) contained within this snapshot
    if rank == 0:
        halo_ids = hdf_current['halo_IDs'][...]
        reals = hdf_current['real_flag'][...]
    else:
        halo_ids = hdf_current['Subhalos']['subhalo_IDs'][...]
        reals = hdf_current['Subhalos']['real_flag'][...]

    hdf_current.close()  # close the root group

    # Get only the real halo ids
    halo_ids = halo_ids[reals]

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
        prog_snap_haloIDs = hdf_prog['particle_halo_IDs'][:, rank]

        # Get all the unique halo IDs in this snapshot and the number of times they appear
        prog_unique, prog_counts = np.unique(prog_snap_haloIDs, return_counts=True)

        # Make sure all halos at this point in the data have more than ten particles
        assert all(prog_counts >= 10), 'Not all halos are large than the minimum mass threshold'

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        prog_unique = prog_unique[1:]
        prog_counts = prog_counts[1:]

        # Get progenitor snapshot data
        if rank == 0:
            prog_reals = hdf_prog['real_flag'][...]
        else:
            prog_reals = hdf_prog['Subhalos']['real_flag'][...]

        hdf_prog.close()

    else:
        prog_snap_haloIDs = np.array([])
        prog_reals = np.array([])
        prog_counts = np.array([])

    # Define the descendant snapshot ID (current ID + 1)
    if int(snapshot) > 8:
        desc_snap = '0' + str(int(snapshot) + 1)
    else:
        desc_snap = '00' + str(int(snapshot) + 1)

    # Only look for descendant data if there is a descendant snapshot (last snapshot has the ID '061')
    if int(desc_snap) <= 61:

        # Load the descenitor snapshot
        hdf_desc = h5py.File(halopath + 'halos_' + desc_snap + '.hdf5', 'r')

        # Extract the particle halo ID array and particle ID array
        desc_snap_haloIDs = hdf_desc['particle_halo_IDs'][:, rank]

        # Get all unique halos in this snapshot
        desc_unique, desc_counts = np.unique(desc_snap_haloIDs, return_counts=True)

        # Make sure all halos at this point in the data have more than ten particles
        assert all(desc_counts >= 10), 'Not all halos are large than the mass threshold'

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        desc_unique = desc_unique[1:]
        desc_counts = desc_counts[1:]

        # Get the reality flag array
        if rank == 0:
            desc_reals = hdf_desc['real_flag'][...]
        else:
            desc_reals = hdf_desc['Subhalos']['real_flag'][...]

        hdf_desc.close()

    else:
        desc_snap_haloIDs = np.array([])
        desc_counts = np.array([])
        desc_reals = np.array([])

    print(len(prog_reals), len(prog_counts), len(desc_reals), len(desc_counts))

    # =============== Find all Direct Progenitors And Descendant Of Halos In This Snapshot ===============

    # Initialise the progress
    progress = -1

    # Assign the number of halos for progress reporting
    size = len(halo_ids)
    imreal = 0
    notreals = 0
    results = {}

    # Set up arrays to store host results
    if len(halo_ids) != 0:
        nhalo = np.max(halo_ids) + 1
    else:
        nhalo = 0
    index_haloids = np.arange(nhalo, dtype=int)
    halo_nparts = np.full(nhalo, -2, dtype=int)
    nprogs = np.full(nhalo, -2, dtype=int)
    ndescs = np.full(nhalo, -2, dtype=int)
    prog_start_index = np.full(nhalo, -2, dtype=int)
    desc_start_index = np.full(nhalo, -2, dtype=int)

    progs = []
    descs = []
    prog_mass_conts = []
    desc_mass_conts = []
    prog_nparts = []
    desc_nparts = []

    # Loop through all the halos in this snapshot
    for num, haloID in enumerate(halo_ids):

        # =============== Current Halo ===============

        # Load the snapshot data
        hdf_current = h5py.File(halopath + 'halos_' + snapshot + '.hdf5', 'r')

        # Assign the particle IDs contained in the current halo
        current_halo_pids = hdf_current[str(haloID)]['Halo_Part_IDs'][...]

        hdf_current.close()  # close the root group to reduce overhead when looping

        # If halo falls below the 20 cutoff threshold do not look for progenitors or descendants
        if current_halo_pids.size >= part_threshold and reals[haloID]:

            # =============== Run The Direct Progenitor and Descendant Finder ===============

            # Run the progenitor/descendant finder

            result = directProgDescFinder(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
                                          prog_counts, desc_counts, part_threshold, prog_reals)
            (nprog, prog_haloids, prog_npart, prog_mass_contribution,
             ndesc, desc_haloids, desc_npart, desc_mass_contribution, preals, current_halo_pids) = result

            npart = current_halo_pids.size

            if ndesc > 40:
                print("-----------------------", haloID, "-----------------------")
                print(nprog, prog_haloids.size, ndesc, desc_haloids.size, preals, current_halo_pids.size)

            # If this halo has no real progenitors and is less than 20 particle it is by definition not
            # a halo
            if nprog == 0 and npart < 20:
                reals[haloID] = False
                notreals += 1
                continue

            # If the halo has neither descendants or progenitors we do not need to store it
            elif nprog == ndesc == -1 or nprog == ndesc == 0:
                notreals += 1
                reals[haloID] = False
                continue

            # If progenitor is real and this halo is not then it is real
            elif True in preals and not reals[haloID]:

                reals[haloID] = True

            # If this halo is real then it's descendents are real
            if desc_snap != None and reals[haloID]:
                desc_reals[desc_haloids] = True

            # # If the halo has neither descendants or progenitors we do not need to store it
            # if nprog == ndesc == -1 or nprog == ndesc == 0:
            #
            #     # Load the snapshot data
            #     hdf_current = h5py.File(halopath + 'halos_' + snapshot + '.hdf5', 'r+')
            #
            #     real = False
            #     hdf_current[str(haloID)].attrs['Real'] = real
            #
            #     hdf_current.close()  # close the root group to reduce overhead when looping
            #     continue
            #
            # # If progenitor is real and this halo is not then it is real
            # elif True in preals and not real:
            #
            #     # Load the snapshot data
            #     hdf_current = h5py.File(halopath + 'halos_' + snapshot + '.hdf5', 'r+')
            #
            #     real = True
            #     hdf_current[str(haloID)].attrs['Real'] = real
            #
            #     hdf_current.close()  # close the root group to reduce overhead when looping
            #
            # # If this halo has no real progenitors and is less than 20 particle it is by definition not
            # # a halo
            # elif nprog == 0 and current_halo_pids.size < 20:
            #
            #     # Load the snapshot data
            #     hdf_current = h5py.File(halopath + 'halos_' + snapshot + '.hdf5', 'r+')
            #
            #     real = False
            #     hdf_current[str(haloID)].attrs['Real'] = real
            #
            #     hdf_current.close()  # close the root group to reduce overhead when looping
            #     continue
            #
            # # If this halo is real then it's descendents are real
            # if real and int(desc_snap) < 62:
            #
            #     # Load the descendant snapshot
            #     hdf_desc = h5py.File(halopath + 'halos_' + desc_snap + '.hdf5', 'r+')
            #
            #     # Loop over decendents
            #     for desc in desc_haloids:
            #         hdf_desc[str(desc)].attrs['Real'] = True
            #
            #     hdf_desc.close()

            if reals[haloID]:

                # =============== Write Out Data ===============

                # Write out the data produced
                nprogs[haloID] = nprog  # number of progenitors
                ndescs[haloID] = ndesc  # number of descendants
                halo_nparts[int(haloID)] = current_halo_pids.size  # mass of the halo

                if nprog > 0:
                    prog_start_index[haloID] = len(progs)
                    progs.extend(prog_haloids)
                    prog_mass_conts.extend(prog_mass_contribution)
                    prog_nparts.extend(prog_npart)
                else:
                    prog_start_index[haloID] = 2 ** 30

                if ndesc > 0:
                    desc_start_index[haloID] = len(descs)
                    descs.extend(desc_haloids)
                    desc_mass_conts.extend(desc_mass_contribution)
                    desc_nparts.extend(desc_npart)
                else:
                    desc_start_index[haloID] = 2 ** 30

                imreal += 1

    progs = np.array(progs)
    descs = np.array(descs)
    prog_mass_conts = np.array(prog_mass_conts)
    desc_mass_conts = np.array(desc_mass_conts)
    prog_nparts = np.array(prog_nparts)
    desc_nparts = np.array(desc_nparts)

    # Create file to store this snapshots graph results
    if rank == 0:
        hdf = h5py.File(savepath + 'Mgraph_' + snap + '.hdf5', 'w')
    else:
        hdf = h5py.File(savepath + 'SubMgraph_' + snap + '.hdf5', 'w')

    hdf.create_dataset('halo_IDs', shape=index_haloids.shape, dtype=int, data=index_haloids, compression='gzip')
    hdf.create_dataset('nProgs', shape=nprogs.shape, dtype=int, data=nprogs, compression='gzip')
    hdf.create_dataset('nDescs', shape=ndescs.shape, dtype=int, data=ndescs, compression='gzip')
    hdf.create_dataset('nparts', shape=halo_nparts.shape, dtype=int, data=halo_nparts, compression='gzip')
    hdf.create_dataset('prog_start_index', shape=prog_start_index.shape, dtype=int, data=prog_start_index,
                       compression='gzip')
    hdf.create_dataset('desc_start_index', shape=desc_start_index.shape, dtype=int, data=desc_start_index,
                       compression='gzip')
    hdf.create_dataset('Prog_haloIDs', shape=progs.shape, dtype=int, data=progs, compression='gzip')
    hdf.create_dataset('Desc_haloIDs', shape=descs.shape, dtype=int, data=descs, compression='gzip')
    hdf.create_dataset('Prog_Mass_Contribution', shape=prog_mass_conts.shape, dtype=int, data=prog_mass_conts,
                       compression='gzip')
    hdf.create_dataset('Desc_Mass_Contribution', shape=desc_mass_conts.shape, dtype=int, data=desc_mass_conts,
                       compression='gzip')
    hdf.create_dataset('Prog_nPart', shape=prog_nparts.shape, dtype=int, data=prog_nparts, compression='gzip')
    hdf.create_dataset('Desc_nPart', shape=desc_nparts.shape, dtype=int, data=desc_nparts, compression='gzip')
    hdf.create_dataset('real_flag', shape=reals.shape, dtype=bool, data=reals, compression='gzip')

    hdf.close()

    if int(desc_snap) <= 61:

        # Load the descendant snapshot
        hdf_desc = h5py.File(halopath + 'halos_' + desc_snap + '.hdf5', 'r+')

        # Set the reality flag in the halo catalog
        if rank == 0:
            print("Overwriting descendant real flags")
            del hdf_desc['real_flag']
            hdf_desc.create_dataset('real_flag', shape=desc_reals.shape, dtype=bool, data=desc_reals,
                                    compression='gzip')
        else:
            print("Overwriting descendant subhalo real flags")
            sub_desc = hdf_desc['Subhalos']
            del sub_desc['real_flag']
            sub_desc.create_dataset('real_flag', shape=desc_reals.shape, dtype=bool, data=desc_reals,
                                    compression='gzip')

        hdf_desc.close()

    # Load the descendant snapshot
    hdf_current = h5py.File(halopath + 'halos_' + snap + '.hdf5', 'r+')

    # Set the reality flag in the halo catalog
    if rank == 0:
        print("Overwriting current real flags")
        del hdf_current['real_flag']
        hdf_current.create_dataset('real_flag', shape=reals.shape, dtype=bool, data=reals,
                                   compression='gzip')
    else:
        print("Overwriting current subhalo real flags")
        sub_current = hdf_current['Subhalos']
        del sub_current['real_flag']
        sub_current.create_dataset('real_flag', shape=reals.shape, dtype=bool, data=reals,
                                   compression='gzip')

    hdf_current.close()

    print(np.unique(nprogs, return_counts=True))
    print(np.unique(ndescs, return_counts=True))
    print(snapshot, 'Real=', imreal, 'All=', len(halo_ids), 'Not Real=', len(halo_ids) - imreal)

    return


def main(snap):
     directProgDescWriter(snap,
                          halopath='/Users/willroper/Documents/University/Merger_Trees_to_Merger_Graphs/mega/data/halos/',
                          savepath='/Users/willroper/Documents/University/Merger_Trees_to_Merger_Graphs/mega/data/dgraph',
                          part_threshold=10,
                          rank=0)


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
        if int(snap) > 30:
            continue
        print(snap)
        main(snap)
    print('Total:', time.time() - start)
