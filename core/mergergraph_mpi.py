import numpy as np
import h5py
from mpi4py import MPI
from guppy import hpy; hp = hpy()
import utilities


# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # density_rank of this process
status = MPI.Status()   # get MPI status object


def directProgDescFinder(haloID, prog_snap, desc_snap, prog_haloids, desc_haloids, prog_reals,
                         prog_nparts, desc_nparts, npart):
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
    if prog_snap != None:

        # Find the unique halo IDs and the number of times each appears
        uniprog_haloids, uniprog_counts = np.unique(prog_haloids, return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value.
        if uniprog_haloids[0] == -2:
            uniprog_haloids = uniprog_haloids[1:]
            uniprog_counts = uniprog_counts[1:]

        # Halos are only linked if they have 10 or more particles in common
        okinds = uniprog_counts >= 10
        uniprog_haloids = uniprog_haloids[okinds]
        uniprog_counts = uniprog_counts[okinds]

        # Get the reality flag
        preals = prog_reals[uniprog_haloids]

        # Get only real halos
        uniprog_haloids = uniprog_haloids[preals]
        uniprog_counts = uniprog_counts[preals]

        # Find the number of progenitor halos from the size of the unique array
        nprog = uniprog_haloids.size

        # Assign the corresponding number of particles in each progenitor for sorting and storing
        # This can be done simply by using the ID of the progenitor since again np.unique returns
        # sorted results.
        prog_npart = prog_nparts[uniprog_haloids]

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
    if desc_snap != None:

        # Find the unique halo IDs and the number of times each appears
        unidesc_haloids, unidesc_counts = np.unique(desc_haloids, return_counts=True)

        # Remove single particle halos (ID=-2) for the counts, since np.unique returns a sorted array this can be
        # done by removing the first value.
        if unidesc_haloids[0] == -2:
            unidesc_haloids = unidesc_haloids[1:]
            unidesc_counts = unidesc_counts[1:]

        # Halos are only linked if they have 10 or more particles in common
        okinds = unidesc_counts >= 10
        unidesc_haloids = unidesc_haloids[okinds]
        unidesc_counts = unidesc_counts[okinds]

        # Find the number of descendant halos from the size of the unique array
        ndesc = unidesc_haloids.size

        # Assign the corresponding number of particles in each progenitor for sorting and storing
        # This can be done simply by using the ID of the progenitor since again np.unique returns
        # sorted results.
        desc_npart = desc_nparts[unidesc_haloids]

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

    return (haloID, nprog, prog_haloids, prog_npart, prog_mass_contribution,
            ndesc, desc_haloids, desc_npart, desc_mass_contribution,
            preals, npart)


def directProgDescWriter(snap, prog_snap, desc_snap, halopath, savepath,
                         density_rank, verbose, final_snapnum):
    """ A function which cycles through all halos in a snapshot finding and writing out the
    direct progenitor and descendant data.

    :param snapshot: The snapshot ID.
    :param halopath: The filepath to the halo finder HDF5 file.
    :param savepath: The filepath to the directory where the Merger Graph should be written out to.
    :param part_threshold: The mass (number of particles) threshold defining a halo.

    :return: None
    """

    # Define MPI message tags
    tags = utilities.enum('READY', 'DONE', 'EXIT', 'START')
    
    if rank == 0:

        # =============== Current Snapshot ===============

        # Load the current snapshot data
        hdf_current = h5py.File(halopath + 'halos_' + snap + '.hdf5', 'r')
    
        # Extract the halo IDs (group names/keys) contained within this snapshot
        if density_rank == 0:
            halo_ids = hdf_current['halo_IDs'][...]
            reals = hdf_current['real_flag'][...]
        else:
            halo_ids = hdf_current['Subhalos']['subhalo_IDs'][...]
            reals = hdf_current['Subhalos']['real_flag'][...]

        if prog_snap != None:

            # Load the progenitor snapshot
            hdf_prog = h5py.File(halopath + 'halos_' + prog_snap + '.hdf5', 'r')

            # Extract the particle halo ID array and particle ID array
            prog_haloids = hdf_prog['particle_halo_IDs'][:, density_rank]

            # Get progenitor snapshot data
            if density_rank == 0:
                preals = hdf_prog['real_flag'][...]
                prog_npart = hdf_prog['nparts'][...]
            else:
                preals = hdf_prog['Subhalos']['real_flag'][...]
                prog_npart = hdf_prog['Subhalos']['nparts'][...]

            hdf_prog.close()

        else:
            prog_haloids = np.array([])
            preals = np.array([])
            prog_npart = np.array([])

        if desc_snap != None:

            # Load the descenitor snapshot
            hdf_desc = h5py.File(halopath + 'halos_' + desc_snap + '.hdf5', 'r')

            # Extract the particle halo ID array and particle ID array
            desc_haloids = hdf_desc['particle_halo_IDs'][:, density_rank]

            # Get descenitor snapshot data
            if density_rank == 0:
                desc_npart = hdf_desc['nparts'][...]
            else:
                desc_npart = hdf_desc['Subhalos']['nparts'][...]

            hdf_desc.close()

        else:
            desc_haloids = np.array([])
            desc_npart = np.array([])

        # =============== Find all Direct Progenitors And Descendant Of Halos In This Snapshot ===============

        results = {}

        # Master process executes code below
        tasks = set(halo_ids)
        num_workers = size - 1
        closed_workers = 0
        while closed_workers < num_workers:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == tags.READY:

                # Worker is ready, so send it a task
                if len(tasks) != 0:

                    real = False
                    haloID = None

                    while not real and len(tasks) > 0:

                        haloID = tasks.pop()

                        real = reals[haloID]

                    # Assign the particle IDs contained in the current halo
                    current_halo_pids = hdf_current[str(haloID)]['Halo_Part_IDs'][...]
                    progs = prog_haloids[current_halo_pids]
                    descs = desc_haloids[current_halo_pids]

                    comm.send((haloID, prog_snap, desc_snap, progs, descs, preals, prog_npart, desc_npart,
                               current_halo_pids.size),
                              dest=source, tag=tags.START)

                else:

                    # There are no tasks left so terminate this process
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:
                result = data

                results[result[0]] = result[1:]

            elif tag == tags.EXIT:

                closed_workers += 1

        hdf_current.close()  # close the root group to reduce overhead when looping

    else:

        results = None
        reals = None
        halo_ids = None

        # Worker processes execute code below
        name = MPI.Get_processor_name()
        # print("I am a worker with rank %d on %s." % (rank, name))
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            thisTask = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:

                result = directProgDescFinder(thisTask[0], thisTask[1], thisTask[2], thisTask[3], thisTask[4],
                                              thisTask[5], thisTask[6], thisTask[7], thisTask[8])

                comm.send(result, dest=0, tag=tags.DONE)

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)

    if rank == 0:

        notreals = 0

        # Set up arrays to store host results
        nhalo = len(halo_ids)
        index_haloids = halo_ids
        halo_nparts = np.full(nhalo, -2, dtype=int)
        nprogs = np.full(nhalo, -2, dtype=int)
        ndescs = np.full(nhalo, -2, dtype=int)
        prog_start_index = np.full(nhalo, -2, dtype=int)
        desc_start_index = np.full(nhalo, -2, dtype=int)
        # progs = np.full(nhalo * 2, -2, dtype=int)
        # descs = np.full(nhalo, -2, dtype=int)
        # prog_mass_conts = np.full(nhalo * 2, -2, dtype=int)
        # desc_mass_conts = np.full(nhalo * 2, -2, dtype=int)
        # prog_nparts = np.full(nhalo * 2, -2, dtype=int)
        # desc_nparts = np.full(nhalo * 2, -2, dtype=int)

        progs = []
        descs = []
        prog_mass_conts = []
        desc_mass_conts = []
        prog_nparts = []
        desc_nparts = []

        # prog_ind = 0
        # desc_ind = 0

        # Load the descendant snapshot
        hdf_desc = h5py.File(halopath + 'halos_' + desc_snap + '.hdf5', 'r')

        # Get the reality flag array
        if density_rank == 0:
            desc_reals = hdf_desc['real_flag'][...]
        else:
            desc_reals = hdf_desc['Subhalos']['real_flag'][...]

        hdf_desc.close()

        for num, haloID in enumerate(results):

            (nprog, prog_haloids, prog_npart, prog_mass_contribution,
             ndesc, desc_haloids, desc_npart, desc_mass_contribution, preals, npart) = results[haloID]

            # If this halo has no real progenitors and is less than 20 particle it is by definition not
            # a halo
            if nprog == 0 and npart < 20:

                reals[haloID] = False
                notreals += 1

                continue

            # If this halo is real then it's descendents are real
            if int(desc_snap) < final_snapnum and reals[haloID]:

                desc_reals[desc_haloids] = True

            # Write out the data produced
            nprogs[haloID] = nprog  # number of progenitors
            ndescs[haloID] = ndesc  # number of descendants
            halo_nparts[int(haloID)] = npart  # mass of the halo

            if nprog > 0:
                prog_start_index[haloID] = len(progs)
                progs.extend(prog_haloids)
                prog_mass_conts.extend(prog_mass_contribution)
                prog_nparts.extend(prog_npart)
            else:
                prog_start_index[haloID] = 2**30

            if ndesc > 0:
                desc_start_index[haloID] = len(descs)
                descs.extend(desc_haloids)
                desc_mass_conts.extend(desc_mass_contribution)
                desc_nparts.extend(desc_npart)
            else:
                desc_start_index[haloID] = 2**30


            # if nprog > 0:
            #     prog_start_index[haloID] = prog_ind
            #     progs[prog_ind: prog_ind + nprog] = prog_haloids
            #     prog_mass_conts[prog_ind: prog_ind + nprog] = prog_mass_contribution
            #     prog_nparts[prog_ind: prog_ind + nprog] = prog_npart
            #
            # if ndesc > 0:
            #     desc_start_index[haloID] = desc_ind
            #     descs[desc_ind: desc_ind + ndesc] = desc_haloids
            #     desc_mass_conts[desc_ind: desc_ind + ndesc] = desc_mass_contribution
            #     desc_nparts[desc_ind: desc_ind + ndesc] = desc_npart

            # prog_ind += nprog
            # desc_ind += ndesc

        # progs = progs[:prog_ind]
        # descs = descs[:desc_ind]
        # prog_mass_conts = prog_mass_conts[:prog_ind]
        # desc_mass_conts = desc_mass_conts[:desc_ind]
        # prog_nparts = prog_nparts[:prog_ind]
        # desc_nparts = desc_nparts[:desc_ind]

        progs = np.array(progs)
        descs = np.array(descs)
        prog_mass_conts = np.array(prog_mass_conts)
        desc_mass_conts = np.array(desc_mass_conts)
        prog_nparts = np.array(prog_nparts)
        desc_nparts = np.array(desc_nparts)

        # Create file to store this snapshots graph results
        if density_rank == 0:
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

        # # Load the descendant snapshot
        # hdf_desc = h5py.File(halopath + 'halos_' + desc_snap + '.hdf5', 'r+')
        #
        # # Set the reality flag in the halo catalog
        # if density_rank == 0:
        #     del hdf_desc['real_flag']
        #     hdf_desc.create_dataset('real_flag', shape=desc_reals.shape, dtype=bool, data=desc_reals,
        #                             compression='gzip')
        # else:
        #     sub_desc = hdf_desc['Subhalos']
        #     del sub_desc['real_flag']
        #     sub_desc.create_dataset('real_flag', shape=desc_reals.shape, dtype=bool, data=desc_reals,
        #                             compression='gzip')
        #
        # hdf_desc.close()
        #
        # # Load the descendant snapshot
        # hdf_current = h5py.File(halopath + 'halos_' + snap + '.hdf5', 'r+')
        #
        # # Set the reality flag in the halo catalog
        # if density_rank == 0:
        #     del hdf_current['real_flag']
        #     hdf_current.create_dataset('real_flag', shape=reals.shape, dtype=bool, data=reals,
        #                             compression='gzip')
        # else:
        #     sub_current = hdf_current['Subhalos']
        #     del sub_current['real_flag']
        #     sub_current.create_dataset('real_flag', shape=reals.shape, dtype=bool, data=reals,
        #                             compression='gzip')
        #
        # hdf_current.close()

        print(np.unique(nprogs, return_counts=True))
        print(np.unique(ndescs, return_counts=True))
        print("Not real halos", notreals, 'of', nhalo)
