import pickle
import h5py
import multiprocessing as mp
import functools as ft

lock = mp.Lock()


def tree_data_worker(halo, treepath, snap):

    halo_content = h5py.File(treepath + snap + '.hdf5', 'r')

    # Initialise this halos entry
    tree_data = {}

    # Assign halo's merger graph data
    tree_data['current_halo_nPart'] = halo_content[halo].attrs['current_halo_nPart']
    tree_data['current_halo_partIDs'] = halo_content[halo]['current_halo_partIDs'][...]
    tree_data['Prog_nPart'] = halo_content[halo]['Prog_nPart'][...]
    tree_data['Desc_nPart'] = halo_content[halo]['Desc_nPart'][...]
    tree_data['Prog_haloIDs'] = halo_content[halo]['Prog_haloIDs'][...]
    tree_data['Desc_haloIDs'] = halo_content[halo]['Desc_haloIDs'][...]
    tree_data['nProg'] = halo_content[halo].attrs['nProg']
    tree_data['nDesc'] = halo_content[halo].attrs['nDesc']
    # tree_data[snap][halo]['halo_energy'] = halo_content[halo].attrs['halo_energy']

    halo_content.close()

    return halo, tree_data


def loader(treepath, halopath, extracttree=False, extractsnap=False, extracthalo=False,
           halo_attributes=[], snap_attributes=[], savepath='treedata.pck'):
    """ A helper function to take the HDF5 files and convert them to dictionaries for convenience
    and efficiency when computing statistics.

    :param treepath: The filepath to the merger graph HDF5 file.
    :param halopath: The filepath to the halo finder HDF5 file.
    :param extracttree: Boolean for whether to create merger graph dictionary.
    :param extractsnap: Boolean for whether to create snpashot metadata dictionary.
    :param extracthalo: Boolean for whether to create halo finder dictionary.
    :param halo_attributes: List of halo attributes and datasets to extract.
    :param snap_attributes:List of snapshot attributes and datasets to extract.

    :return: None
    """

    # Create snapshot list (past to present day) for looping
    snaplist = []
    for snap in range(0, 62):
        if snap < 10:
            snaplist.append('00' + str(snap))
        elif snap >= 10:
            snaplist.append('0' + str(snap))

    # Initialise dictionaries for storing data
    tree_data = {}
    snap_data = {}
    halo_data = {}

    # Loop over snapshots
    for snap in iter(snaplist):

        # Initialise this snapshots dictionary entries
        tree_data[snap] = {}
        snap_data[snap] = {}
        halo_data[snap] = {}

        # Print progress
        progress = int(int(snap)/61. * 100)
        print('Loading Tree Data... ', progress, '%')

        # Extract merger graph data if required
        if extracttree:

            # Snapshots without halos do not have merger graph files so they must be ignored using the exception
            try:
                snap_tree_data = h5py.File(treepath + snap + '.hdf5', 'r')
                halos = list(snap_tree_data.keys())
                snap_tree_data.close()
            except OSError:
                print(snap, 'No such file')
                continue

            pool = mp.Pool(int(mp.cpu_count() - 2))
            results = pool.map(ft.partial(tree_data_worker, treepath=treepath, snap=snap), halos)
            pool.close()
            pool.join()

            for result in results:

                halo = result[0]
                contents = result[1]
                tree_data[snap][halo] = contents

        # Extract snapshot data if required
        if extractsnap:

            # Open this snapshots halo finder HDF5 file
            snap_halo_data = h5py.File(halopath + snap + '.hdf5', 'r', driver='core')

            # Assign the snapshot's attributes to the dictionary for output
            for attri, attrival in zip(snap_halo_data.attrs.keys(), snap_halo_data.attrs.values()):
                if attri in snap_attributes:  # only extract the required data

                    # For time the returned value must be converted from units of 10 Gyrs to yrs
                    if attri == 'time':
                        snap_data[snap][attri] = attrival * 10*10**9

                    else:
                        snap_data[snap][attri] = attrival

            # Assign the snapshot's datasets to the dictionary for output
            for key, dataset in zip(snap_halo_data.keys(), snap_halo_data.values()):
                    if key in snap_attributes:  # only extract the required data
                        snap_data[snap][key] = dataset.value

            snap_halo_data.close()

        # Extract halo data if required
        if extracthalo:

            # Load the halo data
            snap_halo_data = h5py.File(halopath + snap + '.hdf5', 'r', driver='core')

            # Assign the halo attributes and datasets to the dictionary for output
            for halo in snap_halo_data.keys():

                # Ignore the datasets from the entire snapshot rather than halo groups in the keys
                if halo == 'Halo_IDs' or halo == 'Part_IDs':
                    continue

                # Initialise this halos dictionary
                halo_data[snap][halo] = {}

                # Assign the halo's attributes to the dictionary for output
                for attri, attrival in zip(snap_halo_data[halo].attrs.keys(), snap_halo_data[halo].attrs.values()):
                    if attri in halo_attributes:  # only extract the required data
                        halo_data[snap][halo][attri] = attrival

                # Assign the halo's datasets to the dictionary for output
                for key, dataset in zip(snap_halo_data[halo].keys(), snap_halo_data[halo].values()):
                    if key in halo_attributes:  # only extract the required data
                        halo_data[snap][halo][key] = dataset.value

            snap_halo_data.close()

    # Write out the dictionaries
    if extracttree:
        with open(savepath, 'wb') as pfile1:
            pickle.dump(tree_data, pfile1)
    if extractsnap:
        with open('snapdata' + str(*snap_attributes) + '.pck', 'wb') as pfile2:
            pickle.dump(snap_data, pfile2)
    if extracthalo:
        with open('halodata' + str(*halo_attributes) + '.pck', 'wb') as pfile3:
            pickle.dump(halo_data, pfile3)

    return


if __name__ == '__main__':
    loader('MergerGraphs_sub/Mgraph_', 'halo_snapshots_sub/halos_', extracttree=True,
           halo_attributes=[], snap_attributes=['time'], savepath='treedata_sub.pck')
