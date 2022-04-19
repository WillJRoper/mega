import numpy as np


from mega.halo_core.phase_space import get_real_host_halos, get_real_sub_halos
import mega.core.serial_io as serial_io
import mega.core.utilities as utils
from mega.halo_core.spatial import get_sub_halos


def get_halos(tictoc, this_task, meta, results, sub_results, vlinkl_indp,
              haloID, subhaloID, hdf_part_key):
    """

    :param tictoc:
    :param this_task:
    :param meta:
    :param results:
    :param sub_results:
    :param vlinkl_indp:
    :param haloID:
    :param subhaloID:
    :param hdf_part_key:
    :return:
    """

    # Sort the particle ids to index the hdf5 array
    this_task = utils.set_2_sorted_array(tictoc, this_task)

    # Get halo data from file
    halo = serial_io.read_halo_data(tictoc, this_task,
                                    meta.inputpath,
                                    meta.snap, hdf_part_key,
                                    meta.ini_vlcoeff, meta.boxsize,
                                    meta.soft, meta.z, meta.G,
                                    meta.cosmo)

    # Test the halo in phase space
    result = get_real_host_halos(tictoc, halo, meta.boxsize, vlinkl_indp,
                                 meta.linkl, meta.decrement, meta.z,
                                 meta.G,
                                 meta.soft, meta.min_vlcoeff,
                                 meta.cosmo)

    # Save the found halos
    for res in result:
        results[(meta.rank, haloID)] = res

        haloID += 1

    if meta.findsubs:

        # Define a dictionary to temporarily store the spatial subhalos
        spatial_sub_results = {}

        # Loop over results getting spatial halos
        for host in result:

            # Sort the particle ids to index the hdf5 array
            thishalo_pids = utils.timed_sort(tictoc, host.pids)

            # Get the particle positions
            subhalo_poss = serial_io.read_subset(tictoc, meta,
                                                 hdf_part_key
                                                 + "/part_pos",
                                                 thishalo_pids)

            # Test the subhalo in phase space
            sub_result = get_sub_halos(tictoc, thishalo_pids,
                                       subhalo_poss,
                                       meta.sub_linkl)

            # Store the found halos
            while len(sub_result) > 0:
                key, res = sub_result.popitem()
                spatial_sub_results[subhaloID] = res

                subhaloID += 1

        # Loop over spatial subhalos
        while len(spatial_sub_results) > 0:

            key, this_stask = spatial_sub_results.popitem()

            # Sort the particle ids to index the hdf5 array
            this_stask = utils.set_2_sorted_array(tictoc, this_stask)

            # Get halo data from file
            subhalo = serial_io.read_halo_data(tictoc, this_stask,
                                               meta.inputpath,
                                               meta.snap,
                                               hdf_part_key,
                                               meta.ini_vlcoeff,
                                               meta.boxsize,
                                               meta.soft, meta.z,
                                               meta.G,
                                               meta.cosmo)

            # Test the subhalo in phase space
            result = get_real_sub_halos(tictoc, subhalo,
                                        meta.boxsize,
                                        vlinkl_indp * 8 ** (
                                                1 / 6),
                                        meta.linkl, meta.decrement,
                                        meta.z, meta.G,
                                        meta.soft, meta.min_vlcoeff,
                                        meta.cosmo)

            # Save the found subhalos
            for res in result:
                sub_results[(meta.rank, subhaloID)] = res

                subhaloID += 1

    return results, sub_results, haloID, subhaloID
