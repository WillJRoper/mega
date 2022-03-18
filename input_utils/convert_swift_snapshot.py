import numpy as np
import h5py

from core.serial_io import hdf5_write_dataset


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
