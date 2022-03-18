from pygadgetreader import readheader, readsnap


def readsnapshot(snapshot, PATH='snapshotdata/snapdir_'):
    """ A program to read in gadget-2 snapshot data with filename 62.5_dm_XXX.k .

    :param snapshot: The number of the snapshot as a string (e.g. '001').
    :param PATH: The filepath to the snapshot data directory.

    :return: A list for each variable in the snapshot data no longer split
    into files, along with the relevant header data.
    """

    # Read in the header data and assign to variables
    head = readheader(PATH + snapshot + '/62.5_dm_' + snapshot, 'header')
    npart = head.get('npartTotal')[1]  # number of particles in snapshot
    z = head.get('redshift')  # redshift of snapshot
    t = head.get('time')  # time of snapshot (age)
    h = head.get('h')
    boxsize = head.get('boxsize')  # size of simulation box
    rhocrit = head.get('rhocrit')  # critical density at time t

    # Read in the snapshot data and assign to variable for returning
    pos = readsnap(PATH + snapshot + '/62.5_dm_' + snapshot, 'pos', 1)
    vel = readsnap(PATH + snapshot + '/62.5_dm_' + snapshot, 'vel', 1)
    pid = readsnap(PATH + snapshot + '/62.5_dm_' + snapshot, 'pid', 1)
    pmass = readsnap(PATH + snapshot + '/62.5_dm_' + snapshot, 'mass', 1)[0]

    return pid, pos, vel, npart, z, t, boxsize, rhocrit, pmass, h


# def GADGET_binary_to_hdf5(snapshot, PATH, inputpath='input/'):
#     """ Reads in GADGET binary format simulation data
#     and writes the necessary fields into the expected MEGA format.
#
#     :param snapshot: The snapshot ID as a string (e.g. '061')
#     :param PATH: The filepath to the directory containing the simulation data.
#     :param inputpath: The file path for storing the MEGA format
#                       input data read from the simulation files.
#
#     :return:
#     """
#
#     # =============== Load Simulation Data ===============
#
#     # Load snapshot data from gadget-2 file *** Note: will need to be
#     # changed for use with other simulations data ***
#     snap = readgadgetdata.readsnapshot(snapshot, PATH)
#     pid, pos, vel = snap[
#                     0:3]  # pid=particle ID, pos=all particle's position,
#                           # vel=all particle's velocity
#     head = snap[3:]  # header values
#     npart = head[0]  # number of particles in simulation
#     boxsize = head[3]  # simulation box length(/size) along each axis
#     redshift = head[1]
#     t = head[2]  # elapsed time of the snapshot
#     rhocrit = head[4]  # Critical density
#     pmass = head[5]  # Particle mass
#     h = head[6]  # 'little h' (hubble parameter parametrisation)
#
#     # =============== Sort particles ===============
#
#     # Sort the simulation data arrays by the particle ID
#     sinds = pid.argsort()
#     pid = pid[sinds]
#     pos = pos[sinds, :]
#     vel = vel[sinds, :]
#
#     # =============== Compute Linking Length ===============
#
#     # Compute the mean separation
#     mean_sep = boxsize / npart ** (1. / 3.)
#
#     # Open hdf5 file
#     hdf = h5py.File(inputpath + "mega_inputs_" + snapshot + ".hdf5", 'w')
#
#     # Write out the inputs
#     hdf.attrs['mean_sep'] = mean_sep
#     hdf.attrs['boxsize'] = boxsize
#     hdf.attrs['npart'] = npart
#     hdf.attrs['redshift'] = redshift
#     hdf.attrs['t'] = t
#     hdf.attrs['rhocrit'] = rhocrit
#     hdf.attrs['pmass'] = pmass
#     hdf.attrs['h'] = h
#     hdf.create_dataset('part_pid', shape=pid.shape, dtype=float, data=pid,
#                        compression="gzip")
#     hdf.create_dataset('sort_inds', shape=sinds.shape, dtype=int, data=sinds,
#                        compression="gzip")
#     hdf.create_dataset('part_pos', shape=pos.shape, dtype=float, data=pos,
#                        compression="gzip")
#     hdf.create_dataset('part_vel', shape=vel.shape, dtype=float, data=vel,
#                        compression="gzip")
#
#     hdf.close()
