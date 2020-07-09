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
