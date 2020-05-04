import yaml
import readgadgetdata


def read_param(paramfile):

    # Read in the param file
    with open(paramfile) as yfile:
        parsed_yaml_file = yaml.load(yfile, Loader=yaml.FullLoader)

    # Extract individual dictionaries
    inputs = parsed_yaml_file['inputs']
    flags = parsed_yaml_file['flags']
    params = parsed_yaml_file['params']

    return inputs, flags, params


def read_binary(snapshot, PATH, llcoeff):
    """ Reads in gadget-2 simulation data and computes the host halo linking length. (For more information see Docs)

    :param snapshot: The snapshot ID as a string (e.g. '061')
    :param PATH: The filepath to the directory containing the simulation data.
    :param llcoeff: The host halo linking length coefficient.

    :return: pid: An array containing the particle IDs.
             pos: An array of the particle position vectors.
             vel: An array of the particle velocity vectors.
             npart: The number of particles used in the simulation.
             boxsize: The length of the simulation box along a single axis.
             redshift: The redshift of the current snapshot.
             t: The elapsed time of the current snapshot.
             rhocrit: The critical density at the current snapshot.
             pmass: The mass of a dark matter particle.
             h: 'Little h', The hubble parameter parametrisation.
             linkl: The linking length.

    """

    # =============== Load Simulation Data ===============

    # Load snapshot data from gadget-2 file *** Note: will need to be changed for use with other simulations data ***
    snap = readgadgetdata.readsnapshot(snapshot, PATH)
    pid, pos, vel = snap[0:3]  # pid=particle ID, pos=all particle's position, vel=all particle's velocity
    head = snap[3:]  # header values
    npart = head[0]  # number of particles in simulation
    boxsize = head[3]  # simulation box length(/size) along each axis
    redshift = head[1]
    t = head[2]  # elapsed time of the snapshot
    rhocrit = head[4]  # Critical density
    pmass = head[5]  # Particle mass
    h = head[6]  # 'little h' (hubble parameter parametrisation)

    # =============== Compute Linking Length ===============

    # Compute the mean separation
    mean_sep = boxsize / npart**(1./3.)

    # Compute the linking length for host halos
    linkl = llcoeff * mean_sep

    return pid, pos, vel, npart, boxsize, redshift, t, rhocrit, pmass, h, linkl
