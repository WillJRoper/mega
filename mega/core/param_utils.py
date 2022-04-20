import numpy as np
import yaml
import h5py
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as const
import astropy.units as u


# TODO: would be cleaner the future to have separate meta objects for
#  graphing and finding

def read_param(paramfile):
    # Read in the param file
    with open(paramfile) as yfile:
        parsed_yaml_file = yaml.load(yfile, Loader=yaml.FullLoader)

    # Extract individual dictionaries
    inputs = parsed_yaml_file["inputs"]
    flags = parsed_yaml_file["flags"]
    params = parsed_yaml_file["parameters"]
    cosmology = parsed_yaml_file["cosmology"]
    simulation = parsed_yaml_file["simulation"]

    return inputs, flags, params, cosmology, simulation


class Metadata:
    """
    Attributes:

    """

    def __init__(self, snaplist, snap_ind, cosmology, inputs, flags, params,
                 simulation, boxsize=None, npart=None, z=None, tot_mass=None):
        """

        :param snaplist:
        :param snap_ind:
        :param cosmology:
        :param inputs:
        :param flags:
        :param params:
        :param simulation:
        :param boxsize:
        :param npart:
        :param z:
        :param tot_mass:
        """

        # IO paths
        self.inputpath = inputs["data"] + inputs["basename"]
        self.savepath = inputs["haloSavePath"]
        self.halopath = self.savepath
        self.halo_basename = inputs["halo_basename"]
        self.dgraphpath = inputs["directgraphSavePath"]
        self.profile_path = inputs["profile_plot_path"]

        # Parameter file flags
        self.verbose = flags["verbose"]
        self.findsubs = flags["find_subs"]
        self.profile = flags["profile"]
        self.dmo = flags["DMO"]
        self.with_hydro = not self.dmo
        self.debug = flags["debug_mode"]

        # MPI information (given value if needed)
        self.rank = None
        self.nranks = None

        # Information about the box
        # Open hdf5 file
        self.snap = snaplist[snap_ind]
        if boxsize is None:
            # TODO: Replace this with a function for reading metadata
            hdf = h5py.File(self.inputpath + self.snap + ".hdf5", "r")
            self.boxsize = hdf["Header"].attrs["BoxSize"]
            self.npart = hdf["Header"].attrs["NumPart_Total"]
            self.z = hdf["Header"].attrs["Redshift"]
            hdf.close()
        else:
            self.boxsize = boxsize
            self.npart = npart
            self.z = z
        self.periodic = simulation["periodic"]
        self.box_vol = self.boxsize[0] * self.boxsize[1] * self.boxsize[2]
        self.comoving_soft = simulation["comoving_DM_softening"]
        self.max_physical_soft = simulation["max_physical_DM_softening"]
        self.a = 1 / (1 + self.z)
        if self.comoving_soft * self.a > self.max_physical_soft:
            self.soft = self.max_physical_soft / self.a
        else:
            self.soft = self.comoving_soft

        # Calculate mean separations
        self.mean_sep = np.zeros(len(self.npart), dtype=np.float64)
        for i in range(len(self.npart)):
            if self.npart[i] > 0:
                if i == 1:
                    self.mean_sep[i] = (self.box_vol
                                        / self.npart[i]) ** (1. / 3.)
                else:
                    self.mean_sep[i] = (self.box_vol
                                        / (np.sum(self.npart)
                                           - self.npart[1])) ** (1. / 3.)

        # Let"s get the progenitor and descendent snapshots
        self.prog_snap = None
        self.desc_snap = None
        if snap_ind - 1 >= 0:
            self.prog_snap = snaplist[snap_ind - 1]
        if snap_ind + 1 < len(snaplist):
            self.desc_snap = snaplist[snap_ind + 1]

        # Define booleans for snapshot handling
        self.isfinal = self.desc_snap is None
        self.isfirst = self.prog_snap is None

        # Physical constants
        self.G = (const.G.to(u.km ** 3 * u.M_sun ** -1 * u.s ** -2)).value

        # Unit information

        # Cosmology
        self.h = cosmology["h"]
        self.cosmo = FlatLambdaCDM(H0=cosmology["H0"],
                                   Om0=cosmology["Om0"],
                                   Tcmb0=cosmology["Tcmb0"],
                                   Ob0=cosmology["Ob0"])

        # Extract the mean density
        self.crit_density = self.cosmo.critical_density(self.z)
        self.omega_m = self.cosmo.Om(self.z)
        self.mean_den = (self.omega_m * self.crit_density).to(u.M_sun
                                                              / u.km ** 3)

        # Debugging gubbins
        if self.debug:  # set all snapshots as the same
            self.prog_snap = snaplist[snap_ind]
            self.desc_snap = snaplist[snap_ind]

        # If were running in DMO mode lets ensure we remove all particles
        temp_npart = np.zeros(len(self.npart), dtype=int)
        if self.dmo:
            temp_npart[1] = self.npart[1]
            self.npart = temp_npart
        else:
            # Always ignore boundary particles
            for part_type in range(len(self.npart)):
                if part_type in [2, 3] \
                        or part_type >= simulation["ignored_species_lim"]:
                    continue
                temp_npart[part_type] = self.npart[part_type]
            self.npart = temp_npart

        # Define list of particle types present
        self.part_types = [1, ] + [i for i in range(len(self.npart))
                                   if self.npart[i] != 0 and i != 1]

        # Define particle index offsets
        self.part_ind_offset = np.array([0, ] * len(self.npart), dtype=int)
        offset = 0
        for i in self.part_types:
            self.part_ind_offset[i] = offset
            offset += self.npart[i]
        self.hydro_ind_offset = self.part_ind_offset - self.npart[1]
        self.hydro_ind_offset[1] = 0

        # Find the index offsets for each particle type
        offset = self.npart[1]
        self.part_type_offset = [0] * len(self.npart)
        for i in self.part_types:
            if i == 1:
                continue
            self.part_type_offset[i] = offset
            offset += self.npart[i]

        # Print parameters
        self.report_width = 60
        self.table_width = 170

        # Linking length
        self.llcoeff = params["llcoeff"]
        self.sub_llcoeff = params["sub_llcoeff"]
        self.ini_vlcoeff = params["ini_alpha_v"]
        self.min_vlcoeff = params["min_alpha_v"]
        self.decrement = params["decrement"]
        self.linkl = self.llcoeff * self.mean_sep
        self.sub_linkl = self.sub_llcoeff * self.mean_sep

        # Define the velocity space linking length
        self.vlinkl_indp = (np.sqrt(self.G / 2)
                            * (4 * np.pi * 200
                               * self.mean_den / 3) ** (1 / 6)).value
        self.sub_vlinkl_indp = self.vlinkl_indp \
                               * (self.linkl[1] / self.sub_linkl[1]) ** (1 / 2)

        # Halo
        self.part_thresh = params["part_threshold"]

        # Domain decomp
        self.cdim = int(params["N_cells"])
        self.ncells = self.cdim ** 3

        # Tasks
        self.spatial_task_size = params["spatial_task_size"]

        # Are we cleaning up the real_flags?
        self.clean_snaps = flags["clean_real_flags"]

        # Linking metadata
        self.link_thresh = params["link_threshold"]

        # Timer object
        self.tictoc = None

    def check_verbose(self):
        """

        :return:
        """

        if self.rank == 0 and self.verbose == 1:
            self.verbose = 1
        elif self.verbose == 2:
            self.verbose = 1
        else:
            self.verbose = 0
