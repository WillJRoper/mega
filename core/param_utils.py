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
    inputs = parsed_yaml_file['inputs']
    flags = parsed_yaml_file['flags']
    params = parsed_yaml_file['parameters']
    cosmology = parsed_yaml_file['cosmology']
    simulation = parsed_yaml_file['simulation']

    return inputs, flags, params, cosmology, simulation


class Metadata:
    """
    Attributes:

    """

    def __init__(self, snaplist, snap_ind, cosmology,
                 llcoeff, sub_llcoeff, inputs, inputpath, savepath,
                 ini_vlcoeff, min_vlcoeff, decrement, verbose, findsubs,
                 cdim, profile, profile_path, h, softs, dmo):
        """

        :param snaplist:
        :param snap_ind:
        :param cosmology:
        :param llcoeff:
        :param sub_llcoeff:
        :param inputpath:
        :param savepath:
        :param ini_vlcoeff:
        :param min_vlcoeff:
        :param decrement:
        :param verbose:
        :param findsubs:
        :param cdim:
        :param profile:
        :param profile_path:
        :param h:
        :param softs:
        :param dmo:
        """

        # MPI information (given value if needed)
        self.rank = None
        self.nranks = None

        # Information about the box
        # Open hdf5 file
        self.snap = snaplist[snap_ind]
        hdf = h5py.File(inputpath + self.snap + ".hdf5", 'r')
        self.boxsize = hdf["Header"].attrs["BoxSize"]
        self.npart = hdf["Header"].attrs["NumPart_Total"]
        self.z = hdf["Header"].attrs["Redshift"]
        self.tot_mass = np.sum(hdf["PartType1/Masses"]) * 10 ** 10
        hdf.close()
        self.box_vol = self.boxsize[0] * self.boxsize[1] * self.boxsize[2]
        self.mean_sep = (self.box_vol / self.npart[1]) ** (1. / 3.)
        self.comoving_soft = softs[0]
        self.max_physical_soft = softs[1]
        mean_den = (self.tot_mass * u.M_sun / self.box_vol
                    / u.Mpc ** 3 * (1 + self.z) ** 3)
        self.mean_den = mean_den.to(u.M_sun / u.km ** 3)
        self.a = 1 / (1 + self.z)
        if self.comoving_soft * self.a > self.max_physical_soft:
            self.soft = self.max_physical_soft / self.a
        else:
            self.soft = self.comoving_soft

        # Let's get the progenitor and descendent snapshots
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
        self.h = h
        self.cosmo = FlatLambdaCDM(H0=cosmology["H0"],
                                   Om0=cosmology["Om0"],
                                   Tcmb0=cosmology["Tcmb0"],
                                   Ob0=cosmology["Ob0"])

        # IO paths
        self.inputpath = inputpath
        self.savepath = savepath
        self.halopath = savepath
        self.dgraphpath = inputs["directgraphSavePath"]
        self.profile_path = profile_path

        # Parameter file flags
        self.verbose = verbose
        self.findsubs = findsubs
        self.profile = profile
        self.dmo = dmo
        self.debug = False

        # Debugging gubbins
        if self.debug:  # set all snapshots as the same
            self.prog_snap = snaplist[snap_ind]
            self.desc_snap = snaplist[snap_ind]

        # If were running in DMO mode lets ensure we remove all particles
        if dmo:
            temp_npart = np.zeros(len(self.npart), dtype=int)
            temp_npart[1] = self.npart[1]
            self.npart = temp_npart
        self.part_types = [1, ] + [i for i in range(len(self.npart))
                                   if self.npart[i] != 0 and i != 1]

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
        self.llcoeff = llcoeff
        self.sub_llcoeff = sub_llcoeff
        self.ini_vlcoeff = ini_vlcoeff
        self.min_vlcoeff = min_vlcoeff
        self.decrement = decrement
        self.linkl = self.llcoeff * self.mean_sep
        self.sub_linkl = self.sub_llcoeff * self.mean_sep

        # Define the velocity space linking length
        self.vlinkl_indp = (np.sqrt(self.G / 2)
                            * (4 * np.pi * 200
                               * self.mean_den / 3) ** (1 / 6)).value
        self.sub_vlinkl_indp = self.vlinkl_indp * (self.linkl
                                                   / self.sub_linkl) ** (1 / 2)

        # Halo
        self.part_thresh = 10

        # Domain decomp
        self.cdim = int(cdim)
        self.ncells = self.cdim ** 3

        # Tasks
        self.spatial_task_size = 100000

        # Are we cleaning up the real_flags?
        self.clean_snaps = False

        # Linking metadata
        self.link_thresh = 10

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
