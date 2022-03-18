import numpy as np
import yaml
import h5py
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as const
import astropy.units as u


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

    def __init__(self, snapshot, cosmology,
                 llcoeff, sub_llcoeff, inputpath, savepath,
                 ini_vlcoeff, min_vlcoeff, decrement, verbose, findsubs,
                 cdim, profile, profile_path, h, softs, dmo):
        """

        :param snapshot:
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
        self.snap = snapshot
        hdf = h5py.File(inputpath + self.snap + ".hdf5", 'r')
        self.mean_sep = hdf["PartType1"].attrs['mean_sep']
        self.boxsize = hdf.attrs['boxsize']
        self.npart = hdf.attrs['npart']
        self.z = hdf.attrs['redshift']
        self.tot_mass = hdf["PartType1"].attrs['tot_mass'] * 10 ** 10
        hdf.close()
        if dmo:
            temp_npart = np.zeros(self.npart.size, dtype=np.int64)
            temp_npart[1] = self.npart[1]
            self.npart = temp_npart
        self.comoving_soft = softs[0]
        self.max_physical_soft = softs[1]
        mean_den = (self.tot_mass * u.M_sun / self.boxsize ** 3
                    / u.Mpc ** 3 * (1 + self.z) ** 3)
        self.mean_den = mean_den.to(u.M_sun / u.km ** 3)
        self.a = 1 / (1 + self.z)
        if self.comoving_soft * self.a > self.max_physical_soft:
            self.soft = self.max_physical_soft / self.a
        else:
            self.soft = self.comoving_soft

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
        self.profile_path = profile_path

        # Parameter file flags
        self.verbose = verbose
        self.findsubs = findsubs
        self.profile = profile
        self.dmo = dmo
        self.debug = False

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

        # Halo
        self.part_thresh = 10

        # Domain decomp
        self.cdim = int(cdim)
        self.ncells = self.cdim ** 3

        # Tasks
        self.spatial_task_size = 100000

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
