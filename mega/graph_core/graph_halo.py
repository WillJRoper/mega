import numpy as np


class Halo:
    """
    Attributes:

    """

    # Predefine possible attributes to avoid overhead
    __slots__ = ["memory", "pids", "part_types", "part_masses",
                 "npart", "mass", "mean_pos", "mean_vel", "real",  "halo_id",
                 "nprog", "prog_haloids", "prog_npart", "prog_npart_cont",
                 "prog_npart_cont_type", "prog_mass", "prog_mass_cont",
                 "ndesc",  "desc_haloids", "desc_npart", "desc_npart_cont",
                 "desc_npart_cont_type", "desc_mass", "desc_mass_cont", ]

    def __init__(self, pids, part_types, part_masses, npart, mean_pos,
                 mean_vel, real, halo_id,  meta):

        # Profiling variables
        self.memory = None

        # Particle information
        self.pids = np.array(pids, dtype=int)
        self.part_types = np.array(part_types, dtype=int)
        self.part_masses = np.array(pids, dtype=np.float64)

        # Halo properties
        self.npart = npart
        self.mass = np.array([np.sum(part_masses[part_types == ptype])
                              for ptype in range(len(meta.npart))])
        self.mean_pos = mean_pos
        self.mean_vel = mean_vel
        self.real = real
        self.halo_id = halo_id

        # Progenitor properties
        self.nprog = 0
        self.prog_haloids = []
        self.prog_npart = []
        self.prog_npart_cont = []
        self.prog_npart_cont_type = []
        self.prog_mass = []
        self.prog_mass_cont = []

        # Descendant properties
        self.ndesc = 0
        self.desc_haloids = []
        self.desc_npart = []
        self.desc_npart_cont = []
        self.desc_npart_cont_type = []
        self.desc_mass = []
        self.desc_mass_cont = []

    def compare_prog(self, prog, meta):

        # Get the particles in common
        pids_common, _, prog_inds = np.intersect1d(
            self.pids, prog.pids, return_indices=True
        )

        # Define the number of particles contributed
        ncont = pids_common.size

        # If we have found no particles in common exit early
        if ncont == 0:
            return

        # Otherwise lets include this progenitor
        self.nprog += 1
        self.prog_haloids.append(prog.halo_id)
        self.prog_npart.append(prog.npart)
        self.prog_mass.append(prog.mass)

        # Get the masses and types in common
        mass_common = prog.part_masses[prog_inds]
        type_common = prog.part_types[prog_inds]

        # Calculate contributions
        self.prog_npart_cont.append(mass_common.size)
        self.prog_npart_cont_type.append(
            [mass_common[type_common == ptype].size
             for ptype in range(len(meta.npart))]
        )
        self.prog_mass_cont.append([np.sum(mass_common[type_common == ptype])
                                    for ptype in range(len(meta.npart))])

    def compare_desc(self, desc, meta):

        # Get the particles in common
        pids_common, _, desc_inds = np.intersect1d(
            self.pids, desc.pids, return_indices=True
        )

        # Define the number of particles contributed
        ncont = pids_common.size

        # If we have found no particles in common exit early
        if ncont == 0:
            return

        # Otherwise lets include this progenitor
        self.ndesc += 1
        self.desc_haloids.append(desc.halo_id)
        self.desc_npart.append(desc.npart)
        self.desc_mass.append(desc.mass)

        # Get the masses and types in common
        mass_common = desc.part_masses[desc_inds]
        type_common = desc.part_types[desc_inds]

        # Calculate contributions
        self.desc_npart_cont.append(mass_common.size)
        self.desc_npart_cont_type.append(
            [mass_common[type_common == ptype].size
             for ptype in range(len(meta.npart))]
        )
        self.desc_mass_cont.append([np.sum(mass_common[type_common == ptype])
                                    for ptype in range(len(meta.npart))])

    def clean_progs(self, meta):

        # Convert to arrays
        self.prog_haloids = np.array(self.prog_haloids)
        self.prog_npart_cont = np.array(self.prog_npart_cont)
        self.prog_npart_cont_type = np.array(self.prog_npart_cont_type)
        self.prog_npart = np.array(self.prog_npart)
        self.prog_mass = np.array(self.prog_mass)
        self.prog_mass_cont = np.array(self.prog_mass_cont)

        # Handle array shape for 2D arrays with no progentiors
        if self.nprog == 0:
            self.prog_npart_cont_type = np.empty((0, len(meta.npart)))
            self.prog_mass_cont = np.empty((0, len(meta.npart)))

        # Remove progenitors that don't fulfill the link threshold
        okinds = self.prog_npart_cont >= meta.link_thresh
        self.prog_haloids = self.prog_haloids[okinds]
        self.prog_npart_cont = self.prog_npart_cont[okinds]
        self.prog_npart_cont_type = self.prog_npart_cont_type[okinds, :]
        self.prog_npart = self.prog_npart[okinds]
        self.prog_mass = self.prog_mass[okinds]
        self.prog_mass_cont = self.prog_mass_cont[okinds, :]
        self.nprog = len(self.prog_haloids)

        # Sort the results by contribution
        sinds = np.argsort(self.prog_npart_cont)[:: -1]
        self.prog_haloids = self.prog_haloids[sinds]
        self.prog_npart_cont = self.prog_npart_cont[sinds]
        self.prog_npart_cont_type = self.prog_npart_cont_type[sinds, :]
        self.prog_mass = self.prog_mass[sinds]
        self.prog_mass_cont = self.prog_mass_cont[sinds, :]
        self.prog_npart = self.prog_npart[sinds]

    def clean_descs(self, meta):

        # Convert to arrays
        self.desc_haloids = np.array(self.desc_haloids)
        self.desc_npart_cont = np.array(self.desc_npart_cont)
        self.desc_npart_cont_type = np.array(self.desc_npart_cont_type)
        self.desc_npart = np.array(self.desc_npart)
        self.desc_mass = np.array(self.desc_mass)
        self.desc_mass_cont = np.array(self.desc_mass_cont)

        # Handle array shape for 2D arrays with no progentiors
        if self.ndesc == 0:
            self.desc_npart_cont_type = np.empty((0, len(meta.npart)))
            self.desc_mass_cont = np.empty((0, len(meta.npart)))

        # Remove descendants that don't fulfill the link threshold
        okinds = self.desc_npart_cont >= meta.link_thresh
        self.desc_haloids = self.desc_haloids[okinds]
        self.desc_npart_cont = self.desc_npart_cont[okinds]
        self.desc_npart_cont_type = self.desc_npart_cont_type[okinds, :]
        self.desc_npart = self.desc_npart[okinds]
        self.desc_mass = self.desc_mass[okinds]
        self.desc_mass_cont = self.desc_mass_cont[okinds, :]
        self.ndesc = len(self.desc_haloids)

        # Sort the results by contribution
        sinds = np.argsort(self.desc_npart_cont)[:: -1]
        self.desc_haloids = self.desc_haloids[sinds]
        self.desc_npart_cont = self.desc_npart_cont[sinds]
        self.desc_npart_cont_type = self.desc_npart_cont_type[sinds, :]
        self.desc_mass = self.desc_mass[sinds]
        self.desc_mass_cont = self.desc_mass_cont[sinds, :]
        self.desc_npart = self.desc_npart[sinds]

    def clean_halo(self):

        del self.pids
        del self.part_types
        del self.part_masses
        del self.mean_pos


class LinkHalo:

    # Predefine possible attributes to avoid overhead
    __slots__ = ["memory", "pids", "part_types", "part_masses", "npart",
                 "mass", "mean_pos", "real", "halo_id"]

    def __init__(self, pids, part_types, part_masses, npart, mean_pos, real,
                 halo_id, meta):

        # Profiling variables
        self.memory = None

        # Particle information
        self.pids = np.array(pids, dtype=int)
        self.part_types = np.array(part_types, dtype=int)
        self.part_masses = np.array(pids, dtype=np.float64)

        # Halo properties
        self.npart = npart
        self.mass = np.array([np.sum(part_masses[part_types == ptype])
                              for ptype in range(len(meta.npart))])
        self.mean_pos = mean_pos
        self.real = real
        self.halo_id = halo_id


class Janitor_Halo:
    """
    Attributes:

    """

    # Predefine possible attributes to avoid overhead
    __slots__ = ["memory", "npart", "real", "mass",
                 "nprog", "prog_haloids", "prog_npart", "prog_npart_cont",
                 "prog_npart_cont_type", "prog_mass", "prog_mass_cont",
                 "prog_reals",
                 "ndesc",  "desc_haloids", "desc_npart", "desc_npart_cont",
                 "desc_npart_cont_type", "desc_mass", "desc_mass_cont", ]

    def __init__(self, npart, mass, real,
                 prog_haloids, prog_npart, prog_npart_cont,
                 prog_mass, prog_mass_cont, prog_reals,
                 desc_haloids, desc_npart, desc_npart_cont,
                 desc_mass, desc_mass_cont):

        # Define metadata
        self.memory = None

        # Halo properties
        self.npart = npart
        self.real = real
        self.mass = mass

        # Progenitor properties
        self.prog_reals = prog_reals[prog_reals]
        self.prog_haloids = prog_haloids[prog_reals]
        self.prog_npart = prog_npart[prog_reals]
        self.prog_npart_cont = np.sum(prog_npart_cont[prog_reals, :],
                                      axis=1)
        self.prog_npart_cont_type = prog_npart_cont[prog_reals, :]
        self.prog_mass = prog_mass[prog_reals, :]
        self.prog_mass_cont = prog_mass_cont[prog_reals, :]
        self.nprog = len(self.prog_haloids)

        # Sort progenitors
        self.sort_progs()

        # Descendant properties
        self.desc_haloids = desc_haloids
        self.desc_npart = desc_npart
        self.desc_npart_cont = np.sum(desc_npart_cont, axis=1)
        self.desc_npart_cont_type = desc_npart_cont
        self.desc_mass = desc_mass
        self.desc_mass_cont = desc_mass_cont
        self.ndesc = len(self.desc_haloids)

        # Update realness flag
        self.is_real()

    def sort_progs(self):

        # Get indices to sort progenitors
        sinds = np.argsort(self.prog_npart_cont)[::-1]

        # Apply these indices
        self.prog_reals = self.prog_reals[sinds]
        self.prog_haloids = self.prog_haloids[sinds]
        self.prog_npart = self.prog_npart[sinds]
        self.prog_npart_cont = self.prog_npart_cont[sinds]

    def is_real(self):

        # Consider if this halo is real
        # If neither boolean is satisfied it inherits the energy defined
        # realness flag
        if self.nprog > 0:
            self.real = True
        elif np.sum(self.npart) < 20 and self.nprog == 0:
            self.real = False
