import numpy as np


class Halo:
    """
    Attributes:

    """

    # Predefine possible attributes to avoid overhead
    __slots__ = ["memory", "pids", "npart", "nprog", "prog_haloids",
                 "prog_npart", "prog_npart_cont", "prog_mass",
                 "prog_mass_cont", "ndesc",
                 "desc_haloids", "desc_npart", "desc_npart_cont",
                 "desc_mass", "desc_mass_cont"]

    def __init__(self, pids, npart, nprog, prog_haloids,
                 prog_npart, prog_npart_cont,
                 prog_mass, prog_mass_cont,
                 ndesc, desc_haloids,
                 desc_npart, desc_npart_cont,
                 desc_mass, desc_mass_cont):

        # Define metadata
        self.memory = None

        # Halo properties
        self.pids = pids
        if pids is not None:
            self.pids = np.array(pids, dtype=int)
        self.npart = npart

        # Progenitor properties
        self.nprog = nprog
        self.prog_haloids = prog_haloids
        self.prog_npart = prog_npart
        self.prog_npart_cont = prog_npart_cont
        self.prog_mass = prog_mass
        self.prog_mass_cont = prog_mass_cont

        # Descendant properties
        self.ndesc = ndesc
        self.desc_haloids = desc_haloids
        self.desc_npart = desc_npart
        self.desc_npart_cont = desc_npart_cont
        self.desc_mass = desc_mass
        self.desc_mass_cont = desc_mass_cont

    def update_progs(self, other_halo):

        # Loop over progenitors found for other_halo
        for other_ind, prog in enumerate(other_halo.prog_haloids):

            # If we already have this prog update it
            ind = np.where(self.prog_haloids == prog)[0]
            if len(ind) > 0:
                self.prog_npart_cont[ind] += other_halo.prog_npart_cont[other_ind]
            else:

                # Convert properties to lists
                self.prog_haloids = list(self.prog_haloids)
                self.prog_npart_cont = list(self.prog_npart_cont)
                self.prog_npart = list(self.prog_npart)

                # Include this new progenitor
                self.prog_haloids.append(prog)
                self.prog_npart_cont.append(
                    other_halo.prog_npart_cont[other_ind])
                self.prog_npart.append(other_halo.prog_npart[other_ind])

                # Convert back to arrays
                self.prog_haloids = np.array(self.prog_haloids, dtype=int)
                self.prog_npart_cont = np.array(self.prog_npart_cont,
                                                dtype=int)
                self.prog_npart = np.array(self.prog_npart, dtype=int)

        # Update the number of progendants
        self.nprog = len(self.prog_haloids)

    def update_descs(self, other_halo):

        # Loop over descendents found for other_halo
        for other_ind, desc in enumerate(other_halo.desc_haloids):

            # If we already have this desc update it
            ind = np.where(self.desc_haloids == desc)[0]
            if len(ind) > 0:
                self.desc_npart_cont[ind] += other_halo.desc_npart_cont[other_ind]
            else:

                # Convert properties to lists
                self.desc_haloids = list(self.desc_haloids)
                self.desc_npart_cont = list(self.desc_npart_cont)
                self.desc_npart = list(self.desc_npart)

                # Include this new descendant
                self.desc_haloids.append(desc)
                self.desc_npart_cont.append(
                    other_halo.desc_npart_cont[other_ind])
                self.desc_npart.append(other_halo.desc_npart[other_ind])

                # Convert back to arrays
                self.desc_haloids = np.array(self.desc_haloids, dtype=int)
                self.desc_npart_cont = np.array(self.desc_npart_cont,
                                                dtype=int)
                self.desc_npart = np.array(self.desc_npart, dtype=int)

        # Update the number of descendants
        self.ndesc = len(self.desc_haloids)

    def clean_progs(self, meta):

        # Remove progenitors that don't fulfill the link threshold
        okinds = self.prog_npart_cont >= meta.link_thresh
        self.prog_haloids = self.prog_haloids[okinds]
        self.prog_npart_cont = self.prog_npart_cont[okinds]
        self.prog_npart = self.prog_npart[okinds]
        self.nprog = len(self.prog_haloids)

        # Sort the results by contribution
        sinds = np.argsort(self.prog_npart_cont)[::-1]
        self.prog_haloids = self.prog_haloids[sinds]
        self.prog_npart_cont = self.prog_npart_cont[sinds]
        self.prog_npart = self.prog_npart[sinds]

    def clean_descs(self, meta):

        # Remove descendants that don't fulfill the link threshold
        okinds = self.desc_npart_cont >= meta.link_thresh
        self.desc_haloids = self.desc_haloids[okinds]
        self.desc_npart_cont = self.desc_npart_cont[okinds]
        self.desc_npart = self.desc_npart[okinds]
        self.ndesc = len(self.desc_haloids)

        # Sort the results by contribution
        sinds = np.argsort(self.desc_npart_cont)[::-1]
        self.desc_haloids = self.desc_haloids[sinds]
        self.desc_npart_cont = self.desc_npart_cont[sinds]
        self.desc_npart = self.desc_npart[sinds]


class Janitor_Halo:
    """
    Attributes:

    """

    # Predefine possible attributes to avoid overhead
    __slots__ = ["memory", "npart", "real", "nprog", "prog_haloids",
                 "prog_npart", "prog_npart_cont", "prog_mass",
                 "prog_mass_cont", "prog_reals", "ndesc",
                 "desc_haloids", "desc_npart", "desc_npart_cont",
                 "desc_mass", "desc_mass_cont"]

    def __init__(self, npart, real, prog_haloids, prog_npart, prog_npart_cont,
                 prog_mass, prog_mass_cont, prog_reals,
                 desc_haloids, desc_npart, desc_npart_cont,
                 desc_mass, desc_mass_cont):

        # Define metadata
        self.memory = None

        # Halo properties
        self.npart = npart
        self.real = real

        # Progenitor properties
        self.prog_reals = prog_reals[prog_reals]
        self.prog_haloids = prog_haloids[prog_reals]
        self.prog_npart = prog_npart[prog_reals]
        self.prog_npart_cont = prog_npart_cont[prog_reals]
        self.prog_mass = None
        self.prog_mass_cont = None
        self.nprog = len(self.prog_haloids)

        # Sort progenitors
        self.sort_progs()

        # Descendant properties
        self.desc_haloids = desc_haloids
        self.desc_npart = desc_npart
        self.desc_npart_cont = desc_npart_cont
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
        elif self.npart < 20 and self.nprog == 0:
            self.real = False
