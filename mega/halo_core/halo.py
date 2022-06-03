import mega.halo_core.halo_properties as hprop
import numpy as np
from mega.core.talking_utils import get_heading, pad_print_middle
from mega.halo_core.halo_energy import kinetic, grav


class Halo:
    """
    Attributes:
    """

    # Predefine possible attributes to avoid overhead
    __slots__ = ["memory", "print_props", "parent", "pids", "shifted_inds",
                 "sim_pids", "true_pos", "pos", "true_vel",
                 "vel", "types", "masses", "int_nrg",
                 "vel_with_hubflow", "npart", "npart_types", "real",
                 "mean_pos", "mean_vel", "mean_vel_hubflow", "mass",
                 "ptype_mass", "KE", "therm_nrg", "GE", "rms_r", "rms_vr",
                 "veldisp3d", "veldisp1d", "vmax", "hmr", "hmvr", "vlcoeff"]

    def __init__(self, tictoc, pids, shifted_pids, sim_pids, pos, vel, types,
                 masses, int_nrg, vlcoeff, meta, parent=None):
        """
        :param pids:
        :param sim_pids:
        :param pos:
        :param vel:
        :param types:
        :param masses:
        :param vlcoeff:
        :param boxsize:
        :param s:
        :param z:
        :param G:
        :param cosmo:
        """

        # Define metadata
        # self.meta = meta
        self.memory = None
        self.print_props = ["npart", "npart_types", "KE", "GE", "real",
                            "mean_pos", "mean_vel", "mass", "ptype_mass"]

        # Set the current point of the phase space iteration
        self.vlcoeff = vlcoeff

        # # We need to remove some variables for meta that won't be need in the
        # # halo object and cause issues when calculating memory footprint
        # # TODO: This would be better dealt with by having a light weight meta
        # #  with the necessary properties used in the print function
        # self.meta.cosma = None
        # self.meta.crit_density = None
        # self.meta.omega_m = None
        # self.meta.mean_den = None

        # Store the parent halo, None if no parent
        self.parent = parent

        # If the parent has the same number of particles we don't
        # need to recalculate everything and can inherit the parent's
        # properties
        if parent is not None:
            if len(pids) == parent.npart:
                self.set_attrs_from_parent(parent)
            else:
                self.set_attrs(tictoc, pids, shifted_pids, sim_pids, pos, vel,
                               types, masses, int_nrg, meta)
        else:  # we have a brand new halo
            self.set_attrs(tictoc, pids, shifted_pids, sim_pids, pos, vel, types,
                           masses, int_nrg, meta)

    def __str__(self):

        # Set up string for printing
        pstr = ""

        # Print a heading for this halo
        report_string = get_heading(60, "Halo")
        pstr += "|" + report_string + "|" + "\n" + " " * 9

        # Loop over properties to print
        for prop in self.print_props:
            # Get property value
            pstr += "|" + pad_print_middle(prop, getattr(self, prop),
                                           length=60) + "|" + "\n" + " " * 9

        pstr += "|" + "=" * len(report_string) + "|"

        return pstr

    def set_attrs(self, tictoc, pids, shifted_pids, sim_pids, pos, vel, types,
                  masses, int_nrg, meta):

        # Particle information
        self.pids = np.array(pids, dtype=int)
        self.shifted_inds = shifted_pids
        self.sim_pids = sim_pids
        self.pos = pos
        self.true_pos = pos
        if meta.periodic:
            self.wrap_pos(meta.boxsize)
        self.vel = vel
        self.true_vel = vel
        self.types = types
        self.masses = masses
        self.int_nrg = int_nrg

        # Halo properties
        # (some only populated when a halo exits phase space iteration)
        self.npart = len(self.pids)
        self.npart_types = [len(self.pids[types == i])
                            for i in range(len(meta.npart))]
        self.mass = np.sum(self.masses)
        self.ptype_mass = None
        self.rms_r = None
        self.rms_vr = None
        self.veldisp3d = None
        self.veldisp1d = None
        self.vmax = None
        self.hmr = None
        self.hmvr = None

        # Calculate weighted mean position and velocities
        self.mean_pos = np.average(self.true_pos, weights=self.masses, axis=0)
        self.mean_vel = np.average(self.vel, weights=self.masses, axis=0)

        # Centre position and velocity
        self.pos -= self.mean_pos
        self.vel -= self.mean_vel

        # Add the hubble flow to the velocities
        # *** NOTE: this DOES NOT include a gadget factor of a^-1/2 ***
        hubflow = meta.cosmo.H(meta.z).value * self.pos
        self.vel_with_hubflow = self.vel + hubflow
        self.mean_vel_hubflow = np.average(self.vel,
                                           weights=self.masses, axis=0)

        # Energy properties (energies are in per unit mass units)
        self.KE = kinetic(tictoc, self.vel_with_hubflow, self.masses)
        self.therm_nrg = np.sum(self.int_nrg)
        self.GE = grav(tictoc, self.pos, self.npart, meta.soft,
                       self.masses, meta.z, meta.G)
        self.real = (np.log10(10 ** self.KE + self.therm_nrg) / self.GE) <= 1

    def set_attrs_from_parent(self, parent):

        # Particle information
        self.pids = parent.pids
        self.shifted_inds = parent.shifted_inds
        self.sim_pids = parent.sim_pids
        self.pos = parent.pos
        self.true_pos = parent.true_pos
        self.vel = parent.vel
        self.true_vel = parent.true_vel
        self.types = parent.types
        self.masses = parent.masses
        self.int_nrg = parent.int_nrg

        # Halo properties
        # (some only populated when a halo exits phase space iteration)
        self.npart = parent.npart
        self.npart_types = parent.npart_types
        self.mass = parent.mass
        self.ptype_mass = None
        self.rms_r = None
        self.rms_vr = None
        self.veldisp3d = None
        self.veldisp1d = None
        self.vmax = None
        self.hmr = None
        self.hmvr = None

        # Calculate weighted mean position and velocities
        self.mean_pos = parent.mean_pos
        self.mean_vel = parent.mean_vel

        # Add the hubble flow to the velocities
        # *** NOTE: this DOES NOT include a gadget factor of a^-1/2 ***
        self.vel_with_hubflow = parent.vel_with_hubflow
        self.mean_vel_hubflow = parent.mean_vel_hubflow

        # Energy properties (energies are in per unit mass units)
        self.KE = parent.KE
        self.therm_nrg = parent.therm_nrg
        self.GE = parent.GE
        self.real = parent.real

    def compute_props(self, meta):
        """
        :param G:
        :return:
        """

        # Get rms radii from the centred position and velocity
        self.rms_r = hprop.rms_rad(self.pos)
        self.rms_vr = hprop.rms_rad(self.vel_with_hubflow)

        # Compute the velocity dispersion
        self.veldisp3d, self.veldisp1d = hprop.vel_disp(self.vel_with_hubflow)

        # Compute maximal rotational velocity
        self.vmax = hprop.vmax(self.pos, self.masses, meta.G)

        # Calculate half mass radius in position and velocity space
        self.hmr = hprop.half_mass_rad(self.pos, self.masses)
        self.hmvr = hprop.half_mass_rad(self.vel_with_hubflow, self.masses)

        # Define mass in each particle type
        self.ptype_mass = [np.sum(self.masses[self.types == i])
                           for i in range(len(meta.npart))]

    def decrement(self, decrement):
        """
        :param decrement:
        :return:
        """
        self.vlcoeff = 10 ** (np.log10(self.vlcoeff) - decrement)

    def wrap_pos(self, l):
        """
        :param boxsize:
        :return:
        """

        # Define the comparison particle as the maximum
        # position in the current dimension
        max_part_pos = self.pos.max(axis=0)

        # Compute all the halo particle separations from the maximum position
        sep = max_part_pos - self.pos

        # If any separations are greater than 50% the boxsize
        # (i.e. the halo is split over the boundary)
        # bring the particles at the lower boundary together
        # with the particles at the upper boundary
        # (ignores halos where constituent particles aren't
        # separated by at least 50% of the boxsize)
        # *** Note: fails if halo's extent is greater than 50%
        # of the boxsize in any dimension ***
        for ixyz in range(l.size):
            self.pos[ixyz][np.where(sep[ixyz] > 0.5 * l[ixyz])] += l[ixyz]

    def clean_halo(self):
        """ A helper method to clean memory hogging attributes to limit the
            memory in communications containing output halos.
        :return:
        """
        # Remove attributes that are no longer
        del self.pos
        del self.true_pos
        del self.vel
        del self.true_vel
        del self.masses
        del self.parent
