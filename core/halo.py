import numpy as np

from halo_energy import kinetic, grav
import halo_properties as hprop


class Halo:

    # Predefine possible attributes to avoid overhead
    __slots__ = ["pids", "sim_pids", "pos", "vel", "types", "masses",
                 "part_KE", "part_GE", "vel_with_hubflow",
                 "npart", "real", "mean_pos", "mean_vel", "mean_vel_hubflow",
                 "mass", "ptype_mass", "KE", "GE",
                 "rms_r", "rms_vr", "veldisp3d", "veldisp1d", "vmax",
                 "hmr", "hmvr", "vlcoeff"]
    def __init__(self, pids, sim_pids, pos, vel, types, masses, vlcoeff,
                 boxsize, s, z, G, cosmo):

        # Particle information
        self.pids = np.array(pids, dtype=int)
        self.sim_pids = sim_pids
        self.pos = pos
        self.wrap_pos(boxsize)
        self.vel = vel
        self.types = types
        self.masses = masses

        # Halo properties
        # (some only populated when a halo exits phase space iteration)
        self.npart = len(self.pids)
        self.mass = np.sum(self.masses)
        self.ptype_mass = None
        self.rms_r = None
        self.rms_vr = None
        self.veldisp3d = None
        self.veldisp1d = None
        self.vmax = None
        self.hmr = None
        self.hmvr = None
        self.vlcoeff = vlcoeff

        # Calculate weighted mean position and velocities
        self.mean_pos = np.average(self.pos, weights=self.masses, axis=0)
        self.mean_vel = np.average(self.vel, weights=self.masses, axis=0)

        # Add the hubble flow to the velocities
        # *** NOTE: this DOES NOT include a gadget factor of a^-1/2 ***
        hubflow = cosmo.H(z).value * (self.pos - self.mean_pos)
        self.vel_with_hubflow = self.vel + hubflow
        self.mean_vel_hubflow = np.average(self.vel_with_hubflow,
                                           weights=self.masses, axis=0)

        # Centre position and velocity
        self.pos -= self.mean_pos
        self.vel -= self.mean_vel
        self.vel_with_hubflow -= self.mean_vel_hubflow

        # Energy properties
        self.part_KE = kinetic(self.vel, self.masses)
        self.KE = np.sum(self.part_KE)
        self.part_GE = grav(self.pos, self.npart, s, self.masses, z, G)
        self.GE = np.sum(self.part_GE)
        self.real = (self.KE / self.GE) <= 1

    def compute_props(self, G):

        # Get rms radii from the centred position and velocity
        self.rms_r = hprop.rms_rad(self.pos)
        self.rms_vr = hprop.rms_rad(self.vel_with_hubflow)

        # Compute the velocity dispersion
        self.veldisp3d, self.veldisp1d = hprop.vel_disp(self.vel_with_hubflow)

        # Compute maximal rotational velocity
        self.vmax = hprop.vmax(self.pos, self.masses, G)

        # Calculate half mass radius in position and velocity space
        self.hmr = hprop.half_mass_rad(self.pos, self.masses)
        self.hmvr = hprop.half_mass_rad(self.vel_with_hubflow, self.masses)

        # Define mass in each particle type
        self.ptype_mass = [np.sum(self.masses[self.types == i])
                           for i in range(6)]

    def decrement(self, decrement):
        self.vlcoeff -= decrement

    def wrap_pos(self, boxsize):

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
        self.pos[np.where(sep > 0.5 * boxsize)] += boxsize
