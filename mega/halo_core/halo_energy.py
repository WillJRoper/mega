import numpy as np
from scipy.spatial.distance import cdist

from mega.core.timing import timer


def halo_energy_calc_approx(halo_poss, halo_vels, halo_npart, masses, redshift,
                            G, h, soft):

    # Compute kinetic energy of the halo
    vel_disp = np.var(halo_vels, axis=0)
    KE = 0.5 * np.sum(masses * vel_disp) * 1 / (1 + redshift)

    halo_radii = np.sqrt(
        halo_poss[:, 0] ** 2 + halo_poss[:, 1] ** 2 + halo_poss[:, 2] ** 2)

    srtd_halo_radii = np.sort(halo_radii)

    n_within_radii = np.arange(0, halo_radii.size)
    GE = np.sum(G * masses ** 2 * n_within_radii / srtd_halo_radii)

    # Compute halo's energy
    halo_energy = KE - GE * h * (1 + redshift) * 1 / 3.086e+19

    return halo_energy


@timer("Kinetic-Energy")
def kinetic(tictoc, v, masses):

    # Compute kinetic energy of the halo
    v2 = v ** 2
    vel2 = np.sum([v2[:, 0], v2[:, 1], v2[:, 2]], axis=0)
    KE = np.sum(0.5 * vel2)

    return KE


@timer("Grav-Energy")
def grav(tictoc, meta, halo_poss, halo_npart, soft, masses):

    GE = 0

    # Compute gravitational potential energy
    for i in range(1, halo_npart):
        pos_i = np.array([halo_poss[i, :], ])
        dists = cdist(pos_i, halo_poss[:i, :], metric="sqeuclidean")
        GE += np.sum(masses[:i]
                     / np.sqrt(dists + soft ** 2))

    # Convert GE to be in the same units as KE (M_sun km^2 s^-2)
    GE = meta.G * GE * (1 + meta.z) * 1 / 3.086e+19 * 10 ** 10

    return GE


def halo_energy_calc_exact(halo_poss, halo_vels, halo_npart, masses, redshift,
                           G, h, soft):

    # Compute kinetic energy of the halo
    KE, KE_part = kinetic(halo_vels, masses)

    GE, GE_part = get_grav(halo_poss, halo_npart, soft, masses, redshift, G)

    # Compute halo's energy
    # KE *= 10.**10
    # GE *= 10.**20
    halo_energy = KE - GE

    return halo_energy, KE, GE
