import numpy as np
from scipy.spatial.distance import cdist

from core.timing import timer


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
    KE_part = 0.5 * masses * vel2

    return KE_part


@timer("Grav-Energy")
def grav(tictoc, halo_poss, halo_npart, soft, masses, redshift, G):

    GE_part = np.zeros(halo_poss.shape[0])

    # Compute distances
    dists = cdist(halo_poss, halo_poss, metric="sqeuclidean")

    # Compute gravitational potential energy
    for i in range(1, halo_npart):
        GE_part[i] = np.sum(masses[i] * masses[:i]
                            / np.sqrt(dists[i, :i] + soft ** 2))

    # Convert GE to be in the same units as KE (M_sun km^2 s^-2)
    GE_part = G * np.float64(GE_part) * (1 + redshift) * 1 / 3.086e+19

    return GE_part


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


def halo_energy_calc_cdist(halo_poss, halo_vels, halo_npart, masses, redshift,
                           G, h, soft):

    # Compute kinetic energy of the halo
    KE = kinetic(halo_vels, halo_npart, redshift, masses)

    # Calculate separations
    rij = cdist(halo_poss, halo_poss, metric='sqeuclidean')

    # Extract only the upper triangle of rij
    rij_2 = upper_tri_masking(rij)

    # Calculate the gravitational potential
    GE = grav(rij_2, soft, masses, redshift, G)

    # Compute halo's energy
    halo_energy = KE - GE

    return halo_energy, KE, GE
