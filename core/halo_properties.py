import numpy as np


def rms_rad(coord):

    # Calculate the radii
    rs = np.linalg.norm(coord, axis=1)

    # Calculate the rms radius
    rms_r = np.sqrt(np.mean(rs**2))

    return rms_r


def vel_disp(vels):

    veldisp1d = np.std(vels, axis=0)
    veldisp3d = np.sqrt(veldisp1d[0]**2 + veldisp1d[1]**2 + veldisp1d[2]**2)

    return veldisp3d, veldisp1d


def half_mass_rad(rs, weight):

    # Sort the radii and masses
    sinds = np.argsort(rs)
    rs = rs[sinds]
    weight = weight[sinds]

    # Get the cumulative sum of masses
    weight_profile = np.cumsum(weight)

    # Get the total mass and half the total mass
    tot_weight = np.sum(weight)
    half_weight = tot_weight / 2

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(weight_profile - half_weight))
    hmr = rs[hmr_ind]

    return hmr


def vmax(rs, masses, G):

    # Sort the radii and masses
    sinds = np.argsort(rs)
    rs = rs[sinds]
    masses = masses[sinds]

    # Get the cumulative sum of masses
    mass_profile = np.cumsum(masses)

    # Get the velocity of each particle v = G M / r
    vs = G * mass_profile / rs

    return np.max(vs)
