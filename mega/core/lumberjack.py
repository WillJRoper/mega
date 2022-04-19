import numpy as np
import h5py
from kdhalofinder import read_sim
from shutil import copyfile
import multiprocessing as mp
from itertools import combinations
import functools as ft
from scipy.spatial import cKDTree
from scipy.spatial import distance
import astropy.constants as const
import astropy.units as u
import time
import numba as nb
import pprint
import warnings
import sys

warnings.filterwarnings('ignore')


@nb.njit(nogil=True, parallel=True)
def rms_rad(pos, cent):

    # Get the seperation between particles and halo centre
    sep = pos - cent

    # Get radii squared
    rad_sep = sep[:, 0]**2 + sep[:, 1]**2 + sep[:, 2]**2

    return np.sqrt(5 / 3 * 1 / rad_sep.size * np.sum(rad_sep))


def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] <= r
    return A[mask]


@nb.njit(nogil=True, parallel=True)
def kinetic(halo_vels, halo_npart, redshift, pmass):

    # Compute kinetic energy of the halo
    vel_disp = np.zeros(3, dtype=np.float32)
    for ixyz in [0, 1, 2]:
        vel_disp[ixyz] = np.var(halo_vels[:, ixyz])
    KE = 0.5 * halo_npart * pmass * np.sum(vel_disp) * 1 / (1 + redshift)

    return KE


@nb.njit(nogil=True, parallel=True)
def grav(rij_2, soft, pmass, redshift, h, G):

    # Compute the sum of the gravitational energy of each particle from
    # GE = G*Sum_i(m_i*Sum_{j<i}(m_j/sqrt(r_{ij}**2+s**2)))
    invsqu_dist = 1 / np.sqrt(rij_2 + soft ** 2)
    GE = G * pmass ** 2 * np.sum(invsqu_dist)

    # Convert GE to be in the same units as KE (M_sun km^2 s^-2)
    GE = GE * h * (1 + redshift) * 1 / 3.086e+19

    return GE


@nb.njit(nogil=True, parallel=False)
def get_seps_lm(halo_poss, halo_npart):

    # Compute the separations of all halo particles along each dimension
    seps = np.zeros((halo_npart, halo_npart, 3), dtype=np.float32)
    for ixyz in [0, 1, 2]:
        rows, cols = np.atleast_2d(halo_poss[:, ixyz], halo_poss[:, ixyz])
        seps[:, :, ixyz] = rows - cols.T

    # Compute the separation between all particles
    # NOTE: this is a symmetric matrix where we only need the upper right half
    rij2 = np.sum(seps * seps, axis=-1)

    return rij2


@nb.njit(nogil=True, parallel=True)
def get_grav_hm(halo_poss, halo_npart, soft, pmass, redshift, h, G):

    GE = 0

    for i in range(1, halo_npart):
        sep = (halo_poss[:i, :] - halo_poss[i, :])
        rij2 = np.sum(sep * sep, axis=-1)
        invsqu_dist = np.sum(1 / np.sqrt(rij2 + soft ** 2))

        GE += G * pmass ** 2 * invsqu_dist

    # Convert GE to be in the same units as KE (M_sun km^2 s^-2)
    GE = GE * h * (1 + redshift) * 1 / 3.086e+19

    return GE


@nb.jit(nogil=True, parallel=True)
def halo_energy_calc_exact(halo_poss, halo_vels, halo_npart, pmass, redshift, G, h, soft):

    # Compute kinetic energy of the halo
    KE = kinetic(halo_vels, halo_npart, redshift, pmass)

    if halo_npart < 10000:

        rij2 = get_seps_lm(halo_poss, halo_npart)

        # Extract only the upper triangle of rij
        rij_2 = upper_tri_masking(rij2)

        # Compute gravitational potential energy
        GE = grav(rij_2, soft, pmass, redshift, h, G)

    else:

        GE = get_grav_hm(halo_poss, halo_npart, soft, pmass, redshift, h, G)

    # Compute halo's energy
    halo_energy = KE - GE

    return halo_energy, KE, GE


@nb.jit(nogil=True, parallel=True)
def calc_overlap(halo1_poss, halo2_poss, halo1_vels, halo2_vels, boxsize):

    # Wrap halos and get spatial centres
    halo1_poss, cent1 = wrap_halo(halo1_poss, boxsize)
    halo2_poss, cent2 = wrap_halo(halo2_poss, boxsize)

    # Get halo velocity centres
    vcent1 = np.mean(halo1_vels, axis=0)
    vcent2 = np.mean(halo2_vels, axis=0)

    # Get halp radii
    R1 = rms_rad(halo1_poss, cent1)
    R2 = rms_rad(halo2_poss, cent2)
    vR1 = rms_rad(halo1_vels, vcent1)
    vR2 = rms_rad(halo2_vels, vcent2)

    # Calculate Overlap
    sep = cent1 - cent2
    sep[np.where(sep > 0.5 * boxsize)] -= boxsize
    sep[np.where(sep < 0.5 * -boxsize)] += boxsize
    d = np.sqrt(sep[0] ** 2 + sep[1] ** 2 + sep[2] ** 2)
    overlap = d / (R1 + R2)

    vsep = vcent1 - vcent2
    vd = np.sqrt(vsep[0] ** 2 + vsep[1] ** 2 + vsep[2] ** 2)
    voverlap = vd / (vR1 + vR2)

    return overlap, voverlap


@nb.jit(nogil=True, parallel=True)
def wrap_halo(halo_poss, boxsize):

    # Define the comparison particle as the maximum position in the current dimension
    max_part_pos = halo_poss.max(axis=0)

    # Compute all the halo particle separations from the maximum position
    sep = max_part_pos - halo_poss

    # If any separations are greater than 50% the boxsize (i.e. the halo is split over the boundary)
    # bring the particles at the lower boundary together with the particles at the upper boundary
    # (ignores halos where constituent particles aren't separated by at least 50% of the boxsize)
    # *** Note: fails if halo's extent is greater than 50% of the boxsize in any dimension ***
    halo_poss[np.where(sep > 0.5 * boxsize)] += boxsize

    # Compute the shifted mean position in the dimension ixyz
    mean_halo_pos = halo_poss.mean(axis=0)

    # Centre the halos about the mean in the dimension ixyz
    halo_poss -= mean_halo_pos

    return halo_poss, mean_halo_pos


def get_this_snap(halopath, snap):

    # Open the unsplit halo catalogue to get halo data
    old_hdf = h5py.File(halopath + 'halos_' + snap + '.hdf5', 'r')

    # Load this snapshots particle halo ID data
    parthaloids = old_hdf['Halo_IDs'][...]

    # Store halo realness in array for each halo ID where the ID is the index
    halos = np.unique(parthaloids)
    halos = halos[np.where(halos >= 0)]

    # Extract snapshot '061' halo IDs
    halos_bound = {}
    halos_energies = {}
    halo_partids = {}
    for ind, halo in enumerate(halos):

        halo = str(halo)

        print('This snapshot', snap, halo, end='\r')
        try:
            halo_partids[halo] = set(old_hdf[str(halo)]['Halo_Part_IDs'][...])
            halos_bound[halo] = old_hdf[str(halo)].attrs['Real']
            halos_energies[halo] = {'E': old_hdf[str(halo)].attrs['halo_energy'], 'KE': old_hdf[str(halo)].attrs['KE'],
                                    'GE': old_hdf[str(halo)].attrs['GE']}
        except KeyError:
            halos[ind] = -999
            continue

    halos = halos[np.where(halos != -999)]

    old_hdf.close()

    return halos_bound, halos_energies, halo_partids, halos


def get_desc_snap(newhalopath, desc_snap):

    # Open the unsplit halo catalogue to get halo particle ID data
    desc_hdf = h5py.File(newhalopath + 'halos_' + desc_snap + '.hdf5', 'r')

    # Load this snapshots particle halo ID data
    desc_parthaloids = np.int32(desc_hdf['Halo_IDs'][...])

    # Store halo realness in array for each halo ID where the ID is the index
    uni_descs = np.unique(desc_parthaloids)
    uni_descs = uni_descs[np.where(uni_descs >= 0)]
    all_real_descs = np.full(uni_descs.size, False)
    desc_partids = {}
    for desc in uni_descs:

        print('Descendent snapshot', desc_snap, desc, end='\r')
        try:
            # Store the real flag
            all_real_descs[desc] = desc_hdf[str(desc)].attrs['Real']

            # Get halo particle IDs
            desc_partids[str(desc)] = set(desc_hdf[str(desc)]['Halo_Part_IDs'][...])
        except KeyError:
            continue

    desc_hdf.close()

    return all_real_descs, desc_partids, desc_parthaloids


def haloSplitter(current_halo_partIDs, desc_IDs, desc_mass_cont, desc_partids):

    # Initialise dictionaries for new halos
    new_halos = {}
    masses = {}

    # Compute contribution fractions
    cont_fracs = desc_mass_cont / np.sum(desc_mass_cont)

    # Get main descendent halo ID
    main_halo = desc_IDs[np.argmax(cont_fracs)]

    # Loop through descendants
    for ind, desc in enumerate(desc_IDs):

        # Extract the particle IDs of this descendant
        desc_partIDs = desc_partids[str(desc)]

        # Remove particles not in common between the descendant and current halo
        new_halo_partIDs = desc_partIDs.intersection(current_halo_partIDs)

        # Add these to the new halo
        new_halos[desc] = set(new_halo_partIDs)

        # Remove these particles from the halo part IDs set eventually leaving only unassociated particles
        current_halo_partIDs = current_halo_partIDs - new_halos[desc]

    nonpersist_parts = current_halo_partIDs

    # Compute the number of unassociated particles
    unasso_num = len(nonpersist_parts)

    # Loop through mass contribution fractions to assign unassigned particles
    for frac, desc in zip(cont_fracs, desc_IDs):

        # Assign the remaining particles to each this halo defined by there contribution fraction
        # Must convert halo_partIDs from a set to a list and then back again
        masses[desc] = frac * unasso_num + len(new_halos[desc])

    return new_halos, masses, main_halo


def snapHaloSplitterLoop(snap, halopath, newhalopath, all):

    pid, pos, vel, npart, boxsize, redshift, t, rhocrit, pmass, h, linkl = read_sim(snap, PATH='snapshotdata/snapdir_',
                                                                                    llcoeff=0.2)

    # Sort the simulation data arrays by the particle ID
    sinds = pid.argsort()
    pos = pos[sinds, :]
    vel = vel[sinds, :]

    # Compute the softening length
    soft = 0.05 * boxsize / npart**(1./3.)

    # Define the gravitational constant
    G = (const.G.to(u.km**3 * u.M_sun**-1 * u.s**-2)).value

    # Define and convert particle mass to M_sun
    pmass *= 1e10 * 1 / h

    # Create HDF5 file for this snapshot
    new_hdf = h5py.File(newhalopath + 'halos_' + snap + '.hdf5', 'w')

    # Write out snapshot metadata
    new_hdf.attrs['linkingLength'] = linkl  # host halo linking length
    new_hdf.attrs['rhocrit'] = rhocrit  # critical density parameter
    new_hdf.attrs['redshift'] = redshift
    new_hdf.attrs['time'] = t

    new_hdf.close()

    # Compute the progenitor snapshot ID
    if int(snap) > 8:
        desc_snap = '0' + str(int(snap) + 1)
    else:
        desc_snap = '00' + str(int(snap) + 1)

    # Get descendant snapshot data
    dstart = time.time()
    all_real_descs, desc_partids, desc_parthaloids = get_desc_snap(newhalopath, desc_snap)
    print(f'Descendants loaded in: {time.time()-dstart} seconds')

    # print(sys.getsizeof(all_real_descs))
    # print(sys.getsizeof(desc_partids))
    # print(sys.getsizeof(desc_parthaloids))

    # Get this snapshots data
    thisstart = time.time()
    halos_bound, halos_energies, halo_partids, halos = get_this_snap(halopath, snap)
    print(f'Halos loaded in: {time.time() - thisstart} seconds')

    # print(sys.getsizeof(halos_bound))
    # print(sys.getsizeof(halos_energies))
    # print(sys.getsizeof(halo_partids))
    # print(sys.getsizeof(halos))

    # Initialise this snapshot's halo particle ID dictionary to store the data that will be written out
    snap_halo_pids = {}
    snap_halo_mass = {}
    snap_halo_mass_persist = {}
    split_dict = {}
    snap_halo_reals = {}
    snap_halo_Es = {}
    snap_halo_KEs = {}
    snap_halo_GEs = {}

    # Initialise the progress counter
    snap_progress = -1
    work_start = time.time()

    for num, halo in enumerate(halos):

        halo = str(halo)

        # Compute and print the progress
        previous_progress = snap_progress
        snap_progress = int(num / len(halos) * 100)
        if snap_progress != previous_progress:
            print('Snapshot ' + snap + ' Progress: ', snap_progress, '%', 'Elapsed time:', time.time() - work_start,
                  end='\r')

        # Extract the particle IDs of this halo
        current_halo_partIDs = halo_partids[halo]

        # Find the new descendants in the descendant snapshot which has already been split and only take
        # the unique values
        desc_IDs, desc_mass_cont = np.unique(desc_parthaloids[list(current_halo_partIDs), 0], return_counts=True)

        # Remove single particles if they have been found in the descendants
        desc_mass_cont = desc_mass_cont[np.where(desc_IDs >= 0)]
        desc_IDs = desc_IDs[np.where(desc_IDs >= 0)]

        # Remove single particles if they have been found in the descendants
        desc_IDs = desc_IDs[np.where(desc_mass_cont >= 10)]
        desc_mass_cont = desc_mass_cont[np.where(desc_mass_cont >= 10)]

        # If there is more than one descendant split the halo into halos defined by the mass ratio of descendants
        if len(desc_IDs) > 1:

            new_halos, masses, main_halo = haloSplitter(current_halo_partIDs, desc_IDs, desc_mass_cont, desc_partids)
            # print('split', len(new_halos.keys()))
            split_halos = list(new_halos.keys())

            new_halo_Es = {}

            if not all:

                for key in split_halos:

                    # Get the particle information for this halo
                    halo_pids = np.array(list(new_halos[key]), dtype=int)

                    if len(halo_pids) < 10:
                        del new_halos[key]
                        del masses[key]

                    halo_poss = pos[halo_pids]

                    halo_poss, mean_halo_pos = wrap_halo(halo_poss, boxsize)

                    halo_vels = vel[halo_pids]

                    halo_energy, KE, GE = halo_energy_calc_exact(halo_poss, halo_vels, halo_poss.shape[0],
                                                                 pmass, redshift, G, h, soft)

                    new_halo_Es[key] = {'E': halo_energy, 'KE': KE, 'GE': GE}

                # print(len(new_halos.keys()), list(new_halos.keys()))
                glue_start = time.time()

                overlap_halos = {}

                # Assingn the most massive halo to halo1 for comparison with all other halos
                halo1 = main_halo

                # Loop over halo combinations calculating overlap
                for halo2 in new_halos.keys():

                    if halo1 == halo2:
                        continue

                    halo1_energy = new_halo_Es[halo1]['E']
                    halo2_energy = new_halo_Es[halo2]['E']

                    if halo1_energy > 0 or halo2_energy > 0:

                        halo1_pids = list(new_halos[halo1])
                        halo2_pids = list(new_halos[halo2])

                        halo1_poss = pos[halo1_pids]
                        halo2_poss = pos[halo2_pids]

                        halo1_vels = vel[halo1_pids]
                        halo2_vels = vel[halo2_pids]

                        overlap, voverlap = calc_overlap(halo1_poss, halo2_poss, halo1_vels, halo2_vels, boxsize)

                        if overlap + voverlap < 0.85:
                            overlap_halos.setdefault(halo1, set()).update({halo2, halo1})
                            overlap_halos.setdefault(halo2, set()).update({halo1, halo2})

                for sphalo in overlap_halos.keys():

                    for spolhalo in overlap_halos[sphalo]:
                        overlap_halos[spolhalo].update(overlap_halos[sphalo])

                if len(overlap_halos.keys()) != 0:

                    for olkey in overlap_halos.keys():

                        overlap_halos[olkey] = set(np.sort(list(overlap_halos[olkey])))

                    overlap_arr = [list(x) for x in set(tuple(x) for x in overlap_halos.values())]

                    for overlaps in overlap_arr:

                        key = 1000000 + overlaps[0]
                        masses[key] = 0

                        for sphalo in overlaps:
                            new_halos.setdefault(key, set()).update(new_halos[sphalo])
                            masses[key] += masses[sphalo]

                            del new_halos[sphalo]
                            del masses[sphalo]

                # print(len(new_halos.keys()), list(new_halos.keys()))
                #
                # print('Glue: ', time.time() - glue_start)

            count = 0
            split_count = 0

            for key in new_halos.keys():

                # Get the particle information for this halo
                halo_pids = np.array(list(new_halos[key]), dtype=int)

                # Extract halo data
                halo_poss = pos[halo_pids, :]  # Positions *** NOTE: these are shifted below ***
                halo_vels = vel[halo_pids, :]  # Velocities *** NOTE: these are shifted below ***
                persistent_halo_npart = halo_pids.size

                # Compute mean positions and wrap the halos
                halo_poss, mean_halo_pos = wrap_halo(halo_poss, boxsize)

                # Compute halo's energy
                if key in new_halo_Es.keys():
                    halo_energy, KE, GE = new_halo_Es[key]['E'], new_halo_Es[key]['KE'], new_halo_Es[key]['GE']
                else:
                    estart = time.time()
                    halo_energy, KE, GE = halo_energy_calc_exact(halo_poss, halo_vels, persistent_halo_npart,
                                                                 pmass, redshift, G, h, soft)
                    # print('Energy: ', halo_energy, persistent_halo_npart, time.time() - estart)

                if halo_energy > 0:
                    real = False
                else:
                    real = True

                # Assign this halo's particles to the halo particle dictionary
                snap_halo_pids[str(halo) + '.' + str(count)] = list(new_halos[key])
                snap_halo_mass[str(halo) + '.' + str(count)] = masses[key]
                snap_halo_mass_persist[str(halo) + '.' + str(count)] = len(new_halos[key])

                # Store realness
                snap_halo_reals[str(halo) + '.' + str(count)] = real
                snap_halo_Es[str(halo) + '.' + str(count)] = halo_energy
                snap_halo_KEs[str(halo) + '.' + str(count)] = KE
                snap_halo_GEs[str(halo) + '.' + str(count)] = GE

                count += 1
                split_count += 1

            for i in range(count):

                # Assign split number
                split_dict[str(halo) + '.' + str(i)] = split_count

        elif len(desc_IDs) == 1:

            # Assign split boolean
            split_dict[halo] = 1

            # Extract the particle IDs of this descendant
            desc_partIDs = desc_partids[str(desc_IDs[0])]

            # Remove particles not in common between the descendant and current halo
            halo_pids = np.array(list(desc_partIDs.intersection(current_halo_partIDs)), dtype=int)

            # Assign this halo's particles to the halo particle dictionary
            snap_halo_pids[halo] = list(halo_partids[halo])
            snap_halo_mass[halo] = len(halo_partids[halo])
            snap_halo_mass_persist[halo] = len(halo_pids)

            # Extract halo data
            halo_poss = pos[halo_pids, :]  # Positions *** NOTE: these are shifted below ***
            halo_vels = vel[halo_pids, :]  # Velocities *** NOTE: these are shifted below ***
            persistent_halo_npart = halo_pids.size

            # Compute mean positions and wrap the halos
            halo_poss, mean_halo_pos = wrap_halo(halo_poss, boxsize)

            halo_energy, KE, GE = halo_energy_calc_exact(halo_poss, halo_vels, persistent_halo_npart,
                                                         pmass, redshift, G, h, soft)

            # Αssign realness
            snap_halo_reals[halo] = halos_bound[halo]
            snap_halo_Es[halo] = halo_energy
            snap_halo_KEs[halo] = KE
            snap_halo_GEs[halo] = GE

    print('Snapshot ' + snap + ' Progress: 100%', '\nElapsed time:', time.time() - work_start)

    write_start = time.time()

    # Write out the halos that have been found
    # Open the root group for the new halo catalogue
    new_hdf = h5py.File(newhalopath + 'halos_' + snap + '.hdf5', 'r+')

    # Define snapshots particle halo ID array
    part_haloids = np.full(npart, -2, dtype=int)

    newID = -1

    for key in snap_halo_pids.keys():

        halo_pids = snap_halo_pids[key]

        # Increment the new halo ID
        newID += 1

        # Assign this halo to the particle halo ID array
        part_haloids[halo_pids, 0] = newID

        # Create datasets in the current halo's group in the HDF5 file and assign halo data
        # *** NOTE: this is the minimal data to make the tree currently and can be expanded upon ***
        halo_group = new_hdf.create_group(str(newID))  # create halo group
        halo_group.create_dataset('Halo_Part_IDs', shape=[len(halo_pids)],
                                  dtype=int, compression='gzip', data=halo_pids)  # halo particle ids
        halo_group.attrs['halo_nPart'] = snap_halo_mass[key]
        halo_group.attrs['halo_nPart_persist'] = snap_halo_mass_persist[key]
        halo_group.attrs['splitting'] = split_dict[key]
        halo_group.attrs['Real'] = snap_halo_reals[key]
        halo_group.attrs['halo_energy'] = snap_halo_Es[key]
        halo_group.attrs['KE'] = snap_halo_KEs[key]
        halo_group.attrs['GE'] = snap_halo_GEs[key]

        if '.' in key:
            halo_group.attrs['originalID'] = key.split('.')[0]
        else:
            halo_group.attrs['originalID'] = key

        # Compute and print the progress
        previous_progress = snap_progress
        snap_progress = int(newID / len(halos) * 100 + 1)
        if snap_progress != previous_progress:
            print('Snapshot ' + snap + ' Writing Progress: ', snap_progress, '%', 'Elapsed time:',
                  time.time() - write_start, end='\r')

    # Write out the particle halo ID array
    new_hdf.create_dataset('Halo_IDs', shape=part_haloids.shape, dtype=int, compression='gzip', data=part_haloids)

    new_hdf.close()

    print(snap + ' Writing progress: 100%', '\nElapsed time:', time.time() - write_start)


def mainLumberjack(halopath='halo_snapshots/', newhalopath='split_halos/', all=False):

    start = time.time()
    # Copy the present day snapshot to the new halo catalog
    copyfile(halopath + 'halos_061.hdf5', newhalopath + 'halos_061.hdf5')

    # Create the snapshot list (present day to past) for looping
    snaplist = []
    for snap in range(60, -1, -1):
        if snap < 10:
            snaplist.append('00' + str(snap))
        elif snap >= 10:
            snaplist.append('0' + str(snap))

    print('061: ', time.time() - start)

    # Loop through snapshots
    for snap in snaplist:

        # Run the halo splitting loop for this snapshot, looping over all halos in
        # the snapshot, splitting them and writing them out.
        start = time.time()
        snapHaloSplitterLoop(snap, halopath, newhalopath, all)
        print(snap, ': ', time.time() - start)


if __name__ == '__main__':
    mainLumberjack(halopath='halo_snapshots/', newhalopath='all_split_halos/', all=True)
