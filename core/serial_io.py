import h5py
import numpy as np

from core.halo import Halo
from core.talking_utils import message
from core.timing import timer


# TODO: If we ever need to split the io (particularly creation and writing
#  out of part_haloids arrays) we split the array over nranks and have each
#  rank handshake the particles it has onto the corresponding ranks.


def hdf5_write_dataset(grp, key, data, compression="gzip"):
    """

    :param grp:
    :param key:
    :param data:
    :param compression:
    :return:
    """

    grp.create_dataset(key, shape=data.shape, dtype=data.dtype, data=data,
                       compression=compression)


@timer("Reading")
def read_subset(tictoc, meta, key, subset):
    """

    :param tictoc:
    :param meta:
    :param key:
    :param subset:
    :return:
    """

    # Open hdf5 file
    hdf = h5py.File(meta.inputpath + meta.snap + ".hdf5", "r")

    # Get the positions for searching on this rank
    # NOTE: for now it"s more efficient to read all particles
    # and extract the particles we need and throw away the ones
    # we don"t, could be problematic with large datasets
    all_arr = hdf[key][...]

    hdf.close()

    arr = all_arr[subset]

    return arr


def read_subset_fromobj(hdf, key, subset):
    """

    :param hdf:
    :param key:
    :param subset:
    :return:
    """

    # Get the positions for searching on this rank
    # NOTE: for now it"s more efficient to read all particles
    # and extract the particles we need and throw away the ones
    # we don"t, could be problematic with large datasets
    all_arr = hdf[key][...]

    arr = all_arr[subset]

    return arr


@timer("Reading")
def read_halo_data(tictoc, part_inds, inputpath, snapshot, hdf_part_key,
                   ini_vlcoeff, boxsize, soft, redshift, G, cosmo):
    """

    :param tictoc:
    :param part_inds:
    :param inputpath:
    :param snapshot:
    :param hdf_part_key:
    :param ini_vlcoeff:
    :param boxsize:
    :param soft:
    :param redshift:
    :param G:
    :param cosmo:
    :return:
    """

    # Open hdf5 file
    hdf = h5py.File(inputpath + snapshot + ".hdf5", "r")

    # Get the position and velocity of
    # each particle in this rank
    sim_pids = hdf[hdf_part_key]["part_pid"][part_inds]
    pos = hdf[hdf_part_key]["part_pos"][part_inds, :]
    vel = hdf[hdf_part_key]["part_vel"][part_inds, :]
    masses = hdf[hdf_part_key]["part_masses"][part_inds] * 10 ** 10
    part_types = hdf[hdf_part_key]["part_types"][part_inds]

    hdf.close()

    # Instantiate halo object
    halo = Halo(part_inds, sim_pids, pos, vel, part_types,
                masses, ini_vlcoeff, boxsize, soft,
                redshift, G, cosmo)

    return halo


@timer("Reading")
def read_multi_halo_data(tictoc, meta, part_inds, hdf_part_key):
    """

    :param tictoc:
    :param part_inds:
    :param inputpath:
    :param snapshot:
    :param hdf_part_key:
    :param ini_vlcoeff:
    :param boxsize:
    :param soft:
    :param redshift:
    :param G:
    :param cosmo:
    :return:
    """

    # Open hdf5 file
    hdf = h5py.File(meta.inputpath + meta.snap + ".hdf5", "r")

    # Get the position and velocity of
    # each particle in this rank
    sim_pids = read_subset_fromobj(hdf, hdf_part_key + "/part_pid", part_inds)
    pos = read_subset_fromobj(hdf, hdf_part_key + "/part_pos", part_inds)
    vel = read_subset_fromobj(hdf, hdf_part_key + "/part_vel", part_inds)
    masses = read_subset_fromobj(hdf, hdf_part_key + "/part_masses",
                                 part_inds)* 10 ** 10
    part_types = read_subset_fromobj(hdf, hdf_part_key + "/part_types",
                                     part_inds)

    hdf.close()

    return sim_pids, pos, vel, masses, part_types


@timer("Writing")
def write_data(tictoc, meta, nhalo, nsubhalo, results_dict, haloID_dict,
               sub_results_dict, subhaloID_dict, phase_part_haloids):
    """

    :param tictoc:
    :param meta:
    :param nhalo:
    :param nsubhalo:
    :param results_dict:
    :param haloID_dict:
    :param sub_results_dict:
    :param subhaloID_dict:
    :param phase_part_haloids:
    :return:
    """

    # Set up arrays to store halo results
    all_halo_pids = []
    begin = np.full(nhalo, -1, dtype=int)
    halo_nparts = np.full(nhalo, -1, dtype=int)
    halo_masses = np.full(nhalo, -1, dtype=float)
    halo_type_masses = np.full((nhalo, 6), -1, dtype=float)
    mean_poss = np.full((nhalo, 3), -1, dtype=float)
    mean_vels = np.full((nhalo, 3), -1, dtype=float)
    reals = np.full(nhalo, 0, dtype=bool)
    KEs = np.full(nhalo, -1, dtype=float)
    GEs = np.full(nhalo, -1, dtype=float)
    nsubhalos = np.zeros(nhalo, dtype=float)
    rms_rs = np.zeros(nhalo, dtype=float)
    rms_vrs = np.zeros(nhalo, dtype=float)
    veldisp1ds = np.zeros((nhalo, 3), dtype=float)
    veldisp3ds = np.zeros(nhalo, dtype=float)
    vmaxs = np.zeros(nhalo, dtype=float)
    hmrs = np.zeros(nhalo, dtype=float)
    hmvrs = np.zeros(nhalo, dtype=float)
    exit_vlcoeff = np.zeros(nhalo, dtype=float)

    if meta.findsubs:

        # Set up arrays to store subhalo results
        all_subhalo_pids = []
        sub_begin = np.full(nhalo, -1, dtype=int)
        subhalo_nparts = np.full(nsubhalo, -1, dtype=int)
        subhalo_masses = np.full(nsubhalo, -1, dtype=float)
        subhalo_type_masses = np.full((nsubhalo, 6), -1, dtype=float)
        sub_mean_poss = np.full((nsubhalo, 3), -1, dtype=float)
        sub_mean_vels = np.full((nsubhalo, 3), -1, dtype=float)
        sub_reals = np.full(nsubhalo, 0, dtype=bool)
        sub_KEs = np.full(nsubhalo, -1, dtype=float)
        sub_GEs = np.full(nsubhalo, -1, dtype=float)
        host_ids = np.full(nsubhalo, np.nan, dtype=int)
        sub_rms_rs = np.zeros(nsubhalo, dtype=float)
        sub_rms_vrs = np.zeros(nsubhalo, dtype=float)
        sub_veldisp1ds = np.zeros((nsubhalo, 3), dtype=float)
        sub_veldisp3ds = np.zeros(nsubhalo, dtype=float)
        sub_vmaxs = np.zeros(nsubhalo, dtype=float)
        sub_hmrs = np.zeros(nsubhalo, dtype=float)
        sub_hmvrs = np.zeros(nsubhalo, dtype=float)
        sub_exit_vlcoeff = np.zeros(nhalo, dtype=float)

    else:

        # Set up dummy subhalo results
        all_subhalo_pids = None
        sub_begin = None
        subhalo_nparts = None
        subhalo_masses = None
        subhalo_type_masses = None
        sub_mean_poss = None
        sub_mean_vels = None
        sub_reals = None
        sub_KEs = None
        sub_GEs = None
        host_ids = None
        sub_rms_rs = None
        sub_rms_vrs = None
        sub_veldisp1ds = None
        sub_veldisp3ds = None
        sub_vmaxs = None
        sub_hmrs = None
        sub_hmvrs = None
        sub_exit_vlcoeff = None

    # TODO: nPart should also be split by particle type
    # TODO: sim part ids need including

    # Create the root group
    snap = h5py.File(meta.savepath + "halos_" + str(meta.snap) + ".hdf5", "w")

    # Assign simulation attributes to the root of the z=0 snapshot
    snap.attrs[
        "snap_nPart"] = meta.npart  # number of particles in the simulation
    snap.attrs["boxsize"] = meta.boxsize  # box length along each axis
    snap.attrs["h"] = meta.h  # "little h" (hubble constant parametrisation)

    # Assign snapshot attributes
    snap.attrs["linking_length"] = meta.linkl  # host halo linking length
    snap.attrs["redshift"] = meta.z

    halo_ids = np.arange(nhalo, dtype=int)

    for res in list(results_dict.keys()):
        halo = results_dict.pop(res)
        halo_id = haloID_dict[res]
        halo_pids = halo.pids

        # Extract halo properties and store them in the output arrays
        begin[halo_id] = len(all_halo_pids)
        all_halo_pids.extend(halo_pids)
        mean_poss[halo_id, :] = halo.mean_pos
        mean_vels[halo_id, :] = halo.mean_vel
        halo_nparts[halo_id] = halo.npart
        halo_masses[halo_id] = halo.mass
        halo_type_masses[halo_id, :] = halo.ptype_mass
        reals[halo_id] = halo.real
        KEs[halo_id] = halo.KE
        GEs[halo_id] = halo.GE
        rms_rs[halo_id] = halo.rms_r
        rms_vrs[halo_id] = halo.rms_vr
        veldisp1ds[halo_id, :] = halo.veldisp1d
        veldisp3ds[halo_id] = halo.veldisp3d
        vmaxs[halo_id] = halo.vmax
        hmrs[halo_id] = halo.hmr
        hmvrs[halo_id] = halo.hmvr
        exit_vlcoeff[halo_id] = halo.vlcoeff

    # Save halo property arrays
    hdf5_write_dataset(snap, "start_index", begin)
    hdf5_write_dataset(snap, "part_ids", np.array(all_halo_pids))
    hdf5_write_dataset(snap, "halo_IDs", halo_ids)
    hdf5_write_dataset(snap, "mean_positions", mean_poss)
    hdf5_write_dataset(snap, "mean_velocities", mean_vels)
    hdf5_write_dataset(snap, "rms_spatial_radius", rms_rs)
    hdf5_write_dataset(snap, "rms_velocity_radius", rms_vrs)
    hdf5_write_dataset(snap, "1D_velocity_dispersion", veldisp1ds)
    hdf5_write_dataset(snap, "3D_velocity_dispersion", veldisp3ds)
    hdf5_write_dataset(snap, "nparts", halo_nparts)
    hdf5_write_dataset(snap, "masses", halo_masses)
    hdf5_write_dataset(snap, "part_type_masses", halo_type_masses)
    hdf5_write_dataset(snap, "real_flag", reals)
    hdf5_write_dataset(snap, "halo_kinetic_energies", KEs)
    hdf5_write_dataset(snap, "halo_gravitational_energies", GEs)
    hdf5_write_dataset(snap, "v_max", vmaxs)
    hdf5_write_dataset(snap, "half_mass_radius", hmrs)
    hdf5_write_dataset(snap, "half_mass_velocity_radius", hmvrs)
    hdf5_write_dataset(snap, "exit_vlcoeff", exit_vlcoeff)

    # Assign the full halo IDs array to the snapshot group
    hdf5_write_dataset(snap, "particle_halo_IDs", phase_part_haloids)

    # Report how many halos were found be real
    message(meta.rank,
            "Halos with unbound energies after phase space iteration:",
            halo_ids.size - halo_ids[reals].size, "of", halo_ids.size)

    if meta.findsubs:

        subhalo_ids = np.arange(nsubhalo, dtype=int)

        # Create subhalo group
        sub_root = snap.create_group("Subhalos")

        # Subhalo attributes
        sub_root.attrs["linking_length"] = meta.sub_linkl

        for res in list(sub_results_dict.keys()):
            subhalo = sub_results_dict.pop(res)
            subhalo_id = subhaloID_dict[res]
            subhalo_pids = subhalo.pids
            host = np.unique(phase_part_haloids[subhalo_pids, 0])

            assert len(host) == 1, \
                "subhalo is contained in multiple hosts, " \
                "this should not be possible"

            sub_begin[subhalo_id] = len(all_subhalo_pids)
            all_subhalo_pids.extend(subhalo_pids)
            sub_mean_poss[subhalo_id, :] = subhalo.mean_pos
            sub_mean_vels[subhalo_id, :] = subhalo.mean_vel
            subhalo_nparts[subhalo_id] = subhalo.npart
            subhalo_masses[subhalo_id] = subhalo.mass
            subhalo_type_masses[subhalo_id, :] = subhalo.ptype_mass
            sub_reals[subhalo_id] = subhalo.real
            sub_KEs[subhalo_id] = subhalo.KE
            sub_GEs[subhalo_id] = subhalo.GE
            host_ids[subhalo_id] = host
            nsubhalos[host] += 1
            sub_rms_rs[subhalo_id] = subhalo.rms_r
            sub_rms_vrs[subhalo_id] = subhalo.rms_vr
            sub_veldisp1ds[subhalo_id, :] = subhalo.veldisp1d
            sub_veldisp3ds[subhalo_id] = subhalo.veldisp3d
            sub_vmaxs[subhalo_id] = subhalo.vmax
            sub_hmrs[subhalo_id] = subhalo.hmr
            sub_hmvrs[subhalo_id] = subhalo.hmvr
            sub_exit_vlcoeff[subhalo_id] = subhalo.vlcoeff

        # Save subhalo property arrays
        hdf5_write_dataset(sub_root, "start_index", sub_begin)
        hdf5_write_dataset(sub_root, "part_ids", np.array(all_subhalo_pids))
        hdf5_write_dataset(sub_root, "subhalo_IDs", subhalo_ids)
        hdf5_write_dataset(sub_root, "host_IDs", host_ids)
        hdf5_write_dataset(sub_root, "mean_positions", sub_mean_poss)
        hdf5_write_dataset(sub_root, "mean_velocities", sub_mean_vels)
        hdf5_write_dataset(sub_root, "rms_spatial_radius", sub_rms_rs)
        hdf5_write_dataset(sub_root, "rms_velocity_radius", sub_rms_vrs)
        hdf5_write_dataset(sub_root, "1D_velocity_dispersion", sub_veldisp1ds)
        hdf5_write_dataset(sub_root, "3D_velocity_dispersion", sub_veldisp3ds)
        hdf5_write_dataset(sub_root, "nparts", subhalo_nparts)
        hdf5_write_dataset(sub_root, "masses", subhalo_masses)
        hdf5_write_dataset(sub_root, "part_type_masses", subhalo_type_masses)
        hdf5_write_dataset(sub_root, "real_flag", sub_reals)
        hdf5_write_dataset(sub_root, "halo_kinetic_energies", sub_KEs)
        hdf5_write_dataset(sub_root, "halo_gravitational_energies", sub_GEs)
        hdf5_write_dataset(sub_root, "v_max", sub_vmaxs)
        hdf5_write_dataset(sub_root, "half_mass_radius", sub_hmrs)
        hdf5_write_dataset(sub_root, "half_mass_velocity_radius", sub_hmvrs)
        hdf5_write_dataset(sub_root, "exit_vlcoeff", sub_exit_vlcoeff)

    # Write out the occupancy at the root level
    if meta.findsubs:
        hdf5_write_dataset(snap, "occupancy", nsubhalos)

    snap.close()
