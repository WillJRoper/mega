import h5py
import numpy as np
import astropy.units as u
from mpi4py import MPI

from mega.core.talking_utils import count_and_report_descs
from mega.core.talking_utils import count_and_report_progs
from mega.core.talking_utils import message, count_and_report_halos
from mega.core.timing import timer


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


def read_metadata(meta):
    """ A function to read metadata about the simulation
        from the snapshot's header.

        NOTE: Can't time reading this as the timer is not guaranteed to be
        instantiated, but reading this is essentially instantaneous.

    :param meta: The metadata object in which this data will be stored
    :return:
    """

    hdf = h5py.File(meta.inputpath + meta.snap + ".hdf5", "r")
    boxsize = hdf["Header"].attrs["BoxSize"]
    npart = hdf["Header"].attrs["NumPart_Total"]
    z = hdf["Header"].attrs["Redshift"]
    hdf.close()

    return boxsize, npart, z


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


@timer("Reading")
def read_range(tictoc, meta, key, low, high):
    """

    :param tictoc:
    :param meta:
    :param key:
    :param low:
    :param high:
    :return:
    """

    # Open hdf5 file
    hdf = h5py.File(meta.inputpath + meta.snap + ".hdf5", "r")

    # Get the positions for searching on this rank
    # NOTE: for now it"s more efficient to read all particles
    # and extract the particles we need and throw away the ones
    # we don"t, could be problematic with large datasets
    arr = hdf[key][low:high]

    hdf.close()

    return arr


@timer("Reading")
def read_subset_fromobj(tictoc, meta, hdf, key, subset, part_type=1):
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
def read_phys_subset_fromobj(tictoc, meta, hdf, key, subset,
                             units, internal_units):
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
    conv_factor = hdf[key].attrs["Conversion factor to physical CGS "
                                 "(including cosmological corrections)"]
    arr = all_arr[subset] * conv_factor * units

    return arr.to(internal_units).value


@timer("Reading")
def read_pids(tictoc, inputpath, snapshot, hdf_part_key):
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
    sim_pids = hdf[hdf_part_key]["ParticleIDs"][...]

    hdf.close()

    return sim_pids


@timer("Reading")
def read_baryonic(tictoc, meta, rank_hydro_parts):
    # Set up lists to store baryonic positions
    pos = []
    pinds = []

    # Loop over particle types
    for part_type in meta.part_types:

        if part_type == 1:
            continue

        # Get positions for this particle type
        hydro_pos = read_subset(tictoc, meta,
                                "PartType%d/Coordinates" % part_type,
                                rank_hydro_parts[part_type])

        # Define offset
        offset = len(pos)

        # Include these positions
        pos.extend(hydro_pos)

        # Include the particles indices with the offset
        pinds.extend(rank_hydro_parts[part_type] + offset)

    # Convert lists to arrays
    pos = np.array(pos, dtype=np.float64)
    pinds = np.array(pinds, dtype=int)

    return pos, pinds


@timer("Reading")
def read_multi_halo_data(tictoc, meta, part_inds):
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

    # Initialise particle data lists
    rank_nparts = part_inds.size
    sim_pids = np.zeros(rank_nparts, dtype=int)
    pos = np.zeros((rank_nparts, 3), dtype=np.float64)
    vel = np.zeros((rank_nparts, 3), dtype=np.float64)
    masses = np.zeros(rank_nparts, dtype=np.float64)
    part_types = np.zeros(rank_nparts, dtype=int)
    int_energy = np.zeros(rank_nparts, dtype=int)

    # Loop over particle types
    for part_type in meta.part_types:
        inds = part_inds - meta.part_ind_offset[part_type]
        okinds = np.where(np.logical_and(inds >= 0,
                                         inds < meta.npart[part_type]))[0]
        inds = inds[okinds]

        # Get the particle data for this halo
        sim_pids[okinds] = read_subset_fromobj(tictoc, meta, hdf,
                                               "PartType%d/ParticleIDs"
                                               % part_type,
                                               inds)
        pos[okinds, :] = read_subset_fromobj(tictoc, meta, hdf,
                                             "PartType%d/Coordinates"
                                             % part_type,
                                             inds)
        vel[okinds, :] = read_subset_fromobj(tictoc, meta, hdf,
                                             "PartType%d/Velocities"
                                             % part_type,
                                             inds)
        masses[okinds] = read_subset_fromobj(tictoc, meta, hdf,
                                             "PartType%d/Masses" % part_type,
                                             inds)
        part_types[okinds] = np.full(len(inds), part_type, dtype=int)

        if part_type == 0:
            int_energy[okinds] = read_phys_subset_fromobj(
                tictoc, meta, hdf, "PartType%d/InternalEnergies" % part_type,
                inds, units=(u.cm**2 / u.s**2), internal_units=meta.U_EperM
            )
        else:
            int_energy[okinds] = np.zeros(len(inds), dtype=float)

    hdf.close()

    return sim_pids, pos, vel, masses, part_types, int_energy


@timer("Reading")
def read_prog_data(tictoc, meta, density_rank, comm):
    if not meta.isfirst:

        # Open hdf5 file
        hdf = h5py.File(meta.halopath + 'halos_' + meta.prog_snap + '.hdf5',
                        'r')

        if density_rank == 0:
            root = hdf
        else:
            root = hdf["Subhalos"]

        # How many particles do we have?
        prog_npart = root["particle_halo_IDs"].size

        # Get the initial rank particle bins
        prog_rank_partbins = np.linspace(0, prog_npart,
                                         meta.nranks + 1,
                                         dtype=int)

        # Lets get our slice
        myslice = (prog_rank_partbins[meta.rank],
                   prog_rank_partbins[meta.rank + 1])

        # Get minimum and maximum pid in my slice
        rank_pids = root["all_sim_part_ids"][myslice[0]: myslice[1]]

        min_pid = np.zeros((1))
        max_pid = np.zeros((1))
        min_pid[0] = np.min(rank_pids)
        max_pid[0] = np.max(rank_pids)

        # Lets get the global minimum and maximum
        comm.Allreduce(MPI.IN_PLACE, min_pid, op=MPI.MIN)
        comm.Allreduce(MPI.IN_PLACE, max_pid, op=MPI.MAX)

        # Get particle rank bins
        prog_rank_pidbins = np.linspace(min_pid[0], max_pid[0] + 1,
                                        meta.nranks + 1,
                                        dtype=int)

        # Extract our particle's halo ids
        rank_part_progids = root["particle_halo_IDs"][myslice[0]: myslice[1]]

        # Lets get the progenitor halo data
        # (this is nhalo in length so give every rank a copy)
        proghalo_nparts = root["nparts"][...]
        prog_reals = root["real_flag"][...]

        hdf.close()  # close the root group
    else:
        prog_npart = None
        prog_rank_partbins = None
        rank_part_progids = None
        proghalo_nparts = None
        prog_reals = None
        rank_pids = None
        prog_rank_pidbins = None

    return (prog_npart, proghalo_nparts, prog_rank_partbins,
            rank_part_progids, prog_reals, rank_pids, prog_rank_pidbins)


@timer("Reading")
def read_current_data(tictoc, meta, density_rank, comm):
    # How many halos and particles are we dealing with in the current snapshot?
    hdf = h5py.File(meta.halopath + 'halos_' + meta.snap + '.hdf5', 'r')
    if density_rank == 0:
        root = hdf
        nhalo = hdf.attrs['nHalo']
    else:
        root = hdf["Subhalos"]
        nhalo = hdf.attrs['nSubhalo']

    # How many particles do we have?
    npart = root["particle_halo_IDs"].size

    # Lets get the halo particle ids we have on this rank
    rank_partbins = np.linspace(0, npart,
                                meta.nranks + 1,
                                dtype=int)

    # Get minimum and maximum pid in my slice
    rank_pids = root["all_sim_part_ids"][rank_partbins[meta.rank]:
                                         rank_partbins[meta.rank + 1]]
    min_pid = np.zeros((1))
    max_pid = np.zeros((1))
    min_pid[0] = np.min(rank_pids)
    max_pid[0] = np.max(rank_pids)

    # Lets get the global minimum and maximum
    comm.Allreduce(MPI.IN_PLACE, min_pid, op=MPI.MIN)
    comm.Allreduce(MPI.IN_PLACE, max_pid, op=MPI.MAX)

    # Get particle rank bins
    rank_pidbins = np.linspace(min_pid[0], max_pid[0] + 1,
                               meta.nranks + 1,
                               dtype=int)

    # Lets get our slice
    myslice = (rank_partbins[meta.rank],
               rank_partbins[meta.rank + 1])

    # Extract our particle's halo ids
    rank_part_haloids = root["particle_halo_IDs"][myslice[0]: myslice[1]]

    # Lets get the halo data
    # (this is nhalo in length so give every rank a copy)
    halo_nparts = root["nparts"][...]
    reals = root["real_flag"][...]

    hdf.close()

    return nhalo, rank_partbins, reals, rank_part_haloids, halo_nparts, rank_pidbins


@timer("Reading")
def read_desc_data(tictoc, meta, density_rank, comm):
    if meta.desc_snap is not None:

        # How many particles are we dealing with in the descendent snapshot?
        hdf = h5py.File(meta.halopath + 'halos_'
                        + meta.desc_snap + '.hdf5', 'r')

        if density_rank == 0:
            root = hdf
        else:
            root = hdf["Subhalos"]

        # How many particles do we have?
        desc_npart = root["particle_halo_IDs"].size

        # Get the initial rank particle bins
        desc_rank_partbins = np.linspace(0, desc_npart,
                                         meta.nranks + 1,
                                         dtype=int)

        # Get minimum and maximum pid in my slice
        rank_pids = root["all_sim_part_ids"][desc_rank_partbins[meta.rank]:
                                             desc_rank_partbins[meta.rank + 1]]
        min_pid = np.zeros((1))
        max_pid = np.zeros((1))
        min_pid[0] = np.min(rank_pids)
        max_pid[0] = np.max(rank_pids)

        # Lets get the global minimum and maximum
        comm.Allreduce(MPI.IN_PLACE, min_pid, op=MPI.MIN)
        comm.Allreduce(MPI.IN_PLACE, max_pid, op=MPI.MAX)

        # Get particle rank bins
        desc_rank_pidbins = np.linspace(min_pid[0], max_pid[0] + 1,
                                        meta.nranks + 1,
                                        dtype=int)

        # Lets get our slice
        myslice = (desc_rank_partbins[meta.rank],
                   desc_rank_partbins[meta.rank + 1])

        # Extract our particle's halo ids
        rank_part_descids = root["particle_halo_IDs"][myslice[0]: myslice[1]]

        # Lets get the descendant halo data
        # (this is nhalo in length so give every rank a copy)
        deschalo_nparts = root["nparts"][...]

        hdf.close()
    else:
        desc_npart = None
        desc_rank_partbins = None
        rank_part_descids = None
        deschalo_nparts = None
        rank_pids = None
        desc_rank_pidbins = None

    return (desc_npart, deschalo_nparts, desc_rank_partbins, rank_part_descids,
            rank_pids, desc_rank_pidbins)


@timer("Writing")
def write_data(tictoc, meta, nhalo, nsubhalo, results_dict,
               sub_results_dict, sim_pids=None, basename_mod="",
               extra_data=None):
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

    # Initialise particle halo id arrays
    phase_part_haloids = np.full(np.sum(meta.npart), -2, dtype=np.int32)
    phase_part_subhaloids = np.full(np.sum(meta.npart), -2, dtype=np.int32)

    # Set up arrays to store halo results
    all_halo_simpids = []
    all_halo_pids = []
    all_halo_simpids_type = {i: [] for i in meta.part_types}
    all_halo_pids_type = {i: [] for i in meta.part_types}
    start_index = np.zeros(nhalo, dtype=int)
    stride = np.zeros(nhalo, dtype=int)
    start_index_type = np.zeros((nhalo, len(meta.npart)), dtype=int)
    stride_type = np.zeros((nhalo, len(meta.npart)), dtype=int)
    halo_nparts = np.zeros((nhalo, len(meta.npart)), dtype=int)
    halo_masses = np.full(nhalo, -1, dtype=float)
    halo_type_masses = np.full((nhalo, len(meta.npart)), -1, dtype=float)
    mean_poss = np.full((nhalo, 3), -1, dtype=float)
    mean_vels = np.full((nhalo, 3), -1, dtype=float)
    reals = np.full(nhalo, 0, dtype=bool)
    KEs = np.full(nhalo, -1, dtype=float)
    GEs = np.full(nhalo, -1, dtype=float)
    nsubhalos = np.zeros(nhalo, dtype=float)

    # TODO: properties should be split by particle type and
    #  also introduce apertures
    rms_rs = np.zeros(nhalo, dtype=float)
    rms_vrs = np.zeros(nhalo, dtype=float)
    veldisp1ds = np.zeros((nhalo, 3), dtype=float)
    veldisp3ds = np.zeros(nhalo, dtype=float)
    vmaxs = np.zeros(nhalo, dtype=float)
    hmrs = np.zeros(nhalo, dtype=float)
    hmvrs = np.zeros(nhalo, dtype=float)
    exit_vlcoeff = np.zeros(nhalo, dtype=float)
    if meta.with_hydro:
        int_nrg = np.zeros(nhalo, dtype=float)
    else:
        int_nrg = None

    if meta.findsubs:

        # Set up arrays to store subhalo results
        all_subhalo_simpids = []
        all_subhalo_pids = []
        all_subhalo_simpids_type = {i: [] for i in meta.part_types}
        all_subhalo_pids_type = {i: [] for i in meta.part_types}
        sub_start_index = np.zeros(nsubhalo, dtype=int)
        sub_stride = np.zeros(nsubhalo, dtype=int)
        sub_start_index_type = np.zeros((nsubhalo, len(meta.npart)), dtype=int)
        sub_stride_type = np.zeros((nsubhalo, len(meta.npart)), dtype=int)
        subhalo_nparts = np.zeros((nsubhalo, len(meta.npart)), dtype=int)
        subhalo_masses = np.full(nsubhalo, -1, dtype=float)
        subhalo_type_masses = np.full((nsubhalo, len(meta.npart)), -1,
                                      dtype=float)
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
        if meta.with_hydro:
            sub_int_nrg = np.zeros(nhalo, dtype=float)
        else:
            sub_int_nrg = None

    else:

        # Set up dummy subhalo results
        all_subhalo_simpids = None
        all_subhalo_pids = None
        all_subhalo_simpids_type = None
        all_subhalo_pids_type = None
        sub_start_index = None
        sub_stride = None
        sub_start_index_type = None
        sub_stride_type = None
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
        sub_int_nrg = None

    # Create the root group
    snap = h5py.File(meta.savepath + meta.halo_basename + basename_mod + "_"
                     + str(meta.snap) + ".hdf5", "w")

    # Assign simulation attributes to the root of the z=0 snapshot
    snap.attrs[
        "snap_nPart"] = meta.npart  # number of particles in the simulation
    snap.attrs["boxsize"] = meta.boxsize  # box length along each axis
    snap.attrs["h"] = meta.h  # "little h" (hubble constant parametrisation)

    # Assign snapshot attributes
    snap.attrs["linking_length"] = meta.linkl  # host halo linking length
    snap.attrs["redshift"] = meta.z
    snap.attrs["nHalo"] = nhalo
    snap.attrs["nSubhalo"] = nsubhalo
    snap.attrs["part_type_offset"] = meta.part_ind_offset

    # Create halo id array
    halo_ids = np.arange(nhalo, dtype=int)

    # Loop over results
    ihalo = 0
    for res in list(results_dict.keys()):
        halo = results_dict.pop(res)

        # Extract halo properties and store them in the output arrays
        start_index[ihalo] = len(all_halo_simpids)
        stride[ihalo] = len(halo.pids)
        all_halo_simpids.extend(halo.sim_pids)
        all_halo_pids.extend(halo.pids)
        mean_poss[ihalo, :] = halo.mean_pos
        mean_vels[ihalo, :] = halo.mean_vel
        halo_nparts[ihalo, :] = halo.npart_types
        halo_masses[ihalo] = halo.mass
        halo_type_masses[ihalo, :] = halo.ptype_mass
        reals[ihalo] = halo.real
        KEs[ihalo] = halo.KE
        GEs[ihalo] = halo.GE
        rms_rs[ihalo] = halo.rms_r
        rms_vrs[ihalo] = halo.rms_vr
        veldisp1ds[ihalo, :] = halo.veldisp1d
        veldisp3ds[ihalo] = halo.veldisp3d
        vmaxs[ihalo] = halo.vmax
        hmrs[ihalo] = halo.hmr
        hmvrs[ihalo] = halo.hmvr
        exit_vlcoeff[ihalo] = halo.vlcoeff

        # Loop over particle types and store data split by part type
        for part_type in meta.part_types:
            okinds = np.where(halo.types == part_type)[0]
            all_halo_simpids_type[part_type].extend(halo.sim_pids[okinds])
            all_halo_pids_type[part_type].extend(halo.pids[okinds])
            start_index_type[ihalo,
                             part_type] = len(all_halo_simpids_type[part_type])
            stride_type[ihalo, part_type] = len(halo.pids[okinds])

        # Store baryonic results
        if meta.with_hydro:
            int_nrg[ihalo] = halo.therm_nrg

        # Increment halo counter
        ihalo += 1

    # Convert lists to arrays
    all_halo_simpids = np.array(all_halo_simpids)
    all_halo_pids = np.array(all_halo_pids)
    for part_type in meta.part_types:
        all_halo_simpids_type[part_type] = np.array(
            all_halo_simpids_type[part_type]
        )
        all_halo_pids_type[part_type] = np.array(
            all_halo_pids_type[part_type]
        )

    # Save halo property arrays
    hdf5_write_dataset(snap, "start_index", start_index)
    hdf5_write_dataset(snap, "stride", stride)
    hdf5_write_dataset(snap, "sim_part_ids", all_halo_simpids)
    hdf5_write_dataset(snap, "part_ids", all_halo_pids)
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

    # Loop over particle types and write out specific
    if meta.with_hydro:
        for part_type in meta.part_types:
            part_root = snap.create_group("PartType%d" % part_type)
            hdf5_write_dataset(part_root, "start_index",
                               start_index_type[part_type])
            hdf5_write_dataset(part_root, "stride", stride_type[part_type])
            hdf5_write_dataset(part_root, "sim_part_ids",
                               all_halo_simpids_type[part_type])
            hdf5_write_dataset(part_root, "part_ids",
                               all_halo_pids_type[part_type])

    # Write out baryonic results
    if meta.with_hydro:
        hdf5_write_dataset(snap, "halo_thermal_energy", int_nrg)

    # Write out any extra data
    if extra_data is not None:
        for key, val in extra_data.items():
            hdf5_write_dataset(snap, key, val)

    # Do we have an outside input for the sim pids?
    if sim_pids is None:

        # Initialise array to store all particle IDs
        sim_pids = np.zeros(np.sum(meta.npart), dtype=int)

        # Read particle IDs to store combined particle ids array
        for part_type in meta.part_types:
            offset = meta.part_ind_offset[part_type]
            sim_part_ids = read_pids(tictoc, meta.inputpath, meta.snap,
                                     "PartType%d" % part_type)
            sim_pids[offset: offset + meta.npart[part_type]] = sim_part_ids

    # Write out the sorting indices
    hdf5_write_dataset(snap, "all_sim_part_ids", sim_pids)

    # Now we can set the correct halo_ids
    for (halo_id, b), l in zip(enumerate(start_index), stride):
        parts = all_halo_pids[b: b + l]
        phase_part_haloids[parts] = halo_id

    count_and_report_halos(phase_part_haloids, meta,
                           halo_type="Phase Space Host Halos")

    # Assign the full halo IDs array to the snapshot group
    hdf5_write_dataset(snap, "particle_halo_IDs", phase_part_haloids)

    # Report how many halos were found be real
    message(meta.rank,
            "Halos with unbound energies in final sample:",
            halo_ids.size - halo_ids[reals].size, "of", halo_ids.size)

    if meta.findsubs:

        # Create array of subhalo IDs
        subhalo_ids = np.arange(nsubhalo, dtype=int)

        # Create subhalo group
        sub_root = snap.create_group("Subhalos")

        # Subhalo attributes
        sub_root.attrs["linking_length"] = meta.sub_linkl

        # Loop over subhalo results
        isubhalo = 0
        for res in list(sub_results_dict.keys()):
            subhalo = sub_results_dict.pop(res)
            host = np.unique(phase_part_haloids[subhalo.pids])

            assert len(host) == 1, \
                "subhalo is contained in multiple hosts, " \
                "this should not be possible"

            sub_start_index[isubhalo] = len(all_subhalo_simpids)
            sub_stride[isubhalo] = len(subhalo.pids)
            all_subhalo_simpids.extend(subhalo.sim_pids)
            all_subhalo_pids.extend(subhalo.pids)
            sub_mean_poss[isubhalo, :] = subhalo.mean_pos
            sub_mean_vels[isubhalo, :] = subhalo.mean_vel
            subhalo_nparts[isubhalo, :] = subhalo.npart_types
            subhalo_masses[isubhalo] = subhalo.mass
            subhalo_type_masses[isubhalo, :] = subhalo.ptype_mass
            sub_reals[isubhalo] = subhalo.real
            sub_KEs[isubhalo] = subhalo.KE
            sub_GEs[isubhalo] = subhalo.GE
            host_ids[isubhalo] = host
            nsubhalos[host] += 1
            sub_rms_rs[isubhalo] = subhalo.rms_r
            sub_rms_vrs[isubhalo] = subhalo.rms_vr
            sub_veldisp1ds[isubhalo, :] = subhalo.veldisp1d
            sub_veldisp3ds[isubhalo] = subhalo.veldisp3d
            sub_vmaxs[isubhalo] = subhalo.vmax
            sub_hmrs[isubhalo] = subhalo.hmr
            sub_hmvrs[isubhalo] = subhalo.hmvr
            sub_exit_vlcoeff[isubhalo] = subhalo.vlcoeff

            # Loop over particle types and store data split by part type
            for part_type in meta.part_types:
                okinds = np.where(subhalo.types == part_type)[0]
                all_subhalo_simpids_type[part_type].extend(
                    subhalo.sim_pids[okinds])
                all_subhalo_pids_type[part_type].extend(
                    subhalo.pids[okinds])
                sub_start_index_type[isubhalo, part_type] = len(
                    all_subhalo_simpids_type[part_type])
                sub_stride_type[isubhalo, part_type] = len(
                    subhalo.pids[okinds])

            # Store baryonic results
            if meta.with_hydro:
                sub_int_nrg[isubhalo] = subhalo.therm_nrg

        # Convert lists to arrays
        all_subhalo_simpids = np.array(all_subhalo_simpids)
        all_subhalo_pids = np.array(all_subhalo_pids)
        for part_type in meta.part_types:
            all_subhalo_simpids_type[part_type] = np.array(
                all_subhalo_simpids_type[part_type]
            )
            all_subhalo_pids_type[part_type] = np.array(
                all_subhalo_pids_type[part_type]
            )

        # Save subhalo property arrays
        hdf5_write_dataset(sub_root, "start_index", sub_start_index)
        hdf5_write_dataset(sub_root, "stride", sub_stride)
        hdf5_write_dataset(sub_root, "sim_part_ids", all_subhalo_simpids)
        hdf5_write_dataset(sub_root, "part_ids", all_subhalo_pids)
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

        # Loop over particle types and write out specific
        if meta.with_hydro:
            for part_type in meta.part_types:
                part_root = sub_root.create_group("PartType%d" % part_type)
                hdf5_write_dataset(part_root, "start_index",
                                   sub_start_index_type[part_type])
                hdf5_write_dataset(part_root, "stride",
                                   sub_stride_type[part_type])
                hdf5_write_dataset(part_root, "sim_part_ids",
                                   all_subhalo_simpids_type[part_type])
                hdf5_write_dataset(part_root, "part_ids",
                                   all_subhalo_pids_type[part_type])

        # Write out baryonic results
        if meta.with_hydro:
            hdf5_write_dataset(sub_root, "halo_thermal_energy", sub_int_nrg)

        # Now we can set the correct halo_ids
        for (subhalo_id, b), l in zip(enumerate(sub_start_index), sub_stride):
            parts = all_subhalo_pids[b: b + l]
            phase_part_subhaloids[parts] = subhalo_id

        count_and_report_halos(phase_part_subhaloids, meta,
                               halo_type="Phase Space Subhalos")

        # Assign the full halo IDs array to the snapshot group
        hdf5_write_dataset(sub_root, "particle_halo_IDs",
                           phase_part_subhaloids)

        # Write out the occupancy at the root level
        hdf5_write_dataset(snap, "occupancy", nsubhalos)

    snap.close()


@timer("Writing")
def write_dgraph_data(tictoc, meta, all_results, density_rank, reals):
    # Lets combine the list of results from everyone into a single dictionary
    results = {}
    for d in all_results:
        results.update(d)

    # Initialise counter for halos removed due to not being temporally real
    notreals = 0

    # Initialise counter for transient halos (halos with no progs or descs)
    transients = 0

    # Set up arrays to store host results
    nhalo = len(results)
    index_haloids = np.array(list(results.keys()))
    halo_nparts = np.full(nhalo, -2, dtype=int)
    nprogs = np.zeros(nhalo, dtype=int)
    ndescs = np.zeros(nhalo, dtype=int)
    prog_start_index = np.full(nhalo, -2, dtype=int)
    desc_start_index = np.full(nhalo, -2, dtype=int)

    progs = []
    descs = []
    prog_mass_conts = []
    desc_mass_conts = []
    prog_nparts = []
    desc_nparts = []

    if meta.desc_snap is not None:

        # Load the descendant snapshot
        hdf = h5py.File(meta.halopath + 'halos_'
                        + meta.desc_snap + '.hdf5', 'r')

        # Get the reality flag array
        if density_rank == 0:
            desc_reals = hdf['real_flag'][...]
        else:
            desc_reals = hdf['Subhalos']['real_flag'][...]

        hdf.close()
    else:
        desc_reals = np.array([False])

    if meta.prog_snap is not None:

        # Load the progenitor snapshot
        hdf = h5py.File(meta.halopath + 'halos_'
                        + meta.prog_snap + '.hdf5', 'r')

        # Get progenitor snapshot data
        if density_rank == 0:
            prog_reals = hdf['real_flag'][...]
        else:
            prog_reals = hdf['Subhalos']['real_flag'][...]

        hdf.close()

    else:
        prog_reals = np.array([False])

    while len(results) > 0:

        # Extract a halo to store
        ihalo, halo = results.popitem()

        assert (len(set(halo.prog_haloids)) == len(halo.prog_haloids)), \
            "Not all progenitors are unique"
        assert (len(set(halo.desc_haloids)) == len(halo.desc_haloids)), \
            "Not all progenitors are unique"

        # Extract this halo's data
        nprog = halo.nprog
        prog_haloids = halo.prog_haloids
        prog_npart = halo.prog_npart
        prog_npart_cont = halo.prog_npart_cont
        ndesc = halo.ndesc
        desc_haloids = halo.desc_haloids
        desc_npart = halo.desc_npart
        desc_npart_cont = halo.desc_npart_cont
        preals = halo.prog_reals
        npart = halo.npart

        # If this halo has no real progenitors and is less than 20 particle
        # it is by definition not a halo
        if nprog == 0 and npart < 20:
            reals[ihalo] = False
            notreals += 1
            continue

        # If the halo has neither descendants or progenitors we do not
        # need to store it
        elif nprog < 1 and ndesc < 1:
            reals[ihalo] = False
            transients += 1
            continue

        # If any progenitor is real then this halo is real
        if True in preals:
            reals[ihalo] = True

        if reals[ihalo]:

            # If this halo is real then it's descendants are real
            if meta.desc_snap is not None:
                desc_reals[desc_haloids] = True

            # Write out the data produced
            nprogs[ihalo] = nprog  # number of progenitors
            ndescs[ihalo] = ndesc  # number of descendants
            halo_nparts[ihalo] = npart  # mass of the halo

            # If we have progenitors store them and the pointers
            if nprog > 0:
                prog_start_index[ihalo] = len(progs)
                progs.extend(prog_haloids)
                prog_mass_conts.extend(prog_npart_cont)
                prog_nparts.extend(prog_npart)
            else:  # else put null pointer
                prog_start_index[ihalo] = 2 ** 30

            # If we have descendants store them and the pointers
            if ndesc > 0:
                desc_start_index[ihalo] = len(descs)
                descs.extend(desc_haloids)
                desc_mass_conts.extend(desc_npart_cont)
                desc_nparts.extend(desc_npart)
            else:  # else put null pointer
                desc_start_index[ihalo] = 2 ** 30

    progs = np.array(progs)
    descs = np.array(descs)
    prog_mass_conts = np.array(prog_mass_conts)
    desc_mass_conts = np.array(desc_mass_conts)
    prog_nparts = np.array(prog_nparts)
    desc_nparts = np.array(desc_nparts)

    # Create file to store this snapshots graph results
    if density_rank == 0:
        hdf = h5py.File(meta.dgraphpath + 'Mgraph_' + meta.snap + '.hdf5', 'w')
    else:
        hdf = h5py.File(meta.dgraphpath + 'SubMgraph_' + meta.snap + '.hdf5',
                        'w')

    hdf5_write_dataset(hdf, 'halo_IDs', index_haloids)
    hdf5_write_dataset(hdf, 'nProgs', nprogs)
    hdf5_write_dataset(hdf, 'nDescs', ndescs)
    hdf5_write_dataset(hdf, 'nparts', halo_nparts)
    hdf5_write_dataset(hdf, 'prog_start_index', prog_start_index)
    hdf5_write_dataset(hdf, 'desc_start_index', desc_start_index)
    hdf5_write_dataset(hdf, 'Prog_haloIDs', progs)
    hdf5_write_dataset(hdf, 'Desc_haloIDs', descs)
    hdf5_write_dataset(hdf, 'Prog_nPart_Contribution', prog_mass_conts)
    hdf5_write_dataset(hdf, 'Desc_nPart_Contribution', desc_mass_conts)
    hdf5_write_dataset(hdf, 'Prog_nPart', prog_nparts)
    hdf5_write_dataset(hdf, 'Desc_nPart', desc_nparts)
    hdf5_write_dataset(hdf, 'prog_real_flag', prog_reals)
    hdf5_write_dataset(hdf, 'real_flag', reals)
    hdf5_write_dataset(hdf, 'desc_real_flag', desc_reals)

    hdf.close()

    message(meta.rank, "Found %d halos to not be real out of %d" % (notreals,
                                                                    nhalo))
    message(meta.rank, "Found %d transient halos out of %d" % (transients,
                                                               nhalo))

    if density_rank == 0:
        count_and_report_progs(nprogs, meta, halo_type="Host")
        count_and_report_descs(ndescs, meta, halo_type="Host")
    else:
        count_and_report_progs(nprogs, meta, halo_type="Subhalo")
        count_and_report_descs(ndescs, meta, halo_type="Subhalo")

    return reals, desc_reals


@timer("Writing")
def clean_real_flags(tictoc, meta, density_rank, reals, snap):
    # Load the descendant snapshot
    hdf = h5py.File(meta.halopath + 'halos_' + snap + '.hdf5', 'r+')

    # Set the reality flag in the halo catalog
    if density_rank == 0:
        message(meta.rank, "Overwriting host real flags: %s" % snap)
        del hdf['real_flag']
        hdf.create_dataset('real_flag', shape=reals.shape, dtype=bool,
                           data=reals,
                           compression='gzip')
    else:
        message(meta.rank, "Overwriting subhalo real flags: %s" % snap)
        sub_current = hdf['Subhalos']
        del sub_current['real_flag']
        sub_current.create_dataset('real_flag', shape=reals.shape, dtype=bool,
                                   data=reals,
                                   compression='gzip')

    hdf.close()