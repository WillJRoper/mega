import numpy as np
import h5py

from timing import timer


def hdf5_write_dataset(grp, key, data, compression="gzip"):
    grp.create_dataset(key, shape=data.shape, dtype=data.dtype, data=data,
                       compression=compression)


@timer("Writing")
def write_data(tictoc, meta, nhalo, nsubhalo, results_dict, haloID_dict,
               sub_results_dict, subhaloID_dict, phase_part_haloids):

    # Set up arrays to store subhalo results
    halo_nparts = np.full(nhalo, -1, dtype=int)
    halo_masses = np.full(nhalo, -1, dtype=float)
    halo_type_masses = np.full((nhalo, 6), -1, dtype=float)
    mean_poss = np.full((nhalo, 3), -1, dtype=float)
    mean_vels = np.full((nhalo, 3), -1, dtype=float)
    reals = np.full(nhalo, 0, dtype=bool)
    halo_energies = np.full(nhalo, -1, dtype=float)
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

        # Set up arrays to store host results
        subhalo_nparts = np.full(nsubhalo, -1, dtype=int)
        subhalo_masses = np.full(nsubhalo, -1, dtype=float)
        subhalo_type_masses = np.full((nsubhalo, 6), -1, dtype=float)
        sub_mean_poss = np.full((nsubhalo, 3), -1, dtype=float)
        sub_mean_vels = np.full((nsubhalo, 3), -1, dtype=float)
        sub_reals = np.full(nsubhalo, 0, dtype=bool)
        subhalo_energies = np.full(nsubhalo, -1, dtype=float)
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
        subhalo_nparts = None
        subhalo_masses = None
        subhalo_type_masses = None
        sub_mean_poss = None
        sub_mean_vels = None
        sub_reals = None
        subhalo_energies = None
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
    # TODO: output should use halo objects

    # Create the root group
    snap = h5py.File(meta.savepath + 'halos_' + str(meta.snap) + '.hdf5', 'w')

    # Assign simulation attributes to the root of the z=0 snapshot
    snap.attrs[
        'snap_nPart'] = meta.npart  # number of particles in the simulation
    snap.attrs['boxsize'] = meta.boxsize  # box length along each axis
    snap.attrs['h'] = meta.h  # 'little h' (hubble constant parametrisation)

    # Assign snapshot attributes
    snap.attrs['linking_length'] = meta.linkl  # host halo linking length
    snap.attrs['redshift'] = meta.z

    halo_ids = np.arange(nhalo, dtype=int)

    for res in list(results_dict.keys()):
        halo_res = results_dict.pop(res)
        halo_id = haloID_dict[res]
        halo_pids = halo_res['pids']

        mean_poss[halo_id, :] = halo_res['mean_halo_pos']
        mean_vels[halo_id, :] = halo_res['mean_halo_vel']
        halo_nparts[halo_id] = halo_res['npart']
        halo_masses[halo_id] = halo_res["halo_mass"]
        halo_type_masses[halo_id, :] = halo_res["halo_ptype_mass"]
        reals[halo_id] = halo_res['real']
        halo_energies[halo_id] = halo_res['halo_energy']
        KEs[halo_id] = halo_res['KE']
        GEs[halo_id] = halo_res['GE']
        rms_rs[halo_id] = halo_res["rms_r"]
        rms_vrs[halo_id] = halo_res["rms_vr"]
        veldisp1ds[halo_id, :] = halo_res["veldisp1d"]
        veldisp3ds[halo_id] = halo_res["veldisp3d"]
        vmaxs[halo_id] = halo_res["vmax"]
        hmrs[halo_id] = halo_res["hmr"]
        hmvrs[halo_id] = halo_res["hmvr"]
        exit_vlcoeff[halo_id] = halo_res["vlcoeff"]

        # Create datasets in the current halo's group in the HDF5 file
        halo = snap.create_group(str(halo_id))  # create halo group
        halo.create_dataset('Halo_Part_IDs', shape=halo_pids.shape,
                            dtype=int,
                            data=halo_pids)  # halo particle ids

    # Save halo property arrays
    snap.create_dataset('halo_IDs',
                        shape=halo_ids.shape,
                        dtype=int,
                        data=halo_ids,
                        compression='gzip')
    snap.create_dataset('mean_positions',
                        shape=mean_poss.shape,
                        dtype=float,
                        data=mean_poss,
                        compression='gzip')
    snap.create_dataset('mean_velocities',
                        shape=mean_vels.shape,
                        dtype=float,
                        data=mean_vels,
                        compression='gzip')
    snap.create_dataset('rms_spatial_radius',
                        shape=rms_rs.shape,
                        dtype=rms_rs.dtype,
                        data=rms_rs,
                        compression='gzip')
    snap.create_dataset('rms_velocity_radius',
                        shape=rms_vrs.shape,
                        dtype=rms_vrs.dtype,
                        data=rms_vrs,
                        compression='gzip')
    snap.create_dataset('1D_velocity_dispersion',
                        shape=veldisp1ds.shape,
                        dtype=veldisp1ds.dtype,
                        data=veldisp1ds,
                        compression='gzip')
    snap.create_dataset('3D_velocity_dispersion',
                        shape=veldisp3ds.shape,
                        dtype=veldisp3ds.dtype,
                        data=veldisp3ds,
                        compression='gzip')
    snap.create_dataset('nparts',
                        shape=halo_nparts.shape,
                        dtype=int,
                        data=halo_nparts,
                        compression='gzip')
    snap.create_dataset('total_masses',
                        shape=halo_masses.shape,
                        dtype=float,
                        data=halo_masses,
                        compression='gzip')
    snap.create_dataset('masses',
                        shape=halo_type_masses.shape,
                        dtype=float,
                        data=halo_type_masses,
                        compression='gzip')
    snap.create_dataset('real_flag',
                        shape=reals.shape,
                        dtype=bool,
                        data=reals,
                        compression='gzip')
    snap.create_dataset('halo_total_energies',
                        shape=halo_energies.shape,
                        dtype=float,
                        data=halo_energies,
                        compression='gzip')
    snap.create_dataset('halo_kinetic_energies',
                        shape=KEs.shape,
                        dtype=float,
                        data=KEs,
                        compression='gzip')
    snap.create_dataset('halo_gravitational_energies',
                        shape=GEs.shape,
                        dtype=float,
                        data=GEs,
                        compression='gzip')
    snap.create_dataset('v_max',
                        shape=vmaxs.shape,
                        dtype=vmaxs.dtype,
                        data=vmaxs,
                        compression='gzip')
    snap.create_dataset('half_mass_radius',
                        shape=hmrs.shape,
                        dtype=hmrs.dtype,
                        data=hmrs,
                        compression='gzip')
    snap.create_dataset('half_mass_velocity_radius',
                        shape=hmvrs.shape,
                        dtype=hmvrs.dtype,
                        data=hmvrs,
                        compression='gzip')
    snap.create_dataset('exit_vlcoeff',
                        shape=exit_vlcoeff.shape,
                        dtype=exit_vlcoeff.dtype,
                        data=exit_vlcoeff,
                        compression='gzip')

    # Assign the full halo IDs array to the snapshot group
    snap.create_dataset('particle_halo_IDs',
                        shape=phase_part_haloids.shape,
                        dtype=int,
                        data=phase_part_haloids,
                        compression='gzip')

    # Get how many halos were found be real
    print("Halos with unbound energies after phase space iteration:",
          halo_ids.size - halo_ids[reals].size, "of", halo_ids.size)

    if meta.findsubs:

        subhalo_ids = np.arange(nsubhalo, dtype=int)

        # Create subhalo group
        sub_root = snap.create_group('Subhalos')

        # Subhalo attributes
        sub_root.attrs['linking_length'] = meta.sub_linkl

        for res in list(sub_results_dict.keys()):
            subhalo_res = sub_results_dict.pop(res)
            subhalo_id = subhaloID_dict[res]
            subhalo_pids = subhalo_res['pids']
            host = np.unique(phase_part_haloids[subhalo_pids, 0])

            assert len(host) == 1, \
                "subhalo is contained in multiple hosts, " \
                "this should not be possible"

            sub_mean_poss[subhalo_id, :] = subhalo_res['mean_halo_pos']
            sub_mean_vels[subhalo_id, :] = subhalo_res['mean_halo_vel']
            subhalo_nparts[subhalo_id] = subhalo_res['npart']
            subhalo_masses[subhalo_id] = subhalo_res["halo_mass"]
            subhalo_type_masses[subhalo_id, :] = subhalo_res[
                "halo_ptype_mass"]
            sub_reals[subhalo_id] = subhalo_res['real']
            subhalo_energies[subhalo_id] = subhalo_res['halo_energy']
            sub_KEs[subhalo_id] = subhalo_res['KE']
            sub_GEs[subhalo_id] = subhalo_res['GE']
            host_ids[subhalo_id] = host
            nsubhalos[host] += 1
            sub_rms_rs[subhalo_id] = subhalo_res["rms_r"]
            sub_rms_vrs[subhalo_id] = subhalo_res["rms_vr"]
            sub_veldisp1ds[subhalo_id, :] = subhalo_res["veldisp1d"]
            sub_veldisp3ds[subhalo_id] = subhalo_res["veldisp3d"]
            sub_vmaxs[subhalo_id] = subhalo_res["vmax"]
            sub_hmrs[subhalo_id] = subhalo_res["hmr"]
            sub_hmvrs[subhalo_id] = subhalo_res["hmvr"]
            sub_exit_vlcoeff[subhalo_id] = subhalo_res["vlcoeff"]

            # Create subhalo group
            subhalo = sub_root.create_group(str(subhalo_id))
            subhalo.create_dataset('Halo_Part_IDs',
                                   shape=subhalo_pids.shape,
                                   dtype=int,
                                   data=subhalo_pids)

        # Save halo property arrays
        sub_root.create_dataset('subhalo_IDs',
                                shape=subhalo_ids.shape,
                                dtype=int,
                                data=subhalo_ids,
                                compression='gzip')
        sub_root.create_dataset('host_IDs',
                                shape=host_ids.shape,
                                dtype=int, data=host_ids,
                                compression='gzip')
        sub_root.create_dataset('mean_positions',
                                shape=sub_mean_poss.shape,
                                dtype=float,
                                data=sub_mean_poss,
                                compression='gzip')
        sub_root.create_dataset('mean_velocities',
                                shape=sub_mean_vels.shape,
                                dtype=float,
                                data=sub_mean_vels,
                                compression='gzip')
        sub_root.create_dataset('rms_spatial_radius',
                                shape=sub_rms_rs.shape,
                                dtype=sub_rms_rs.dtype,
                                data=sub_rms_rs,
                                compression='gzip')
        sub_root.create_dataset('rms_velocity_radius',
                                shape=sub_rms_vrs.shape,
                                dtype=sub_rms_vrs.dtype,
                                data=sub_rms_vrs,
                                compression='gzip')
        sub_root.create_dataset('1D_velocity_dispersion',
                                shape=sub_veldisp1ds.shape,
                                dtype=sub_veldisp1ds.dtype,
                                data=sub_veldisp1ds,
                                compression='gzip')
        sub_root.create_dataset('3D_velocity_dispersion',
                                shape=sub_veldisp3ds.shape,
                                dtype=sub_veldisp3ds.dtype,
                                data=sub_veldisp3ds,
                                compression='gzip')
        sub_root.create_dataset('nparts',
                                shape=subhalo_nparts.shape,
                                dtype=int,
                                data=subhalo_nparts,
                                compression='gzip')
        sub_root.create_dataset('total_masses',
                                shape=subhalo_masses.shape,
                                dtype=float,
                                data=subhalo_masses,
                                compression='gzip')
        sub_root.create_dataset('masses',
                                shape=subhalo_type_masses.shape,
                                dtype=float,
                                data=subhalo_type_masses,
                                compression='gzip')
        sub_root.create_dataset('real_flag',
                                shape=sub_reals.shape,
                                dtype=bool,
                                data=sub_reals,
                                compression='gzip')
        sub_root.create_dataset('halo_total_energies',
                                shape=subhalo_energies.shape,
                                dtype=float,
                                data=subhalo_energies,
                                compression='gzip')
        sub_root.create_dataset('halo_kinetic_energies',
                                shape=sub_KEs.shape,
                                dtype=float,
                                data=sub_KEs,
                                compression='gzip')
        sub_root.create_dataset('halo_gravitational_energies',
                                shape=sub_GEs.shape,
                                dtype=float,
                                data=sub_GEs,
                                compression='gzip')
        sub_root.create_dataset('v_max',
                                shape=sub_vmaxs.shape,
                                dtype=sub_vmaxs.dtype,
                                data=sub_vmaxs,
                                compression='gzip')
        sub_root.create_dataset('half_mass_radius',
                                shape=sub_hmrs.shape,
                                dtype=sub_hmrs.dtype,
                                data=sub_hmrs,
                                compression='gzip')
        sub_root.create_dataset('half_mass_velocity_radius',
                                shape=sub_hmvrs.shape,
                                dtype=sub_hmvrs.dtype,
                                data=sub_hmvrs,
                                compression='gzip')
        sub_root.create_dataset('exit_vlcoeff',
                                shape=sub_exit_vlcoeff.shape,
                                dtype=sub_exit_vlcoeff.dtype,
                                data=sub_exit_vlcoeff,
                                compression='gzip')

    snap.create_dataset('occupancy',
                        shape=nsubhalos.shape,
                        dtype=nsubhalos.dtype,
                        data=nsubhalos,
                        compression='gzip')

    snap.close()


