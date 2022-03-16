import numpy as np
from scipy.spatial import cKDTree
import sys
import utilities

from halo import Halo


def find_phase_space_halos(halo_phases):
    # =============== Initialise The Halo Finder Variables/Arrays and The KD-Tree ===============

    # Initialise arrays and dictionaries for storing halo data
    phase_part_haloids = np.full(halo_phases.shape[0], -1,
                                 dtype=int)  # halo ID of the halo each particle is in
    phase_assigned_parts = {}  # Dictionary to store the particles in a particular halo
    # A dictionary where each key is an initial subhalo ID and the item is the subhalo IDs it has been linked with
    phase_linked_halos_dict = {}
    # Final halo ID of linked halos (index is initial halo ID)
    final_phasehalo_ids = np.full(halo_phases.shape[0], -1, dtype=int)

    # Initialise subhalo ID counter (IDs start at 0)
    ihaloid = -1

    # Initialise the halo kd tree in 6D phase space
    halo_tree = cKDTree(halo_phases, leafsize=16, compact_nodes=True,
                        balanced_tree=True)

    query = halo_tree.query_ball_point(halo_phases, r=np.sqrt(2))

    # Loop through query results assigning initial halo IDs
    for query_part_inds in iter(query):

        query_part_inds = np.array(query_part_inds, dtype=int)

        # If only one particle is returned by the query and it is new it is a 'single particle halo'
        if query_part_inds.size == 1:
            # Assign the 'single particle halo' halo ID to the particle
            phase_part_haloids[query_part_inds] = -2

        # # Find the previous halo ID associated to these particles
        # this_halo_ids = halo_ids[query_part_inds]
        # uni_this_halo_ids = set(this_halo_ids)
        # if len(uni_this_halo_ids) > 1:
        #     query_part_inds = query_part_inds[np.where(this_halo_ids == this_halo_ids[0])]

        # Find only the particles not already in a halo
        new_parts = query_part_inds[
            np.where(phase_part_haloids[query_part_inds] < 0)]

        # If all particles are new increment the halo ID and assign a new halo
        if new_parts.size == query_part_inds.size:

            # Increment the halo ID by 1 (initialising a new halo)
            ihaloid += 1

            # Assign the new halo ID to the particles
            phase_part_haloids[new_parts] = ihaloid
            phase_assigned_parts[ihaloid] = set(new_parts)

            # Assign the final halo ID to be the newly assigned halo ID
            final_phasehalo_ids[ihaloid] = ihaloid
            phase_linked_halos_dict[ihaloid] = {ihaloid}

        else:

            # ===== Get the 'final halo ID value' =====

            # Extract the IDs of halos returned by the query
            contained_halos = phase_part_haloids[query_part_inds]

            # Get only the unique halo IDs
            uni_cont_halos = np.unique(contained_halos)

            # Remove any unassigned halos
            uni_cont_halos = uni_cont_halos[np.where(uni_cont_halos >= 0)]

            # If there is only one halo ID returned avoid the slower code to combine IDs
            if uni_cont_halos.size == 1:

                # Get the list of linked halos linked to the current halo from the linked halo dictionary
                linked_halos = phase_linked_halos_dict[uni_cont_halos[0]]

            elif uni_cont_halos.size == 0:
                continue

            else:

                # Get all the linked halos from dictionary so as to not miss out any halos IDs that are linked
                # but not returned by this particular query
                linked_halos_set = set()  # initialise linked halo set
                linked_halos = linked_halos_set.union(
                    *[phase_linked_halos_dict.get(halo)
                      for halo in uni_cont_halos])

            # Find the minimum halo ID to make the final halo ID
            final_ID = min(linked_halos)

            # Assign the linked halos to all the entries in the linked halos dictionary
            phase_linked_halos_dict.update(
                dict.fromkeys(list(linked_halos), linked_halos))

            # Assign the final halo ID array entries
            final_phasehalo_ids[list(linked_halos)] = final_ID

            # Assign new particles to the particle halo IDs array with the final ID
            phase_part_haloids[new_parts] = final_ID

            # Assign the new particles to the final ID in halo dictionary entry
            phase_assigned_parts[final_ID].update(new_parts)

    # =============== Reassign All Halos To Their Final Halo ID ===============

    # Loop over initial halo IDs reassigning them to the final halo ID
    for halo_id in list(phase_assigned_parts.keys()):

        # Extract the final halo value
        final_ID = final_phasehalo_ids[halo_id]

        # Assign this final ID to all the particles in the initial halo ID
        phase_part_haloids[list(phase_assigned_parts[halo_id])] = final_ID
        phase_assigned_parts[final_ID].update(phase_assigned_parts[halo_id])

        # Remove non final ID entry from dictionary to save memory
        if halo_id != final_ID:
            phase_assigned_parts.pop(halo_id)

    return phase_part_haloids, phase_assigned_parts


def get_real_host_halos(halo, boxsize, vlinkl_halo_indp, linkl, decrement,
                        redshift, G, h, soft, min_vlcoeff, cosmo):
    # Initialise list to store finished halos
    results = []

    # Define the phase space linking length
    vlinkl = halo.vlcoeff * vlinkl_halo_indp * halo.mass ** (1 / 3)

    # Define the phase space vectors for this halo
    halo_phases = np.concatenate((halo.pos / linkl, halo.vel / vlinkl), axis=1)

    # Query these particles in phase space to find distinct bound halos
    part_haloids, assigned_parts = find_phase_space_halos(halo_phases)

    # Loop over the halos found in phase space
    while len(assigned_parts) > 0:

        # Get the next halo from the dictionary and ensure
        # it has more than 10 particles
        key, val = assigned_parts.popitem()
        if len(val) < 10:
            continue

        # Extract halo particle data
        this_pids = list(val)

        # Instantiate halo object (auto calculates energy)
        new_halo = Halo(halo.pids[this_pids],
                        halo.sim_pids[this_pids],
                        halo.pos[this_pids, :],
                        halo.vel[this_pids, :],
                        halo.types[this_pids],
                        halo.masses[this_pids],
                        halo.vlcoeff,
                        boxsize, soft, redshift, G, cosmo)

        if new_halo.real or new_halo.vlcoeff <= min_vlcoeff:

            # Compute the halo properties
            new_halo.compute_props(G)

            # # Store the resulting halo
            # results.append(new_halo)


            results.append({'pids': new_halo.pids,
                                     'sim_pids': new_halo.sim_pids,
                                     'npart': new_halo.npart,
                                     'real': new_halo.real,
                                     'mean_halo_pos': new_halo.mean_pos,
                                     'mean_halo_vel': new_halo.mean_vel,
                                     'halo_mass': new_halo.mass,
                                     'halo_ptype_mass': new_halo.ptype_mass,
                                     'halo_energy': new_halo.KE - new_halo.GE,
                                     'KE': new_halo.KE, 'GE': new_halo.GE,
                                     "rms_r": new_halo.rms_r,
                                     "rms_vr": new_halo.rms_vr,
                                     "veldisp3d": new_halo.veldisp3d,
                                     "veldisp1d": new_halo.veldisp1d,
                                     "vmax": new_halo.vmax,
                                     "hmr": new_halo.hmr,
                                     "hmvr": new_halo.hmvr,
                                     "vlcoeff": new_halo.vlcoeff,
                            "memory": utilities.get_size(new_halo)})

        else:

            # Decrement the velocity space linking length coefficient
            new_halo.decrement(decrement)

            # We need to run this halo again
            temp_res = get_real_host_halos(new_halo, boxsize, vlinkl_halo_indp,
                                           linkl, decrement, redshift, G, h,
                                           soft, min_vlcoeff, cosmo)

            # Include these results
            for h in temp_res:
                results.append(h)

    return results


def get_real_host_halos_iterate(sim_halo_pids, halo_poss, halo_vels, boxsize,
                                vlinkl_halo_indp, linkl, halo_masses,
                                halo_part_types,
                                ini_vlcoeff, decrement, redshift, G, h, soft,
                                min_vlcoeff, cosmo):
    # Initialise dicitonaries to store results
    results = {}

    # Define the comparison particle as the maximum position
    # in the current dimension
    max_part_pos = halo_poss.max(axis=0)

    # Compute all the halo particle separations from the maximum position
    sep = max_part_pos - halo_poss

    # If any separations are greater than 50% the boxsize
    # (i.e. the halo is split over the boundary)
    # bring the particles at the lower boundary together with
    # the particles at the upper boundary (ignores halos where
    # constituent particles aren't separated by at least 50% of the boxsize)
    # *** Note: fails if halo's extent is greater than 50% of
    # the boxsize in any dimension ***
    halo_poss[np.where(sep > 0.5 * boxsize)] += boxsize

    not_real_pids = {}
    candidate_halos = {0: {"pos": halo_poss,
                           "vel": halo_vels,
                           "pid": sim_halo_pids,
                           "mass": np.sum(halo_masses),
                           "masses": halo_masses,
                           "part_types": halo_part_types,
                           "vlcoeff": ini_vlcoeff}}
    candidateID = 0
    thisresultID = 0

    while len(candidate_halos) > 0:

        key, candidate_halo = candidate_halos.popitem()

        halo_poss = candidate_halo["pos"]
        halo_vels = candidate_halo["vel"]
        sim_halo_pids = candidate_halo["pid"]
        halo_masses = candidate_halo["masses"]
        halo_mass = candidate_halo["mass"]
        halo_part_types = candidate_halo["part_types"]
        new_vlcoeff = candidate_halo["vlcoeff"]

        # Define the phase space linking length
        vlinkl = (new_vlcoeff * vlinkl_halo_indp * halo_mass ** (1 / 3))

        # Add the hubble flow to the velocities
        # *** NOTE: this DOES NOT include a gadget factor of a^-1/2 ***
        ini_cent = np.mean(halo_poss, axis=0)
        sep = cosmo.H(redshift).value * (halo_poss - ini_cent)
        halo_vels_with_HubFlow = halo_vels + sep

        # Define the phase space vectors for this halo
        halo_phases = np.concatenate((halo_poss / linkl,
                                      halo_vels_with_HubFlow / vlinkl), axis=1)

        # Query these particles in phase space to find distinct bound halos
        part_haloids, assigned_parts = find_phase_space_halos(halo_phases)

        not_real_pids = {}

        thiscontID = 0
        while len(assigned_parts) > 0:

            # Get the next halo from the dictionary and ensure
            # it has more than 10 particles
            key, val = assigned_parts.popitem()
            if len(val) < 10:
                continue

            # Extract halo particle data
            this_halo_pids = list(val)
            halo_npart = len(this_halo_pids)
            this_halo_pos = halo_poss[this_halo_pids, :]
            this_halo_vel = halo_vels[this_halo_pids, :]
            this_halo_masses = halo_masses[this_halo_pids]
            this_part_types = halo_part_types[this_halo_pids]
            this_sim_halo_pids = sim_halo_pids[this_halo_pids]

            # Compute the centred positions and velocities
            mean_halo_pos = np.average(this_halo_pos,
                                       weights=this_halo_masses,
                                       axis=0)
            mean_halo_vel = np.average(this_halo_vel,
                                       weights=this_halo_masses,
                                       axis=0)

            # Centre positions and velocities relative to COM
            this_halo_pos -= mean_halo_pos
            this_halo_vel -= mean_halo_vel

            # Compute halo's energy
            halo_energy, KE, GE = halo_energy_calc(this_halo_pos,
                                                   this_halo_vel,
                                                   halo_npart,
                                                   this_halo_masses, redshift,
                                                   G, h, soft)

            # Add the hubble flow to the velocities
            # *** NOTE: this DOES NOT include a gadget factor of a^-1/2 ***
            sep = cosmo.H(redshift).value * this_halo_pos
            this_halo_vel += sep

            if KE / GE <= 1 or new_vlcoeff <= min_vlcoeff:

                # Get rms radii from the centred position and velocity
                r = hprop.rms_rad(this_halo_pos)
                vr = hprop.rms_rad(this_halo_vel)

                # Compute the velocity dispersion
                veldisp3d, veldisp1d = hprop.vel_disp(this_halo_vel)

                # Compute maximal rotational velocity
                vmax = hprop.vmax(this_halo_pos, this_halo_masses, G)

                # Calculate half mass radius in position and velocity space
                hmr = hprop.half_mass_rad(this_halo_pos, this_halo_masses)
                hmvr = hprop.half_mass_rad(this_halo_vel, this_halo_masses)

                # Define mass in each particle type
                part_type_mass = [
                    np.sum(this_halo_masses[this_part_types == i])
                    for i in range(6)]

                results[thisresultID] = {'pids': this_sim_halo_pids,
                                         'npart': halo_npart,
                                         'real': KE / GE <= 1,
                                         'mean_halo_pos': mean_halo_pos,
                                         'mean_halo_vel': mean_halo_vel,
                                         'halo_mass': np.sum(this_halo_masses),
                                         'halo_ptype_mass': part_type_mass,
                                         'halo_energy': halo_energy,
                                         'KE': KE, 'GE': GE,
                                         "rms_r": r, "rms_vr": vr,
                                         "veldisp3d": veldisp3d,
                                         "veldisp1d": veldisp1d,
                                         "vmax": vmax,
                                         "hmr": hmr,
                                         "hmvr": hmvr,
                                         "vlcoeff": new_vlcoeff}

                thisresultID += 1

            else:
                not_real_pids[thiscontID] = this_halo_pids
                candidate_halos[candidateID] = {"pos": (this_halo_pos
                                                        + mean_halo_pos),
                                                "vel": (this_halo_vel
                                                        + mean_halo_vel),
                                                "pid": this_sim_halo_pids,
                                                "mass": np.sum(
                                                    this_halo_masses),
                                                "masses": this_halo_masses,
                                                "part_types": this_part_types,
                                                "vlcoeff": new_vlcoeff - decrement}

                candidateID += 1
                thiscontID += 1

    return results

