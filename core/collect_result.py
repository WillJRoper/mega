import numpy as np

from core.timing import timer
from core.talking_utils import message, count_and_report_halos


@timer("Collecting")
def collect_halos(tictoc, meta, collected_results, sub_collected_results):
    """

    :param tictoc:
    :param meta:
    :param collected_results:
    :param sub_collected_results:
    :return:
    """

    # Intialise halo ID counters, dictionaries and array
    newPhaseID = 0
    newPhaseSubID = 0
    haloID_dict = {}
    subhaloID_dict = {}
    phase_part_haloids = np.full((np.sum(meta.npart), 2), -2,
                                 dtype=np.int32)

    # Collect host halo results
    results_dict = {}
    memory_use = 0
    for halo_task in collected_results:
        for halo_key in halo_task:
            halo = halo_task[halo_key]
            results_dict[(halo_key, newPhaseID)] = halo
            pids = halo.pids
            haloID_dict[(halo_key, newPhaseID)] = newPhaseID
            phase_part_haloids[pids, 0] = newPhaseID
            newPhaseID += 1
            if meta.profile:
                memory_use += halo.memory

    message(meta.rank, "Halo objects total footprint: %.2f MB"
            % (memory_use * 10 ** -6))

    # Collect subhalo results
    sub_results_dict = {}
    memory_use = 0
    for subhalo_task in sub_collected_results:
        for subhalo_key in subhalo_task:
            subhalo = subhalo_task[subhalo_key]
            sub_results_dict[(subhalo_key, newPhaseSubID)] = subhalo
            pids = subhalo.pids
            subhaloID_dict[(subhalo_key, newPhaseSubID)] = newPhaseSubID
            phase_part_haloids[pids, 1] = newPhaseSubID
            newPhaseSubID += 1
            if meta.profile:
                memory_use += subhalo.memory

    message(meta.rank, "Subhalo objects total footprint: %.2f MB"
            % (memory_use * 10 ** -6))

    count_and_report_halos(phase_part_haloids[:, 0], meta,
                           halo_type="Phase Space Host Halos")

    if meta.findsubs:
        count_and_report_halos(phase_part_haloids[:, 1], meta,
                               halo_type="Phase Space Subhalos")

    return (newPhaseID, newPhaseSubID, results_dict, haloID_dict,
            sub_results_dict, subhaloID_dict, phase_part_haloids)