from mega.core.timing import timer
from mega.core.talking_utils import message


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

    # Collect host halo results
    results_dict = {}
    memory_use = 0
    for halo_task in collected_results:
        for halo_key in halo_task:
            halo = halo_task[halo_key]
            results_dict[(halo_key, newPhaseID)] = halo
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
            newPhaseSubID += 1
            if meta.profile:
                memory_use += subhalo.memory

    message(meta.rank, "Subhalo objects total footprint: %.2f MB"
            % (memory_use * 10 ** -6))

    return newPhaseID, newPhaseSubID, results_dict, sub_results_dict