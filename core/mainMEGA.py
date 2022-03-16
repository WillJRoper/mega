import build_graph_mpi as bgmpi
import kdhalofinder as kd
import kdhalofinder_mpi as kdmpi
import mergergraph as mg
import mergergraph_mpi as mgmpi
import numpy as np
import sys
# import mergertrees as mt
# import lumberjack as ld
import time
import utilities
import param_utils as p_utils
from astropy.cosmology import FlatLambdaCDM

# Assign start time
walltime_start = time.time()

# Read the parameter file
paramfile = sys.argv[1]
inputs, flags, params, cosmology, simulation = p_utils.read_param(paramfile)

snap_ind = int(sys.argv[2])

# Load the snapshot list
snaplist = list(np.loadtxt(inputs['snapList'], dtype=str))

# Set up object containing housekeeping information
meta = p_utils.Metadata(snaplist[snap_ind], cosmology,
                        params['llcoeff'], params['sub_llcoeff'],
                        inputs['data'] + inputs["basename"],
                        inputs['haloSavePath'], params['ini_alpha_v'],
                        params['min_alpha_v'], params['decrement'],
                        flags['verbose'], flags['subs'],
                        params['N_cells'], flags['profile'],
                        inputs["profilingPath"], cosmology["h"],
                        (simulation["comoving_DM_softening"],
                         simulation["max_physical_DM_softening"]),
                        flags["DMO"])

# Extract the cosmology object
cosmo = meta.cosmo


def main_kd(snap):
    kd.hosthalofinder(snap, llcoeff=params['llcoeff'],
                      sub_llcoeff=params['sub_llcoeff'],
                      inputpath=inputs['data'],
                      batchsize=params['batchsize'],
                      savepath=inputs['haloSavePath'],
                      ini_vlcoeff=params['ini_alpha_v'],
                      min_vlcoeff=params['min_alpha_v'],
                      decrement=params['decrement'], verbose=flags['verbose'],
                      internal_input=flags['internalInput'],
                      findsubs=flags['subs'], h=cosmology["h"],
                      softs=(simulation["comoving_DM_softening"],
                             simulation["max_physical_DM_softening"]))


def main_kdmpi(meta):
    kdmpi.hosthalofinder(meta)


def main_mg(snap, density_rank):
    mg.directProgDescWriter(snap, halopath=inputs['haloSavePath'],
                            savepath=inputs['directgraphSavePath'],
                            part_threshold=params['part_threshold'],
                            density_rank=density_rank,
                            final_snapnum=len(snaplist))


def main_mgmpi(snap, prog_snap, desc_snap, density_rank):
    mgmpi.directProgDescWriter(snap, prog_snap, desc_snap,
                               halopath=inputs['haloSavePath'],
                               savepath=inputs['directgraphSavePath'],
                               density_rank=density_rank,
                               verbose=flags['verbose'],
                               profile=flags['profile'],
                               profile_path=inputs["profilingPath"])


# def main_mt(snap):
#     mt.directProgDescWriter(snap, halopath=inputs['treehaloSavePath'],
#                             savepath=inputs['directtreeSavePath'],
#                             part_threshold=params['part_threshold'])


if flags['useserial']:

    snaplist = [snaplist[snap_ind], ]

    # ===================== Run The Halo Finder =====================
    if flags['halo']:
        for snap in snaplist:
            main_kd(snap)

    # ===================== Find Direct Progenitors and Descendents =====================
    if flags['graphdirect']:
        for snap in snaplist:
            main_mg(snap, 0)
    if flags['subgraphdirect']:
        for snap in snaplist:
            main_mg(snap, 1)

    # ===================== Build The Graphs =====================
    if flags['graph']:
        mg.get_graph_members(
            treepath=inputs['directgraphSavePath'] + '/Mgraph_',
            graphpath=inputs['graphSavePath'] + '/FullMgraphs',
            halopath=inputs['haloSavePath'] + '/halos_')

    # ===================== Split Graphs Into Trees =====================
    # if flags['treehalos']:
    #     ld.mainLumberjack(halopath=inputs['haloSavePath'],
    #                       newhalopath=inputs['treehaloSavePath'])

    # # ===================== Find Post Splitting Direct Progenitors and Descendents =====================
    # if flags['treedirect']:
    #     for snap in snaplist:
    #         main_mt(snap)
    #     mt.link_cutter(treepath=inputs['directtreeSavePath'] + '/Mtree_')

    # # ===================== Build The Trees =====================
    # if flags['tree']:
    #     mt.get_graph_members(treepath=inputs['directtreeSavePath'] + '/Mtree_',
    #                          graphpath=inputs['treeSavePath'] + '/FullMtrees',
    #                          halopath=inputs['treehaloSavePath'] + '/halos_')

    print('Total: ', time.time() - walltime_start)

elif flags['usempi']:

    import mpi4py
    from mpi4py import MPI

    mpi4py.rc.recv_mprobe = False

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD  # get MPI communicator object
    size = comm.size  # total number of processes
    rank = comm.rank  # rank of this process
    status = MPI.Status()  # get MPI status object

    # Include MPI information in metadata object
    meta.nranks = size
    meta.rank = rank

    if rank == 0:
        print("Running on snapshot:", snaplist[snap_ind])

    # ===================== Run The Halo Finder =====================
    if flags['halo']:
        main_kdmpi(meta)

    # ===================== Find Direct Progenitors and Descendents =====================
    if flags['graphdirect']:

        snap = snaplist[snap_ind]

        if snap_ind - 1 < 0:
            prog_snap = None
        else:
            prog_snap = snaplist[snap_ind - 1]

        if snap_ind + 1 >= len(snaplist):
            desc_snap = None
        else:
            desc_snap = snaplist[snap_ind + 1]

        main_mgmpi(snap, prog_snap, desc_snap, 0)

    comm.barrier()

    if flags['subgraphdirect']:

        snap = snaplist[snap_ind]

        if snap_ind - 1 < 0:
            prog_snap = None
        else:
            prog_snap = snaplist[snap_ind - 1]

        if snap_ind + 1 >= len(snaplist):
            desc_snap = None
        else:
            desc_snap = snaplist[snap_ind + 1]

        main_mgmpi(snap, prog_snap, desc_snap, 1)

    if rank == 0:
        print('Total: ', time.time() - walltime_start)

    if flags["graph"]:
        bgmpi.main_get_graph_members(treepath=inputs['directgraphSavePath'],
                                     graphpath=inputs['graphSavePath'],
                                     snaplist=snaplist,
                                     verbose=flags['verbose'],
                                     halopath=inputs['haloSavePath'])
