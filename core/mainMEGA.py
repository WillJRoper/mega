import core.build_graph_mpi as bgmpi
import core.kdhalofinder_mpi as kdmpi
import graph_core.mergergraph as mgmpi
import numpy as np
import sys
# import mergertrees as mt
# import lumberjack as ld
from core.talking_utils import say_hello
import core.param_utils as p_utils

# TODO: Package mpi information in object similar to meta


def main():

    # Read the parameter file
    paramfile = sys.argv[1]
    (inputs, flags, params, cosmology,
     simulation) = p_utils.read_param(paramfile)

    snap_ind = int(sys.argv[2])

    # Load the snapshot list
    snaplist = list(np.loadtxt(inputs['snapList'], dtype=str))

    # Set up object containing housekeeping metadata
    meta = p_utils.Metadata(snaplist, snap_ind, cosmology,
                            params['llcoeff'], params['sub_llcoeff'], inputs,
                            inputs['data'] + inputs["basename"],
                            inputs['haloSavePath'], params['ini_alpha_v'],
                            params['min_alpha_v'], params['decrement'],
                            flags['verbose'], flags['subs'],
                            params['N_cells'], flags['profile'],
                            inputs["profilingPath"], cosmology["h"],
                            (simulation["comoving_DM_softening"],
                             simulation["max_physical_DM_softening"]),
                            flags["DMO"])

    if flags['useserial']:

        snaplist = [snaplist[snap_ind], ]

        # # ===================== Run The Halo Finder =====================
        # if flags['halo']:
        #     for snap in snaplist:
        #         main_kd(snap)
        #
        # # ===================== Find Direct Progenitors and Descendents =====================
        # if flags['graphdirect']:
        #     for snap in snaplist:
        #         main_mg(snap, 0)
        # if flags['subgraphdirect']:
        #     for snap in snaplist:
        #         main_mg(snap, 1)
        #
        # # ===================== Build The Graphs =====================
        # if flags['graph']:
        #     mg.get_graph_members(
        #         treepath=inputs['directgraphSavePath'] + '/Mgraph_',
        #         graphpath=inputs['graphSavePath'] + '/FullMgraphs',
        #         halopath=inputs['haloSavePath'] + '/halos_')

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

        # print('Total: ', time.time() - walltime_start)

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

        if meta.rank == 0:
            say_hello(meta)
            print("Running on snapshot:", snaplist[snap_ind])

        # Lets check what sort of verbosity we are running with
        meta.check_verbose()

        # ===================== Run The Halo Finder =====================
        if flags['halo']:
            kdmpi.hosthalofinder(meta)

        # ============== Find Direct Progenitors and Descendents ==============
        if flags['graphdirect']:

            mgmpi.graph_main(0, meta)

        comm.barrier()

        if flags['subgraphdirect']:

            mgmpi.graph_main(1, meta)

        if flags["graph"]:
            bgmpi.main_get_graph_members(treepath=inputs['directgraphSavePath'],
                                         graphpath=inputs['graphSavePath'],
                                         snaplist=snaplist,
                                         verbose=flags['verbose'],
                                         halopath=inputs['haloSavePath'])

if __name__ == "__main__":
    exit(main())
