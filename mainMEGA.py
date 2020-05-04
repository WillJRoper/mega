import kdhalofinder as kd
import mergergraph as mg
import mergertrees as mt
import lumberjack as ld
import time
import sys
import multiprocessing as mp
import utilities


# Assign start time
walltime_start = time.time()

# Read the parameter file
paramfile = sys.argv[1]
inputs, flags, params = utilities.read_param(paramfile)

# Get the snapshot IDs
snaplist = np.loadtxt('snaplist.txt', fmt='%s')


def main_kd(snap):
    kd.hosthalofinder(snap, llcoeff=params['llcoef'], sub_llcoeff=params['sub_llcoef'], gadgetpath=inputs['data'],
                      batchsize=params['batchsize'],  savepath=inputs['haloSavePath'], vlcoeff=params['ini_alpha_v'],
                      decrement=params['decrement'], verbose=flags['verbose'])


def main_mg(snap, rank):
    mg.directProgDescWriter(snap, halopath=inputs['haloSavePath'], savepath=inputs['directgraphSavePath'],
                            part_threshold=params['part_threshold'], rank=rank)


def main_mt(snap):
    mt.directProgDescWriter(snap, halopath=inputs['treehaloSavePath'], savepath=inputs['directtreeSavePath'],
                            part_threshold=params['part_threshold'])


if flags['serial']:

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
        mg.get_graph_members(treepath=inputs['directgraphSavePath'] + '/Mgraph_',
                             graphpath=inputs['graphSavePath'] +'/FullMgraphs',
                             halopath=inputs['haloSavePath'] + '/halos_')

    # ===================== Split Graphs Into Trees =====================
    if flags['treehalos']:
        ld.mainLumberjack(halopath=inputs['haloSavePath'], newhalopath=inputs['treehaloSavePath'])

    # ===================== Find Post Splitting Direct Progenitors and Descendents =====================
    if flags['treedirect']:
        for snap in snaplist:
            main_mt(snap)
        mt.link_cutter(treepath=inputs['directtreeSavePath'] + '/Mtree_')

    # ===================== Build The Trees =====================
    if flags['tree']:
        mt.get_graph_members(treepath=inputs['directtreeSavePath'] + '/Mtree_',
                             graphpath=inputs['treeSavePath'] + '/FullMtrees',
                             halopath=inputs['treehaloSavePath'] + '/halos_')

elif flags['multiprocessing']:

    # ===================== Run The Halo Finder =====================
    if flags['halo']:
        pool = mp.Pool(int(mp.cpu_count() - 6))
        pool.map(main_kd, snaplist)

        pool.close()
        pool.join()

    # ===================== Find Direct Progenitors and Descendents =====================
    if flags['graphdirect']:
        for snap in snaplist:
            main_mg(snap, 0)
    if flags['subgraphdirect']:
        for snap in snaplist:
            main_mg(snap, 1)

    # ===================== Build The Graphs =====================
    if flags['graph']:
        mg.get_graph_members(treepath=inputs['directgraphSavePath'] + '/Mgraph_',
                             graphpath=inputs['graphSavePath'] +'/FullMgraphs',
                             halopath=inputs['haloSavePath'] + '/halos_')

    # ===================== Split Graphs Into Trees =====================
    if flags['treehalos']:
        ld.mainLumberjack(halopath=inputs['haloSavePath'], newhalopath=inputs['treehaloSavePath'])

    # ===================== Find Post Splitting Direct Progenitors and Descendents =====================
    if flags['treedirect']:
        for snap in snaplist:
            main_mt(snap)
        mt.link_cutter(treepath=inputs['directtreeSavePath'] + '/Mtree_')

    # ===================== Build The Trees =====================
    if flags['tree']:
        mt.get_graph_members(treepath=inputs['directtreeSavePath'] + '/Mtree_',
                             graphpath=inputs['treeSavePath'] + '/FullMtrees',
                             halopath=inputs['treehaloSavePath'] + '/halos_')


print('Total: ', time.time() - walltime_start)

