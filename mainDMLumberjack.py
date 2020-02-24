import kdhalofinder as kd
import mergergraph as mg
import mergertrees as mt
import lumberjack as ld
import treedataloader as tdl
import time
import h5py
import os
import shutil
import multiprocessing as mp


def main_kd(snap):
    kd.hosthalofinder(snap, batchsize=2000000, savepath='halo_snapshots/', vlcoeff=10.)


def main_mg(snap):
    mg.directProgDescWriter(snap, halopath='halo_snapshots/', savepath='MergerGraphs/', part_threshold=10, rank=0)


def main_mg_spatial(snap):
    mg.directProgDescWriter(snap, halopath='halo_snapshots_spatial/',
                            savepath='MergerGraphs_spatial/', part_threshold=10)


def main_mt(snap):
    mt.directProgDescWriter(snap, part_threshold=10, halopath='split_halos/', savepath='MergerTrees/')


# Create a snapshot list (past to present day) for looping
snaplist = []
for snap in range(0, 62):
    if snap < 10:
        snaplist.append('00' + str(snap))
    elif snap >= 10:
        snaplist.append('0' + str(snap))

start = time.time()
if __name__ == '__main__':

    # # Empty and create all directories necessary to store results
    # if 'halo_snapshots' in os.listdir(os.getcwd()):
    #     shutil.rmtree('halo_snapshots')
    #     os.mkdir('halo_snapshots')
    # else:
    #     os.mkdir('halo_snapshots')
    #
    # if 'split_halos' in os.listdir(os.getcwd()):
    #     shutil.rmtree('split_halos/')
    #     os.mkdir('split_halos/')
    # else:
    #     os.mkdir('split_halos/')
    #
    # if 'MergerGraphs' in os.listdir(os.getcwd()):
    #     shutil.rmtree('MergerGraphs')
    #     os.mkdir('MergerGraphs')
    # else:
    #     os.mkdir('MergerGraphs')
    #
    # if 'MergerTrees' in os.listdir(os.getcwd()):
    #     shutil.rmtree('MergerTrees')
    #     os.mkdir('MergerTrees')
    # else:
    #     os.mkdir('MergerTrees')

    # # ===================== Run The Halo Finder =====================
    # pool = mp.Pool(int(mp.cpu_count() - 6))
    # pool.map(main_kd, snaplist)

    # pool.close()
    # pool.join()

    # for snap in snaplist:
    #
    #     hdf = h5py.File('halo_snapshots/halos_' + snap + '.hdf5', 'r+', driver='core')
    #
    #     for halo in hdf.keys():
    #
    #         print(halo, snap, end='\r')
    #
    #         try:
    #             if hdf[halo].attrs['halo_energy'] <= 0:
    #                 hdf[halo].attrs['Real'] = True
    #                 continue
    #         except KeyError:
    #             continue
    #         if hdf[halo].attrs['halo_energy'] > 0:
    #             hdf[halo].attrs['Real'] = False
    #
    #         print(halo, 'reset to', hdf[halo].attrs['halo_energy'], hdf[halo].attrs['Real'])
    #
    #     hdf.close()

    # # ===================== Find Direct Progenitors and Descendents =====================
    # for snap in snaplist:
    #     main_mg(snap)
    #
    # # ===================== Build The Graphs =====================
    # mg.get_graph_members(treepath='MergerGraphs/Mgraph_', graphpath='MergerGraphs/FullMgraphs',
    #                      halopath='halo_snapshots/halos_')

    # # ===================== Build Graph Dictionary =====================
    # tdl.loader('MergerGraphs/Mgraph_', 'halo_snapshots/halos_', extracttree=True, extractsnap=False,
    #            halo_attributes=[], snap_attributes=['time'], savepath='treedata_graph.pck')
    #
    # # ===================== Split Graphs Into Trees =====================
    # ld.mainLumberjack(halopath='halo_snapshots/', newhalopath='split_halos/')
    #
    # # ===================== Find Post Splitting Direct Progenitors and Descendents =====================
    # for snap in snaplist:
    #     main_mt(snap)
    # mt.link_cutter(treepath='MergerTrees/Mtree_')
    #
    # # ===================== Build The Trees =====================
    # mt.get_graph_members(treepath='MergerTrees/Mtree_', graphpath='MergerTrees/FullMtrees',
    #                      halopath='split_halos/halos_')
    #
    # # ===================== Build Dictionaries =====================
    # tdl.loader('MergerTrees/Mtree_', 'split_halos/halos_', extracttree=True, halo_attributes=[],
    #            snap_attributes=['time'], savepath='treedata_tree.pck')
#
#     # # ===================== Find Direct Progenitors and Descendents =====================
#     for snap in snaplist:
#         main_mg(snap)
#     # mg.notreal_extract(treepath='MergerGraphs_spatial/Mgraph_', halopath='halo_snapshots_spatial/halos_')
#
#     # # ===================== Build The Graphs =====================
#     # mg.get_graph_members(treepath='MergerGraphs_spatial/Mgraph_', graphpath='MergerGraphs_spatial/FullMgraphs')
#
#
# print('Total: ', time.time() - start)
#
