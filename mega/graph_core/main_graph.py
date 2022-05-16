import sys

import mega.core.build_graph_mpi as bgmpi
import mega.core.param_utils as p_utils
import mega.graph_core.mergergraph as mgmpi
from mega.core.talking_utils import say_hello
from mega.core.timing import TicToc

import mpi4py
import numpy as np
from mpi4py import MPI

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


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
    meta = p_utils.Metadata(snaplist, snap_ind, cosmology, inputs,
                            flags, params, simulation)

    # Include MPI information in metadata object
    meta.nranks = size
    meta.rank = rank

    if meta.rank == 0:
        say_hello(meta)
        print("Running on snapshot:", snaplist[snap_ind])

    # Lets check what sort of verbosity we are running with
    meta.check_verbose()

    # Instantiate timer
    tictoc = TicToc(meta)
    tictoc.start()
    meta.tictoc = tictoc

    # ============== Find Direct Progenitors and Descendents ==============
    if flags['graphdirect']:
        mgmpi.graph_main(0, meta)

    comm.barrier()

    if flags['subgraphdirect']:
        mgmpi.graph_main(1, meta)

    # If we are at the final snapshot we have to clean up all links
    # if meta.isfinal:
        # walk_and_purge(tictoc, meta)

    tictoc.end()

    if meta.profile:
        tictoc.end_report(comm)


if __name__ == "__main__":
    exit(main())
