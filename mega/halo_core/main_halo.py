import sys

import mega.halo_core.kdhalofinder as kdmpi
import mega.core.param_utils as p_utils
from mega.core.talking_utils import say_hello

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
        print("Running on snapshot %s with %d ranks" % (snaplist[snap_ind],
                                                        meta.nranks))

    # Lets check what sort of verbosity we are running with
    meta.check_verbose()

    # ===================== Run The Halo Finder =====================
    kdmpi.hosthalofinder(meta)


if __name__ == "__main__":
    exit(main())
