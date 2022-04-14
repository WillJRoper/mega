import sys

import core.build_graph_mpi as bgmpi
import core.kdhalofinder_mpi as kdmpi
import core.param_utils as p_utils
import graph_core.mergergraph as mgmpi
import mpi4py
import numpy as np
# import mergertrees as mt
# import lumberjack as ld
from core.talking_utils import say_hello
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
                            flags["DMO"], periodic=simulation["periodic"])

    # Include MPI information in metadata object
    meta.nranks = size
    meta.rank = rank

    if meta.rank == 0:
        say_hello(meta)
        print("Running on snapshot:", snaplist[snap_ind])

    # Lets check what sort of verbosity we are running with
    meta.check_verbose()

    # ===================== Run The Halo Finder =====================
    kdmpi.hosthalofinder(meta)


if __name__ == "__main__":
    exit(main())
