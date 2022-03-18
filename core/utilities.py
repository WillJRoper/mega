import sys
import inspect
import logging
import networkx
import numpy as np

from core.halo import Halo
from core.serial_io import hdf5_write_dataset
from core.timing import timer
from core.talking_utils import message, pad_print_middle
from networkx.algorithms.components.connected import connected_components


logger = logging.getLogger(__name__)


def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes

    :param obj:
    :param seen:
    :return:
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(
                        d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj,
                                                     (str, bytes, bytearray)):
        try:
            size += sum((get_size(i, seen) for i in obj))
        except TypeError:
            logging.exception(
                "Unable to get size of %r. This may lead to incorrect sizes. "
                "Please report this error.",
                obj)
    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if
                    hasattr(obj, s))

    return size


def get_cellid(cdim, i, j, k):
    """

    :param cdim:
    :param i:
    :param j:
    :param k:
    :return:
    """
    return (k + cdim * (j + cdim * i))


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def wrap_pos(pos, boxsize):
    """

    :param pos:
    :param boxsize:
    :return:
    """
    # Define the comparison particle as the maximum
    # position in the current dimension
    max_part_pos = pos.max(axis=0)

    # Compute all the halo particle separations from the maximum position
    sep = max_part_pos - pos

    # If any separations are greater than 50% the boxsize
    # (i.e. the halo is split over the boundary)
    # bring the particles at the lower boundary together
    # with the particles at the upper boundary
    # (ignores halos where constituent particles aren't
    # separated by at least 50% of the boxsize)
    # *** Note: fails if halo's extent is greater than 50%
    # of the boxsize in any dimension ***
    pos[np.where(sep > 0.5 * boxsize)] += boxsize

    return pos


def decomp_nodes(npart, ranks, cells_per_rank, rank):
    """

    :param npart:
    :param ranks:
    :param cells_per_rank:
    :param rank:
    :return:
    """

    # Define the limits for particles on all ranks
    rank_edges = np.linspace(0, npart, ranks + 1, dtype=int)
    rank_cell_edges = np.linspace(rank_edges[rank], rank_edges[rank + 1],
                                  cells_per_rank + 1, dtype=int)

    # Define the nodes
    tasks = []
    for low, high in zip(rank_cell_edges[:-1], rank_cell_edges[1:]):
        tasks.append(np.arange(low, high, dtype=int))

    nnodes = cells_per_rank * ranks

    # Get the particles in this rank
    parts_in_rank = np.arange(rank_edges[rank], rank_edges[rank + 1],
                              dtype=int)

    return tasks, parts_in_rank, nnodes, rank_edges


@timer("Sorting")
def set_2_sorted_array(tictoc, s):

    # Convert to array and sort
    sarr = np.array(list(s))
    sarr.sort()

    return sarr


@timer("Sorting")
def timed_sort(tictoc, arr):
    return np.sort(arr)


def get_linked_halo_data(all_linked_halos, start_ind, nlinked_halos):
    """ A helper function for extracting a halo's linked halos
        (i.e. progenitors and descendants)

    :param all_linked_halos: Array containing all progenitors and descendants.
    :type all_linked_halos: float[N_linked halos]
    :param start_ind: The start index for this halos progenitors or
    descendents elements in all_linked_halos
    :type start_ind: int
    :param nlinked_halos: The number of progenitors or descendents
    (linked halos) the halo in question has
    :type nlinked_halos: int
    :return:
    """

    return all_linked_halos[start_ind: start_ind + nlinked_halos]
