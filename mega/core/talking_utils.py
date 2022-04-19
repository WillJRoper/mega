import numpy as np


def say_hello(meta):
    """

    :param meta:
    :return:
    """

    hello = [r"          __/\\\\____________/\\\\___/\\\\\\\\\\\\\\\______/\\\\\\\\\\\\______/\\\\\\\\\--O-O-O-O-O-O-O         O-O         0          ",
             r"           _\/\\\\\\________/\\\\\\__\/\\\///////////_____/\\\//////////_____/\\\\\\\\\\\\\__      \ / \       /           /           ",
             r"            _\/\\\//\\\____/\\\//\\\__\/\\\_______________/\\\_______________/\\\/////////\\\_      O   O-O-O-O     O     O            ",
             r"             _\/\\\\///\\\/\\\/_\/\\\__\/\\\\\\\\\\\______\/\\\____/\\\\\\\__\/\\\_______\/\\\_          \     \   / \   /             ",
             r"              _\/\\\__\///\\\/___\/\\\__\/\\\///////_______\/\\\___\/////\\\__\/\\\\\\\\\\\\\\\-O-O-O-O   O-O-O-O-O-O-O-O-O-O-O-O-O-O-O",
             r"               _\/\\\____\///_____\/\\\__\/\\\______________\/\\\_______\/\\\__\/\\\/////////\\\_      \ /   \       /     \           ",
             r"                _\/\\\_____________\/\\\__\/\\\______________\/\\\_______\/\\\__\/\\\_______\/\\\_      O     O     O       O          ",
             r"                 _\/\\\_____________\/\\\__\/\\\\\\\\\\\\\\\__\//\\\\\\\\\\\\/___\/\\\_______\/\\\_    / \     \   /                   ",
             r"                  _\///______________\///___\///////////////____\////////////_____\///________\///--O-O   O-O-O-O-O                    "]

    print()
    print()
    for line in hello:
        print('{:^{width}s}'.format(line, width=meta.table_width))
    print()
    print()


def message(rank, *args):
    """

    :param rank:
    :param args:
    :return:
    """

    print("[%s]" % ("%d" % rank).zfill(5) + ":", *args)


def pad_print_middle(string1, string2, length):
    """

    :param string1:
    :param string2:
    :param length:
    :return:
    """

    if type(string1) == list:
        string1 = "[" + " ".join(str(x) for x in string1) + "]"
    if type(string2) == list:
        string2 = "[" + " ".join(str(x) for x in string2) + "]"

    if type(string1) != str:
        string1 = str(string1)
    if type(string2) != str:
        string2 = str(string2)

    return string1 + " " * (length - (len(string1) + len(string2))) + string2


def get_heading(width, title):
    """

    :param width:
    :param title:
    :return:
    """

    report_pad = width // 2 - len(title) // 2
    report_string = ((report_pad - 1) * "="
                     + " %s " % title
                     + "=" * (report_pad - 1))

    return report_string


def count_and_report_halos(part_haloids, meta, halo_type="Spatial Host"):
    """

    :param part_haloids:
    :param meta:
    :param halo_type:
    :return:
    """

    # Get the unique halo ids
    unique, counts = np.unique(part_haloids, return_counts=True)

    # Print the number of halos found by the halo finder in npart bins
    report_string = get_heading(meta.report_width, halo_type)
    message(meta.rank, report_string)
    message(meta.rank, pad_print_middle("N_{part}",
                           "N_{%s}" % "_".join(halo_type.lower().split()),
                           length=meta.report_width))

    # First loop over some standard counts
    for count in [10, 15, 20, 50, 100, 500]:
        halos = unique[np.where(counts >= count)].size - 1
        message(meta.rank, pad_print_middle(">%d" % count,
                               str(halos),
                               length=meta.report_width))

    # Then loop in powers of 10 until we find 0 particles
    halos = np.inf
    count = 1000
    while halos > 0:
        halos = unique[np.where(counts >= count)].size - 1
        message(meta.rank, pad_print_middle(">%d" % count,
                               str(halos),
                               length=meta.report_width))
        count *= 10

    message(meta.rank, "=" * len(report_string))


def count_and_report_progs(nprogs, meta, halo_type="Host"):
    """

    :param nprogs:
    :param meta:
    :param halo_type:
    :return:
    """

    # Get the unique halo ids
    unique, counts = np.unique(nprogs, return_counts=True)

    # Print the number of halos found by the halo finder in npart bins
    report_string = get_heading(meta.report_width, halo_type + " Progenitors")
    message(meta.rank, report_string)
    message(meta.rank, pad_print_middle("N_{prog}",
                           "N_{%s}" % "_".join(halo_type.lower().split()),
                           length=meta.report_width))

    if len(nprogs) == 0:
        message(meta.rank, pad_print_middle("N/A",
                                            "N/A",
                                            length=meta.report_width))

    # First loop over some small values
    for u, c in zip(unique, counts):
        if u < 10:
            message(meta.rank, pad_print_middle("%d" % u,
                                   str(c),
                                   length=meta.report_width))

    message(meta.rank, "=" * len(report_string))


def count_and_report_descs(ndescs, meta, halo_type="Host"):
    """

    :param ndescs:
    :param meta:
    :param halo_type:
    :return:
    """

    # Get the unique halo ids
    unique, counts = np.unique(ndescs, return_counts=True)

    # Print the number of halos found by the halo finder in npart bins
    report_string = get_heading(meta.report_width, halo_type + " Descendants")
    message(meta.rank, report_string)
    message(meta.rank, pad_print_middle("N_{desc}",
                           "N_{%s}" % "_".join(halo_type.lower().split()),
                           length=meta.report_width))

    if len(ndescs) == 0:
        message(meta.rank, pad_print_middle("N/A",
                                            "N/A",
                                            length=meta.report_width))

    # First loop over some standard counts
    for u, c in zip(unique, counts):
        message(meta.rank, pad_print_middle("%d" % u,
                               str(c),
                               length=meta.report_width))

    message(meta.rank, "=" * len(report_string))
