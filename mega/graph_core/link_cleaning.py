import numpy as np
import h5py
from mega.core.timing import timer
from mega.graph_core.graph_halo import Janitor_Halo as Halo
from mega.core.serial_io import write_cleaned_dgraph_data, clean_real_flags


def clean_snap(tictoc, meta, comm, snap, density_rank, prog_reals, out_hdf):

    # Open this snapshots file
    if density_rank == 0:
        hdf = h5py.File(meta.dgraphpath + meta.graph_basename
                        + snap + '.hdf5', 'r')
    else:
        hdf = h5py.File(meta.dgraphpath + "Sub_" + meta.graph_basename
                        + snap + '.hdf5', 'r')

    # Open necessary datasets
    nprogs = hdf["n_progs"][...]
    ndescs = hdf["n_descs"][...]
    nparts = hdf["n_parts"][...]
    masses = hdf["halo_mass"][...]
    prog_start_index = hdf["prog_start_index"][...]
    desc_start_index = hdf["desc_start_index"][...]
    prog_haloids = hdf["ProgHaloIDs"][...]
    desc_haloids = hdf["DescHaloIDs"][...]
    prog_cont = hdf["ProgNPartContribution"][...]
    desc_cont = hdf["DescNPartContribution"][...]
    prog_mass_cont = hdf["ProgMassContribution"][...]
    desc_mass_cont = hdf["DescMassContribution"][...]
    prog_npart = hdf["ProgNPart"][...]
    desc_npart = hdf["DescNPart"][...]
    prog_masses = hdf["ProgMass"][...]
    desc_masses = hdf["DescMass"][...]
    reals = hdf["real_flag"][...]

    hdf.close()

    # How many halos are there?
    nhalo = nprogs.size

    # Divide the work over our ranks
    rank_halo_bins = np.linspace(0, nhalo, meta.nranks + 1, dtype=int)

    # Handle the case where there are more ranks than halos
    # NOTE: the range(nhalo, 0) does not loop
    if meta.nranks <= nhalo:
        rank_halo_bins = [0, nhalo] + (meta.nranks - 1) * [0, ]

    # Set up dictionary to store the cleaned halos
    results = {}

    # Loop over halos designated for this rank
    for ihalo in range(rank_halo_bins[meta.rank],
                       rank_halo_bins[meta.rank + 1]):

        # Extract this halos values
        nprog = nprogs[ihalo]
        ndesc = ndescs[ihalo]
        npart = nparts[ihalo, :]
        real = reals[ihalo]
        mass = masses[ihalo, :]
        this_pstart = prog_start_index[ihalo]
        this_dstart = desc_start_index[ihalo]
        if nprog > 0:
            this_progs = prog_haloids[this_pstart: this_pstart + nprog]
            this_pcont = prog_cont[this_pstart: this_pstart + nprog, :]
            this_pmcont = prog_mass_cont[this_pstart: this_pstart + nprog, :]
            this_pnpart = prog_npart[this_pstart: this_pstart + nprog]
            this_pmass = prog_masses[this_pstart: this_pstart + nprog, :]
            this_preals = prog_reals[this_progs]
        else:
            this_progs = np.array([], dtype=int)
            this_pcont = np.empty((0, len(meta.npart)))
            this_pmcont = np.empty((0, len(meta.npart)))
            this_pnpart = np.array([], dtype=int)
            this_pmass = np.empty((0, len(meta.npart)))
            this_preals = np.array([], dtype=bool)
        if ndesc > 0:
            this_descs = desc_haloids[this_dstart: this_dstart + ndesc]
            this_dcont = desc_cont[this_dstart: this_dstart + ndesc]
            this_dmcont = desc_mass_cont[this_dstart: this_dstart + ndesc]
            this_dmass = prog_masses[this_pstart: this_pstart + nprog, :]
            this_dnpart = desc_npart[this_dstart: this_dstart + ndesc]
        else:
            this_descs = np.array([], dtype=int)
            this_dcont = np.empty((0, len(meta.npart)))
            this_dmcont = np.empty((0, len(meta.npart)))
            this_dnpart = np.array([], dtype=int)
            this_dmass = np.empty((0, len(meta.npart)))

        # Instantiate and store this halo
        results[ihalo] = Halo(npart, mass, real, this_progs, this_pnpart,
                              this_pcont, this_pmass, this_pmcont, this_preals,
                              this_descs, this_dnpart, this_dcont,
                              this_dmass, this_dmcont)

    # Collect our results
    tictoc.start_func_time("Collecting")
    all_results = comm.gather(results, root=0)
    tictoc.stop_func_time()

    if meta.verbose:
        tictoc.report("Collecting results")

    # Write out the results
    if meta.rank == 0:

        # Create group
        if density_rank == 0:
            grp = out_hdf.create_group(snap)
        else:
            snap_grp = out_hdf[snap]
            grp = snap_grp.create_group("Subhalos")

        # Write data
        reals = write_cleaned_dgraph_data(tictoc, meta, grp,
                                          all_results, density_rank)

        # Add the temporally informed real flags to halo catalog
        clean_real_flags(tictoc, meta, density_rank, reals, snap)

    else:
        reals = np.empty(nhalo, dtype=bool)

    # Communicate the real flags to all ranks
    reals = comm.bcast(reals, root=0)

    return reals


@timer("Cleaning")
def walk_and_purge(tictoc, meta, comm, snaplist):

    # Clean up basename if necessary
    basename = meta.graph_basename
    if basename[-1] == "_":
        basename = basename[:-1]

    # Set up output file
    if meta.rank == 0:
        out_hdf = h5py.File(meta.dgraphpath + basename + ".hdf5", "w")
    else:
        out_hdf = None

    # Set up variables we'll need for the looping
    prog_reals = np.array([], dtype=bool)
    sub_prog_reals = np.array([], dtype=bool)

    # Loop over snapshot list (beginning of time till present day)
    for snap in snaplist:

        # Clean up host halos in this snap
        prog_reals = clean_snap(tictoc, meta, comm, snap, 0,
                                prog_reals, out_hdf)

        # Clean up subhalos in this snap
        if meta.findsubs:
            sub_prog_reals = clean_snap(tictoc, meta, comm, snap, 1,
                                        sub_prog_reals, out_hdf)
    if meta.rank == 0:
        out_hdf.close()
