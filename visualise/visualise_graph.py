import numpy as np
import h5py
import sys
sys.path.insert(1, "/Users/willroper/Documents/University/Merger_Trees_to_Merger_Graphs/mega/core")
import matplotlib.pyplot as plt
import utilities
import networkx as nx


# Read the parameter file
paramfile = sys.argv[1]
inputs, flags, params = utilities.read_param(paramfile)

# Open Graph file
hdf = h5py.File(inputs['graphSavePath'] + ".hdf5", "r")

if sys.argv[2] == "All":
    graphs = list(hdf.keys())
else:
    graphs = [sys.argv[2], ]

for g in graphs:

    print("Visualising graph", g)

    posz = {}
    pos = {}
    edges = set()
    sizes = []
    nodes = []

    halo_ids = hdf[g]["graph_halo_ids"][...]
    nprogs = hdf[g]["nprog"][...]
    ndescs = hdf[g]["ndesc"][...]
    prog_sind = hdf[g]["prog_start_index"][...]
    desc_sind = hdf[g]["desc_start_index"][...]
    progs = hdf[g]["direct_prog_ids"][...]
    descs = hdf[g]["direct_desc_ids"][...]
    nparts = hdf[g]["nparts"][...]
    snapshots = hdf[g]["snapshots"][...]
    mean_pos = hdf[g]['mean_pos'][...]
    zs = hdf[g]["redshifts"][...]
    pmass = hdf["Header"].attrs["part_mass"]

    prev_snap = snapshots[0]
    snap_count = -1

    if np.max(nparts) < 10000:
        continue

    for i, nprog, ndesc, pstrt, dstrt, npart, z, snap, mp in zip(halo_ids, nprogs, ndescs, prog_sind,
                                                            desc_sind, nparts, zs, snapshots, mean_pos):

        this_progs = utilities.get_linked_halo_data(progs, pstrt, nprog)
        this_descs = utilities.get_linked_halo_data(descs, dstrt, ndesc)

        edges.update({(i, p) for p in this_progs})
        edges.update({(d, i) for d in this_descs})


        # if snap == prev_snap:
        #     if i % 2 == 0:
        #         snap_count = np.abs(snap_count) + 10
        #     else:
        #         snap_count = - (np.abs(snap_count) + 10)
        # else:
        #     snap_count = 0

        # pos[i] = (snap_count, int(snap))
        pos[i] = (mp[0], int(snap))
        posz[i] = (mp[0], z)

        nodes.append(i)

        # Append node size
        sizes.append(np.log10(npart * pmass))

        prev_snap = snap

    nodes = np.array(nodes)
    sizes = np.array(sizes)

    sinds = np.argsort(nodes)[::-1]
    nodes = nodes[sinds]
    sizes = sizes[sinds]

    # Create the graph and draw it with the node labels
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(list(edges))

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    im = nx.draw_networkx_nodes(G, pos, node_size=10, cmap=plt.get_cmap('plasma'), ax=ax, node_color=sizes)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='k', arrows=True, ax=ax)

    cbar = fig.colorbar(im)

    ax.set_xlabel("$x/[\mathrm{cMpc}]$")
    ax.set_ylabel("$S$")
    cbar.set_label("$\log_{10}(M)$")

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.savefig("visualise/Graphs/Graph_" + g + "snap.png", bbox_inches="tight")

    plt.close()

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    im = nx.draw_networkx_nodes(G, posz, node_size=10, cmap=plt.get_cmap('plasma'), ax=ax, node_color=sizes)
    nx.draw_networkx_edges(G, posz, edgelist=edges, edge_color='k', arrows=True, ax=ax)

    cbar = fig.colorbar(im)

    ax.set_xlabel("$x/[\mathrm{cMpc}]$")
    ax.set_ylabel("$z$")
    cbar.set_label("$\log_{10}(M)$")

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.savefig("visualise/Graphs/Graph_" + g + "z.png", bbox_inches="tight")

    plt.close()
