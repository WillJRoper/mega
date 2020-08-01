import numpy as np
import sys
sys.path.insert(1, "/Users/willroper/Documents/University/Merger_Trees_to_Merger_Graphs/mega/core")
import utilities
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_style("whitegrid")


def my_autopct(pct):
    return ('%.2f' % pct + "%") if pct >= 10 else ''


# Read the parameter file
paramfile = sys.argv[1]
inputs, flags, params = utilities.read_param(paramfile)

snap_ind = int(sys.argv[2])

# Load the snapshot list
snaplist = list(np.loadtxt(inputs['snapList'], dtype=str))

snap = snaplist[snap_ind]

nranks = int(sys.argv[3])

# Initialise time taken dictionary
portion_of_time = {}
rank_time = {}
master_time = 0

# Color dict 
cols = {"Writing": "red", "Collecting": "gold", "Domain-Decomp": "yellowgreen", "Reading": "darkorchid",
        "Assigning": "darkgreen", "Worker Idle": 'lightskyblue', "Master Idle": 'violet', "Housekeeping": "aquamarine",
        "Task-Munging": "darkgoldenrod", "Host-Spatial": "firebrick", "Host-Phase": "lime", "Sub-Spatial": "cyan",
        "Sub-Phase": "darkmagenta"}

# Set up figure
fig = plt.figure()
ax = fig.add_subplot(111)

start_time = None
total_time = None
master_total = None

for rank in range(nranks):

    with open(inputs["profilingPath"] + "Halo_" + str(rank) + '_' + snap + '.pck', 'rb') as pfile:
        prof_dict = pickle.load(pfile)

    if rank == 0:
        start_time = prof_dict["START"]
        master_total = prof_dict["END"] - start_time

    rank_start_time = prof_dict["START"]
    rank_time[rank] = prof_dict["END"] - rank_start_time

    for task_type in prof_dict:

        if task_type in ["START", "END"]:
            continue

        starts = np.array(prof_dict[task_type]["Start"]) - start_time
        ends = np.array(prof_dict[task_type]["End"]) - start_time
        elapsed = ends - starts

        portion_of_time.setdefault(task_type, 0)
        portion_of_time[task_type] += np.sum(elapsed)

        plt_times = np.zeros((starts.size, 2))
        plt_times[:, 0] = starts
        plt_times[:, 1] = elapsed

        if rank == 0:
            master_time = np.sum(elapsed)

        ax.broken_barh(plt_times, (rank - 0.5, 1),
                       facecolor=cols[task_type], edgecolor='none', label=task_type)

# Ensure tick labels are integers
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Plot lines to denote each task
for rank in range(nranks + 1):
    ax.plot((0, master_total), (rank - 0.5, rank - 0.5), color='k', linewidth=0.1)

for spine in ax.spines.values():
        spine.set_edgecolor('k')
        spine.set_linewidth(1)

legend_elements = []
for key in portion_of_time:
    legend_elements.append(Patch(facecolor=cols[key], edgecolor=cols[key], label=key))

ax.set_xlabel("Time (Seconds)")
ax.set_ylabel("Ranks")

# Get and draw legend
ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1),
          fancybox=True, ncol=4)

ax.set_xlim(0, master_total)
ax.set_ylim(-0.51, nranks - 0.49)

fig.savefig(inputs["analyticPlotPath"] + "halo_task_time_series_" + str(snap) + ".png", bbox_inches="tight", dpi=300)

plt.close(fig)

total_time = 0
for rank in rank_time:
    total_time += rank_time[rank]

# Define labels and calculate data for pie chart
labels = []
pcent = []
explode = []
working_time = 0
for key in portion_of_time:
    labels.append(key)
    pcent.append(portion_of_time[key] / total_time * 100)
    working_time += portion_of_time[key]
    explode.append(0.0)

# Append down time
labels.append("Worker Idle")
pcent.append((total_time - working_time - (master_total - master_time)) / total_time * 100)
explode.append(0.0)
labels.append("Master Idle")
pcent.append((master_total - master_time) / total_time * 100)
explode.append(0.0)

# Sort tasks by contribution
sinds = np.argsort(pcent)[::-1]
labels = np.array(labels)[sinds]
explode = np.array(explode)[sinds]
pcent = np.array(pcent)[sinds]

print("=============== Runtime Breakdown ===============")
for pc, lab in zip(pcent, labels):
    print("{:<15}".format(lab), "{:<7}".format("= %.2f" % pc),  "%")

# Set up figure
fig = plt.figure()
ax = fig.add_subplot(111)

wedges, texts, autotexts = ax.pie(pcent, explode=explode, colors=[cols[lab] for lab in labels], shadow=True,
                                  startangle=90, autopct=my_autopct)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax.legend(wedges, [lab + " (%.2f" % pc + "%)" for pc, lab in zip(pcent, labels)],
          title="Tasks", loc="center left", bbox_to_anchor=(0.9, 0, 0.5, 1))

fig.savefig(inputs["analyticPlotPath"] + "halo_task_pie_chart_" + str(snap) + ".png", bbox_inches="tight")

plt.close(fig)

# Set up figure
fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(labels, pcent, color=[cols[lab] for lab in labels])

ax.set_ylabel("%")

ax.set_yscale("log")

for tick in ax.get_xticklabels():
    tick.set_rotation(90)

fig.savefig(inputs["analyticPlotPath"] + "halo_task_bar_chart_" + str(snap) + ".png", bbox_inches="tight")
