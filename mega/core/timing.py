import numpy as np
import time

from mega.core.talking_utils import message, get_heading


def timer(process=None):
    def decorator(func):
        def wrapper(*args, **kwargs):

            # Lets time this process
            args[0].start_func_time(process)

            # Do the stuff
            data = func(*args, **kwargs)

            # Lets stop timing this process
            args[0].stop_func_time()

            return data
        return wrapper
    return decorator


class TicToc:

    def __init__(self, meta):

        # Store the meta data object
        self.meta = meta

        # Basic times
        self.tic = 0
        self.toc = 0
        self.runtime = 0

        # Process timing dictionary
        self.processes = ("Housekeeping", "Domain-Decomp", "Reading",
                          "Tree-Building", "Host-Spatial", "Hydro-DM-CrossRef",
                          "PartID-Decomp", "Linking", "Progenitor-Linking",
                          "Descendant-Linking", "Cleaning", "Stitching",
                          "Kinetic-Energy", "Grav-Energy", "Create Halo",
                          "Create Subhalo", "Compute-Props", "Host-Phase",
                          "Sub-Spatial",  "Sub-Phase",
                          "Collecting", "Writing")
        self.task_time = {k: {"Start": [], "End": []} for k in self.processes}
        self.task_time["START"] = None
        self.task_time["END"] = None

        # Variables to handle nested tasks
        self.current_tasks = []

    def get_tic(self):
        self.tic = time.perf_counter()
        return self.tic

    def get_toc(self):
        self.toc = time.perf_counter()
        return self.toc

    @staticmethod
    def get_extic():
        return time.perf_counter()

    @staticmethod
    def get_extoc():
        return time.perf_counter()

    def how_long(self):
        return self.toc - self.tic

    def start(self):
        self.task_time["START"] = time.perf_counter()

    def end(self):
        self.task_time["END"] = time.perf_counter()
        self.get_runtime()

        # Convert task time lists to arrays
        for k in self.task_time:
            if k in ["START", "END"]:
                continue
            self.task_time[k]["Start"] = np.array(self.task_time[k]["Start"])
            self.task_time[k]["End"] = np.array(self.task_time[k]["End"])

    def get_runtime(self):
        self.runtime = self.task_time["END"] - self.task_time["START"]

    def start_func_time(self, process):

        # Start timing
        self.get_tic()

        # Record that we are doing something
        self.current_tasks.append(process)

        # Are we already doing something?
        if len(self.current_tasks) > 1:

            # Temporarily end the previous task timing
            self.task_time[self.current_tasks[-2]]["End"].append(self.tic)

            # Start timing
            self.get_tic()

        # Start timing this task
        self.task_time[self.current_tasks[-1]]["Start"].append(self.tic)

    def stop_func_time(self):

        # Stop timing
        self.get_toc()

        # We have finished the current task so remove it
        finished = self.current_tasks.pop(-1)
        self.task_time[finished]["End"].append(self.toc)

        # Check if we are inside another process
        if len(self.current_tasks) > 0:

            # Restart the old task
            self.get_tic()

            self.task_time[self.current_tasks[-1]]["Start"].append(self.tic)

    def record_time(self, process):

        self.task_time.setdefault(process, {})
        self.task_time[process].setdefault("Start", []).append(self.tic)
        self.task_time[process].setdefault("End", []).append(self.toc)

    def report(self, process, start=None, end=None):

        # Are we using internal timers?
        if start is None:
            # How long did we take?
            how_long = self.how_long()
        else:
            # Use what has been handed to us
            if end is None:
                # How long did we take?
                how_long = self.toc - start
            else:
                # How long did we take?
                how_long = end - start

        # Print in us/milliseconds for short periods, seconds otherwise
        if how_long < 1 * 10**-4:
            message(self.meta.rank, "%s took: %.2f us" % (process,
                                                          how_long * 10**6))
        elif how_long < 1 * 10**-2:
            message(self.meta.rank, "%s took: %.2f ms" % (process,
                                                          how_long * 10**3))
        else:
            message(self.meta.rank, "%s took: %.2f secs" % (process, how_long))

    def end_report(self, comm):

        # Delete unused entries
        for k in self.processes:
            if len(self.task_time[k]["Start"]) == 0:
                del self.task_time[k]

        # Lets collect everyone's timings
        all_timings = comm.gather((self.meta.rank, self.task_time), root=0)

        # we only want to print the table from the master
        if self.meta.rank == 0:

            # Get the total runtime from master
            master_total = self.runtime

            # We need a dictionary with each ranks timings
            all_task_time = {r: {} for r in range(self.meta.nranks)}

            # We need to collect all tasks that were done on all ranks
            # (no guarantee all ranks did the same tasks)
            all_keys = []
            for r, d in all_timings:
                all_task_time[r]["Dead Time"] = master_total
                for k in d.keys():
                    if k in ["START", "END"]:
                        all_task_time[r]["Total"] = d["END"] - d["START"]
                        continue

                    # Calculate the time spent doing this job
                    all_task_time[r][k] = np.sum(d[k]["End"] - d[k]["Start"])

                    # Subtract this time from the dead time
                    all_task_time[r]["Dead Time"] -= all_task_time[r][k]
                    if k not in all_keys:
                        all_keys.append(k)

            # Include the total and dead time in the keys
            all_keys.extend(["Dead Time", "Total"])

            # Define column width
            ncols = 6
            col_width = int(np.ceil(self.meta.table_width / ncols))
            self.meta.table_width = col_width * ncols

            # Get and print the heading
            heading = get_heading(self.meta.table_width, "Task Timings")
            message(self.meta.rank, heading)

            # Split the columns up into rows of ncols columns
            row = 1
            table_headings = {}
            table_keys = {}
            ini_col = "Rank" + "".ljust(col_width - len("Rank") - 1)
            for k in all_keys:
                s = k + "".ljust(col_width - len(k) - 1)
                table_headings.setdefault(row, [ini_col, ]).append(s)
                table_keys.setdefault(row, []).append(k)
                if len(table_headings[row]) % ncols == 0:
                    row += 1

            for tab_row in table_headings:

                # Lets recalculate col width for the number of columns in this
                # table row
                ncols = len(table_headings[tab_row])

                # Print this rows header
                message(self.meta.rank, "|".join(table_headings[tab_row]))

                message(self.meta.rank, ("-" * (col_width - 1) + "+")
                        * (ncols - 1) + "-" * col_width)

                # Loop over ranks and print table
                for r in range(self.meta.nranks):
                    rank_row = ["".ljust(col_width - len(str(r)) - 1) +
                                "%d" % r, ]
                    for k in table_keys[tab_row]:

                        if k in all_task_time[r].keys():
                            spent = all_task_time[r][k]
                            pcent = spent / master_total * 100
                        else:
                            spent = 0.0
                            pcent = 0.0
                        if spent < 1:
                            s = "%.2f ms / %.2f" % (spent *
                                                    10**3, pcent) + " %"
                            rank_row.append(
                                "".ljust(col_width - len(s) - 1) + s)
                        else:
                            s = "%.2f s / %.2f" % (spent, pcent) + " %"
                            rank_row.append(
                                "".ljust(col_width - len(s) - 1) + s)

                    message(self.meta.rank, "|".join(rank_row))

                # Print the footer
                message(self.meta.rank, ("=" * col_width) * ncols)
