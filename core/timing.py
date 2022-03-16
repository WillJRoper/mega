import time


def timer(process=None):
    def decorator(func):
        def wrapper(*args, **kwargs):

            if process not in args[0].task_time:
                args[0].task_time[process] = {"Start": [], "End": []}

            # Get tic
            args[0].get_tic()

            # If we are timing a task lets write the start
            if process is not None:
                args[0].task_time[process]["Start"].append(args[0].tic)

            # Do the stuff
            data = func(*args, **kwargs)

            # Get toc
            args[0].get_toc()

            # If we are timing a task lets write the end
            if process is not None:
                args[0].task_time[process]["End"].append(args[0].toc)

            return data
        return wrapper
    return decorator


class TicToc:

    def __init__(self):

        # Basic times
        self.tic = 0
        self.toc = 0

        # Process timing
        self.task_time = {}
        self.task_time["START"] = self.get_tic()
        self.task_time["Reading"] = {"Start": [], "End": []}
        self.task_time["Domain-Decomp"] = {"Start": [], "End": []}
        self.task_time["Communication"] = {"Start": [], "End": []}
        self.task_time["Housekeeping"] = {"Start": [], "End": []}
        self.task_time["Task-Munging"] = {"Start": [], "End": []}
        self.task_time["Host-Spatial"] = {"Start": [], "End": []}
        self.task_time["Host-Phase"] = {"Start": [], "End": []}
        self.task_time["Sub-Spatial"] = {"Start": [], "End": []}
        self.task_time["Sub-Phase"] = {"Start": [], "End": []}
        self.task_time["Assigning"] = {"Start": [], "End": []}
        self.task_time["Collecting"] = {"Start": [], "End": []}
        self.task_time["Writing"] = {"Start": [], "End": []}
        self.task_time["END"] = None

    def get_tic(self):
        self.tic = time.time()
        return self.tic

    def get_toc(self):
        self.toc = time.time()
        return self.toc

    def how_long(self):
        return self.toc - self.tic

    def report(self, process):

        # How long did we take?
        how_long = self.how_long()

        # Print in milliseconds for short periods, seconds otherwise
        if how_long < 0.01:
            print("%s took: %.2f ms" % (process, how_long * 10**-3))
        else:
            print("%s took: %.2f secs" % (process, how_long))

    # TODO: function to report percentage of time spent doing each task


