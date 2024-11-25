import logging
import threading

lg = logging.getLogger(__name__)


class Logger(object):
    def __init__(self):
        """Logger class that logs data for the agents
        """
        self.logs = {}

    def add(self, log_field, data):
        """Add data to the log_field of the logger

        :param log_field: log_field
        :param data: data
        """
        if log_field not in self.logs:
            self.logs[log_field] = [data]
        else:
            self.logs[log_field].append(data)

    def get_log(self, log_field):
        """Get the log of the logger for a specific log_field

        :param log_field: log_field
        :return: data associated with the log_field
        """
        if log_field not in self.logs:
            print("No data in " + log_field)
            return []
        else:
            return self.logs[log_field]
        
    def clear(self):
        """Clear all logs of the logger
        """
        del self.logs
        self.logs = {}


class RoutineHandler(object):
    """RoutineHandler class that handles routines for the NotebookController and its Entities
    """
    def __init__(self):
        """Initialize the RoutineHandler
        """
        self._routines = {}
        self._lock = threading.Lock()

    def attach_routine(self, routine_fn, name=None, interval=1):
        """Attach a routine to the entity

        :param routine_fn: routine_fn
        :param name: routine name, defaults to None
        :param interval: routine execution interval, defaults to 1
        """
        assert isinstance(interval, int) and interval > 0, "Interval must be a positive integer"
        with self._lock:
            self._routines[name or routine_fn.__name__] = (routine_fn, interval)

    def detach_routine(self, name):
        """Detach a routine from the entity

        :param name: routine name
        """
        with self._lock:
            del self._routines[name]

    def detach_all_routines(self):
        """Detach all routines from the entity
        """
        for name in list(self._routines.keys()):
            self.detach_routine(name)

    def routine_step(self, entity, time, catch_errors=True):
        """Execute the entity's routines with their corresponding execution intervals, and remove the ones that cause errors
        """
        to_remove = []
        with self._lock:
            # iterate over the routines and check if they work
            for name, (fn, interval) in self._routines.items():
                if time % interval == 0:
                    # if the catch_errors flag is set to True, catch the errors and remove the routine if it fails
                    if catch_errors:
                        try:
                            # execute the function on the entity object if the routine works
                            fn(entity)
                        except Exception as e:
                            # else plot an error message and remove the routine
                            lg.error(f"Error while executing routine: {e}, removing routine {name}")
                            to_remove.append(name)
                    # else just execute the routines normally
                    else:
                        fn(entity)
        
        # remove all problematic routines at the end to prevent spamming error messages and crashing the program
        if catch_errors:
            for name in to_remove:
                del self._routines[name]
    
    def print_routines(self):
        """Print the routines attached to the entity
        """
        with self._lock:
            print(f"Attached routines: {list(self._routines.keys())}")