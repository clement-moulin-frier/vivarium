import os
import time
import multiprocessing
import subprocess
import signal
import logging

lg = logging.getLogger(__name__)

SERVER_PROCESS_NAME = "scripts/run_server.py"
INTERFACE_PROCESS_NAME = "scripts/run_interface.py"

def get_process_pid(process_name: str):
    """Get the process ID of a running process by name

    :param process_name: process name
    :return: process ID
    """
    process = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
    out, err = process.communicate()
    for line in out.splitlines():
        if process_name.encode('utf-8') in line:
            pid_str = line.split()[1]
            pid = pid_str.decode()
            lg.warning(f" Found the process {process_name} running with this PID: {pid}")
            return pid

def kill_process(pid):
    """Kill a process by its ID

    :param pid: process ID
    """
    os.kill(int(pid), signal.SIGTERM)
    lg.warning(f"Killed process with PID: {pid}")

# Define parameters of the simulator
def start_server_and_interface(scene_name: str, notebook_mode: bool = True, wait_time: int = 6):
    """Start the server and interface for the given scene

    :param scene_name: scene name
    :param notebook_mode: notebook_mode to adapt the interface, defaults to True
    """
    # first ensure no interface or server is running
    processes_running = stop_server_and_interface(auto_kill=False)
    if processes_running:
        lg.warning("\nServer and Interface processes are still running, please stop them before starting new ones")
        lg.warning("ERROR: New processes will not be started")
        return

    # find the path to the server and interface scripts
    print(f"{os.path.dirname(__file__)}")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    server_script = os.path.join(project_root, SERVER_PROCESS_NAME)
    interface_script = os.path.join(project_root, INTERFACE_PROCESS_NAME)

    server_command = [
        "python3", 
        server_script, 
        f"scene={scene_name}" 
    ]

    def start_server_process():
        subprocess.run(server_command)

    print("\nSTARTING SERVER")
    server_process = multiprocessing.Process(target=start_server_process)
    server_process.start()
    time.sleep(wait_time)

    interface_command = [
        "panel", 
        "serve", 
        interface_script, 
        "--args", 
        f"--notebook_mode={str(notebook_mode)}",
    ]

    def start_interface_process():
        subprocess.run(interface_command)

    # start the interface 
    print("\nSTARTING INTERFACE")
    interface_process = multiprocessing.Process(target=start_interface_process)
    interface_process.start()

def terminate_process(pid):
    """Terminate the process if the PID is not None"""
    if pid is not None:
        kill_process(pid)

def stop_server_and_interface(auto_kill=False):
    """Stop the server and interface
    """
    processes_running = False
    interface_pid = get_process_pid(INTERFACE_PROCESS_NAME)
    server_pid = get_process_pid(SERVER_PROCESS_NAME)
    
    if interface_pid is not None or server_pid is not None:
        processes_running = True
        if auto_kill:
            terminate_process(interface_pid)
            terminate_process(server_pid)
            processes_running = False
        else:
            message = "\nThe following processes are running:\n"
            if interface_pid is not None:
                message += f" - Interface (PID: {interface_pid})\n"
            if server_pid is not None:
                message += f" - Server (PID: {server_pid})\n"
            message += "Do you want to stop them? (y/n): "
            user_input = input(message)

            if user_input.lower() == "y":
                terminate_process(interface_pid)
                terminate_process(server_pid)
                processes_running = False

    return processes_running

# TODO : older version of the function, see if we remove it
def stop_server_and_interface_unsafe():
    """Stop the server and interface processes if they are running"""
    interface_pid = get_process_pid(INTERFACE_PROCESS_NAME)
    if interface_pid is not None:
        kill_process(interface_pid)
        stopped = True
    else: 
        lg.info("Interface process not found")

    server_pid = get_process_pid(SERVER_PROCESS_NAME)
    if server_pid is not None:
        kill_process(server_pid)
        stopped = True
    else:
        lg.info("Server process not found")

    if stopped:
        print("\nServer and Interface Stopped")