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
            pid = line.split()[1]
            lg.info(f"{process_name} PID: {pid.decode()}")
            return pid.decode()

def kill_process(pid):
    """Kill a process by its ID

    :param pid: process ID
    """
    os.kill(int(pid), signal.SIGTERM)

# Define parameters of the simulator
def start_server_and_interface(scene_name: str, notebook_mode: bool = True):
    """Start the server and interface for the given scene

    :param scene_name: scene name
    :param notebook_mode: notebook_mode to adapt the interface, defaults to True
    """
    # first ensure no interface or server is running
    stop_server_and_interface()
    project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    server_script = os.path.join(project_root, SERVER_PROCESS_NAME)
    interface_script = os.path.join(project_root, INTERFACE_PROCESS_NAME)

    server_command = [
        "python3", 
        server_script, 
        f"scene={scene_name}" 
    ]

    def start_server_process():
        subprocess.run(server_command)

    print("STARTING SERVER")
    server_process = multiprocessing.Process(target=start_server_process)
    server_process.start()
    time.sleep(5)

    interface_command = [
        "panel", 
        "serve", 
        interface_script, 
        "--args", 
        f"--notebook_mode={str(notebook_mode)}",
    ]

    def start_interface_process():
        subprocess.run(interface_command)

    time.sleep(2)
    # start the interface 
    print("\nSTARTING INTERFACE")
    interface_process = multiprocessing.Process(target=start_interface_process)
    interface_process.start()

def stop_server_and_interface():
    """Stop the server and interface
    """
    stopped = False
    # interface_process_name = "vivarium/scripts/run_interface.py"
    interface_pid = get_process_pid(INTERFACE_PROCESS_NAME)
    if interface_pid is not None:
        kill_process(interface_pid)
        stopped = True
    else: 
        lg.info("Interface process not found")

    # server_process_name = "vivarium/scripts/run_server.py"
    server_pid = get_process_pid(SERVER_PROCESS_NAME)
    if server_pid is not None:
        kill_process(server_pid)
        stopped = True
    else:
        lg.info("Server process not found")

    if stopped:
        print("Server and Interface Stopped")