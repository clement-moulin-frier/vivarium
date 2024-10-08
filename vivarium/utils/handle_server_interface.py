import os
import time
import multiprocessing
import subprocess
import signal
import logging

lg = logging.getLogger(__name__)


# TODO : Later remove print statements and add logging instead
def get_process_pid(process_name: str):
    process = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
    out, err = process.communicate()
    for line in out.splitlines():
        if process_name.encode('utf-8') in line:
            pid = line.split()[1]
            lg.info(f"{process_name} PID: {pid.decode()}")
            return pid.decode()

def kill_process(pid):
    os.kill(int(pid), signal.SIGTERM)

# Define parameters of the simulator
def start_server_and_interface(scene_name: str):
    # first ensure no interface or server is running
    stop_server_and_interface()
    project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    server_script = os.path.join(project_root, "scripts/run_server.py")
    interface_script = os.path.join(project_root, "scripts/run_interface.py")

    def start_server_process():
        subprocess.run(["python3", server_script, f"scene={scene_name}"])

    print("STARTING SERVER")
    server_process = multiprocessing.Process(target=start_server_process)
    server_process.start()

    time.sleep(5)
    print("")

    def start_interface_process():
        subprocess.run(["panel", "serve", interface_script])

    # start the interface 
    print("STARTING INTERFACE")
    interface_process = multiprocessing.Process(target=start_interface_process)
    interface_process.start()

def stop_server_and_interface():
    stopped = False
    interface_process_name = "vivarium/scripts/run_interface.py"
    interface_pid = get_process_pid(interface_process_name)
    if interface_pid is not None:
        kill_process(interface_pid)
        stopped = True
    else: 
        lg.info("Interface process not found")

    server_process_name = "vivarium/scripts/run_server.py"
    server_pid = get_process_pid(server_process_name)
    if server_pid is not None:
        kill_process(server_pid)
        stopped = True
    else:
        lg.info("Server process not found")

    if stopped:
        print("Server and Interface Stopped")