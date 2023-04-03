from flask import Flask, jsonify, request, Response
import queue
import logging

import numpy as np
import jax.numpy as jnp

import os

import requests
from urllib.parse import urljoin
import threading
from functools import partial

from vivarium.simulator.simulator import SimulatorConfig, Simulator, EntityType, sim_state_to_populations

main_thread_queue = queue.Queue()
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# sim_server_url = 'http://127.0.0.1:5000' #'http://localhost:8086'
#
# pop_config = {EntityType.PREY: 20, EntityType.PREDATOR: 2, EntityType.OBSTACLE: 2}
#
# sim_config = SimulatorConfig()
#
# behavior_set = [[[1., 0., 0.],  # Fear
#                  [0., 1., 0.]],
#
#                 [[0., 1., 0.],  # Aggression
#                  [1., 0., 0.]],
#
#                 [[-1., -0., 1.],  # Love
#                  [0., -1., 1.]],
#
#                 [[0., -1., 1.],  # Shy
#                  [-1., -0., 1.]]]
#
# behavior_set = jnp.array(behavior_set)
#
# behavior_map = {EntityType.PREY.name: jnp.array([1., 0., 0., 0.]), EntityType.PREDATOR.name: jnp.array([0., 1, 0., 0.]), EntityType.OBSTACLE.name: jnp.array([0., 0, 0., 0.])}
#
# beh_config = behavior_set, behavior_map
#
#
# simulator = Simulator(sim_config, pop_config, beh_config, proxs_dist_max=sim_config.box_size, proxs_cos_min=0., to_jit=True)
#
#
#
# main_thread_queue = queue.Queue()
#

class RestApi:
    def __init__(self, simulator):
        self.simulator = simulator
        #self.app = Flask(__name__)

    def start_api(self):
        threading.Thread(target=app.run).start()
    #@tranquilize()
    @app.route("/simulator/get_sim_config", methods=["GET"])
    def get_sim_config(self):

        return self.simulator.sim_config.get_sim_config()

    #@tranquilize(method='post')
    @app.route("/simulator/get_state", methods=["POST"])
    def get_state(self):
        return serialize_state(sim_state_to_populations(self.simulator.state, self.simulator.entity_slices))

    @app.route("/simulator/start", methods=["GET"])
    def open_session(self):
        print('s/o')
        # ask main thread to start simulator
        #main_thread_queue.put(lambda: simulator.run())
        self.simulator.run(threaded=True)
        return Response(status=200)

    @app.route("/simulator/stop", methods=["GET"])
    def close_session(self):
        self.simulator.stop()
        return Response(status=200)

    @app.route("/simulator/is_started", methods=["GET"])
    def is_started(self):
        return {'is_started': self.simulator.is_started}

# #
# class SimulatorServer:
#     def __init__(self, sim_server_url=sim_server_url):
#         self.server_url = sim_server_url
#         self.prefix = 'simulator'
#     def get_sim_config(self):
#         sim_config = requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'get_sim_config')))
#         return sim_config.json()
#
#     def get_sim_state(self):
#         state = requests.post(urljoin(self.server_url, os.path.join(self.prefix, 'get_state')))
#         return state.json()
#     # def run(self):
#     #     requests.get(urljoin(self.server_url, 'run'))
#     def start(self):
#         requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'start')))
#
#     def stop(self):
#         requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'stop')))
#
#     def is_started(self):
#         #print(urljoin(self.server_url, os.path.join(self.prefix, 'is_started')))
#         req = requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'is_started')))
#         return req.json()['is_started']
#
#     def set_motors(self, agent_idx, motors):
#         args = {'agent_idx': agent_idx, 'motors': motors}
#         requests.post(urljoin(self.server_url, 'set_motors'), data=args)
#
#     def no_set_motors(self):
#         requests.get(urljoin(self.server_url, 'no_set_motors'))
#     def get_motors(self):
#         req = requests.post(urljoin(self.server_url, 'get_motors'))
#         return req.json()




from dataclasses import asdict
#

if __name__ == "__main__":
    print('START FLASK')
    # launch flask in a separated thread
    #simulator.run(threaded=True)
    app.run()
    # threading.Thread(target=app.run).start()
    #
    # # main thread waits for execute process
    # # especially for start the sgp simulator
    # while True:
    #     callback = main_thread_queue.get()  # blocks until an item is available
    #     print(callback)
    #     callback()


#
#
#
# #@tranquilize(method='post')
# @app.route("/set_motors", methods=["POST"])
# def set_motors(agent_idx: int, motors: List[float]):
#     global motor_input
#     res = np.zeros((pop_config[EntityType.PREY], map_dim))
#     #print(res[agent_idx, :])
#     res[agent_idx, :] = np.array(motors)
#     motor_input = jnp.array(res)
#     return Response(status=200)
#
# #@tranquilize()
# @app.route("/no_set_motors", methods=["GET"])
# def no_set_motors():
#     global motor_input
#     motor_input = None
#     return Response(status=200)
#
# #@tranquilize(method='post')
# @app.route("/get_motors", methods=["POST"])
# def get_motors():
#     global motor_input
#     return np.array(motor_input).tolist()
#
#
#

#
# global sim_start
# sim_start = False
#
# #@tranquilize()
# @app.route("/start", methods=["GET"])
#
# def start():
#     global sim_start
#     sim_start = True
#     return Response(status=200)
#
# #@tranquilize()
# @app.route("/stop", methods=["GET"])
# def stop():
#     global sim_start
#     sim_start = False
#     return Response(status=200)
#
# #@tranquilize()
# @app.route("/is_started", methods=["GET"])
# def is_started():
#     global sim_start
#     return {'is_started': sim_start}
#
# #@tranquilize()
# @app.route("/run", methods=["GET"])
#
#
