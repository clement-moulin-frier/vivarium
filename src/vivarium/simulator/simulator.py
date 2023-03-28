# from jax import jit, vmap
# import jax.numpy as jnp
# import numpy as np
# from collections import namedtuple
# from tranquilizer import tranquilize
#
# global x
#
# n = 10
#
# x = jnp.zeros(10)
#
# def compute(i):
#     j = i + 1
#
#     return jnp.array(j)
#
# compute = vmap(compute)
#
# compute = jit(compute)
#
# @tranquilize()
# def publish():
#     global x
#
#     y = compute(x)
#
#     #x = x.at[0].set(y + 1)
#
#     return {'arr': np.array(y).tolist()}

import numpy as np

# from jax.config import config ; config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax
# from jax import random
from jax import jit
from jax import vmap
from jax import lax
from jax import ops

import typing
#vectorize = np.vectorize

from functools import partial
from collections import namedtuple
#import base64

import os

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
# from jax_md.util import f32


# normalize = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)

from enum import Enum
from dataclasses import dataclass

from tranquilizer import tranquilize


from flask import Flask, jsonify, request, Response
import queue
import logging

main_thread_queue = queue.Queue()
app = Flask(__name__)
app.logger.setLevel(logging.INFO)




Population = namedtuple('Population', ['positions', 'thetas', 'entity_type'])

PopulationObstacle = namedtuple('Population', ['positions', 'thetas', 'entity_type', 'diameters'])

EntityType = Enum('EntityType', ['PREY', 'PREDATOR', 'OBSTACLE'])



@dataclass()
class SimulatorConfig(box_size = 100.0, map_dim = 2, wheel_diameter = 2., base_lenght = 10.,
                      key = jax.random.PRNGKey(0), speed_mul = 0.1, theta_mul = 0.1,
                      num_steps_lax = 50, num_lax_loops = 1):

    displacement, shift = space.periodic(box_size)
    neighbor_radius = box_size
    neighbor_fn = partition.neighbor_list(displacement,
                                          box_size,
                                          r_cutoff=neighbor_radius,
                                          dr_threshold=10.,
                                          capacity_multiplier=1.5,
                                          format=partition.Sparse)
    def generate_entities(self, dict_type_to_count):
      pops = {}
      for entity_type, count in dict_type_to_count.items():
        self.key, subkey = jax.random.split(self.key)
        positions = self.box_size * jax.random.uniform(subkey, (count, map_dim))
        key, subkey = jax.random.split(key)
        thetas = jax.random.uniform(subkey, (count,), maxval=2 * np.pi)
        pops[entity_type.name] = Population(positions, thetas, jnp.array(entity_type.value))
        if entity_type == EntityType.OBSTACLE:
            diameters = 10. * jnp.ones((count,))
            pops[entity_type.name] = PopulationObstacle(positions, thetas, jnp.array(entity_type.value), diameters)
        else:
            pops[entity_type.name] = Population(positions, thetas, jnp.array(entity_type.value))
      return pops

    def neighbors(self, populations, entity_types: typing.List[EntityType]):
        assert len(entity_types) == 1, "Only did with a singly entity type for now"

        e = entity_types[0]
        return self.neighbor_fn.allocate(populations[e.name].positions)


pop_config = {EntityType.PREY: 20, EntityType.PREDATOR: 2, EntityType.OBSTACLE: 2}

sim_config = SimulatorConfig()

populations = sim_config.generate_entities(pop_config)
# global neighbors
neighbors = sim_config.neighbors(populations, EntityType.PREY)


global motor_input
motor_input = None

from typing import List

#@tranquilize(method='post')
@app.route("/set_motors", methods=["POST"])
def set_motors(agent_idx: int, motors: List[float]):
    global motor_input
    res = np.zeros((pop_config[EntityType.PREY], map_dim))
    #print(res[agent_idx, :])
    res[agent_idx, :] = np.array(motors)
    motor_input = jnp.array(res)
    return Response(status=200)

#@tranquilize()
@app.route("/no_set_motors", methods=["GET"])
def no_set_motors():
    global motor_input
    motor_input = None
    return Response(status=200)

#@tranquilize(method='post')
@app.route("/get_motors", methods=["POST"])
def get_motors():
    global motor_input
    return np.array(motor_input).tolist()

def behavior(proxs, external_motors):
    if external_motors is not None:
        motors = external_motors
        #print(motors)
    else:
        motors = cross(proxs) # Braitenberg simple
    fwd, rot = vmap(motor_command)(motors)
    return fwd, rot


global state
state = populations

def serialize_state(s):
    serial_s = {}

    for type, pop in s.items():
        serial_pop_kwargs = {}
        for field, jarray in pop._asdict().items():
            serial_pop_kwargs[field] = np.array(jarray).tolist()
        # if isinstance(s[type], Population):
        #     serial_pop = Population(**serial_pop_kwargs)
        # elif isinstance(s[type], PopulationObstacle):
        #     serial_pop = PopulationObstacle(**serial_pop_kwargs)
        serial_s[type] = serial_pop_kwargs
    return serial_s

#@tranquilize(method='post')
@app.route("/get_state", methods=["POST"])

def get_state():
    global state
    print('state')
    return serialize_state(state)

global sim_start
sim_start = False

#@tranquilize()
@app.route("/start", methods=["GET"])

def start():
    global sim_start
    sim_start = True
    return Response(status=200)

#@tranquilize()
@app.route("/stop", methods=["GET"])
def stop():
    global sim_start
    sim_start = False
    return Response(status=200)

#@tranquilize()
@app.route("/is_started", methods=["GET"])
def is_started():
    global sim_start
    return {'is_started': sim_start}

#@tranquilize()
@app.route("/run", methods=["GET"])



main_thread_queue = queue.Queue()

@app.route("/start_sim", methods=["GET"])
def start_sim():
    main_thread_queue.put(lambda: run())
    return Response(status=200)

import threading


if __name__ == "__main__":

    # launch flask in a separated thread
    threading.Thread(target=app.run).start()

    # main thread waits for execute process
    # especially for start the sgp simulator
    while True:
        callback = main_thread_queue.get()  # blocks until an item is available
        callback()