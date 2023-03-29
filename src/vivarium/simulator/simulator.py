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


from vivarium.simulator.sim_computation import dynamics, Population, PopulationObstacle

import os

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
# from jax_md.util import f32


# normalize = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)

from enum import Enum
from dataclasses import dataclass, asdict
import time

from tranquilizer import tranquilize





EntityType = Enum('EntityType', ['PREY', 'PREDATOR', 'OBSTACLE'])



@dataclass()
class SimulatorConfig():

    box_size = 100.0
    map_dim = 2
    wheel_diameter = 2.
    base_length = 10.
    key = jax.random.PRNGKey(0)
    speed_mul = 0.1
    theta_mul = 0.1


    def get_sim_config(self):

        conf = {'box_size': self.box_size,
                'map_dim': self.map_dim,
                'wheel_diameter': self.wheel_diameter,
                'base_length': self.base_length,
                'speed_mul': self.speed_mul,
                'theta_mul': self.theta_mul
                }
        return conf

    def generate_entities(self, dict_type_to_count):
      pops = {}
      for entity_type, count in dict_type_to_count.items():
        self.key, subkey = jax.random.split(self.key)
        positions = self.box_size * jax.random.uniform(subkey, (count, self.map_dim))
        self.key, subkey = jax.random.split(self.key)
        thetas = jax.random.uniform(subkey, (count,), maxval=2 * np.pi)
        pops[entity_type.name] = Population(positions, thetas, jnp.array(entity_type.value))
        if entity_type == EntityType.OBSTACLE:
            diameters = 10. * jnp.ones((count,))
            pops[entity_type.name] = PopulationObstacle(positions, thetas, jnp.array(entity_type.value), diameters)
        else:
            pops[entity_type.name] = Population(positions, thetas, jnp.array(entity_type.value))
      return pops





class Simulator():
    def __init__(self, sim_config, pop_config, proxs_dist_max, proxs_cos_min, num_steps_lax = 50, num_lax_loops = 1, freq=100., to_jit=True):

        self.proxs_dist_max = proxs_dist_max
        self.proxs_cos_min = proxs_cos_min
        self.populations = sim_config.generate_entities(pop_config)

        self.displacement, self.shift = space.periodic(sim_config.box_size)
        neighbor_radius = sim_config.box_size
        self.neighbor_fn = partition.neighbor_list(self.displacement,
                                                   sim_config.box_size,
                                                   r_cutoff=neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   format=partition.Sparse)
        # global neighbors
        self.neighbors = self.neighbor_fn.allocate(self.populations[EntityType.PREY.name].positions)

        self.update_fn = dynamics(self.shift, self.displacement, sim_config.map_dim,
                                  sim_config.base_length, sim_config.wheel_diameter,
                                  self.proxs_dist_max, self.proxs_cos_min,
                                  sim_config.speed_mul, sim_config.theta_mul, 1e-1)
        self.to_jit = to_jit
        if self.to_jit:
            self.update_fn = jit(self.update_fn)

        self.num_steps_lax = num_steps_lax
        self.num_lax_loops = num_lax_loops
        self.boids_buffer = []

        self.state = self.populations['PREY']

        self.is_started = False

        self.freq = freq

        # self_is_running = False
    def run(self):

        self.is_started = True
        print('Run starts')
        while True:

            time.sleep(1. / self.freq)

            if not self.is_started:
                break
            if self.to_jit:
                new_state, neighbors = lax.fori_loop(0, self.num_steps_lax, self.update_fn,
                                                    (self.state, self.neighbors))
            else:
                assert False, "not good, modifies self.state"
                val = (self.state, self.neighbors)
                for i in range(0, self.num_steps_lax):
                    val = self.update_fn(i, val)
                new_state, neighbors = val

            # If the neighbor list can't fit in the allocation, rebuild it but bigger.
            if neighbors.did_buffer_overflow:
                print('REBUILDING')
                neighbors = self.neighbor_fn.allocate(self.state.positions)
                new_state, neighbors = lax.fori_loop(0, self.num_lax_loops, self.update_fn, (self.state, neighbors))
                assert not neighbors.did_buffer_overflow

            self.state = new_state
            # print(self.state.positions[0, :])
            self.boids_buffer += [self.state]

        print('Run stops')

    def stop(self):
        self.is_started = False


if __name__ == "__main__":
    pop_config = {EntityType.PREY: 20, EntityType.PREDATOR: 2, EntityType.OBSTACLE: 2}

    sim_config = SimulatorConfig()

    simulator = Simulator(sim_config, pop_config, proxs_dist_max=sim_config.box_size, proxs_cos_min=0., to_jit=True)

    simulator.run()
