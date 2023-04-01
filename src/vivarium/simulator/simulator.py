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


from vivarium.simulator.sim_computation import dynamics, Population, PopulationObstacle, agression, fear, noop

import os

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
# from jax_md.util import f32


# normalize = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)

from enum import Enum
from dataclasses import dataclass, asdict
import time
import queue
import threading

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


def populations_to_jax(populations, sim_config, pop_config):
    entity_slices = {}
    n_agents = 0
    for e_type, n in pop_config.items():
        entity_slices[e_type.name] = (n_agents, n_agents + n)
        n_agents += n

    all_positions = np.zeros((n_agents, sim_config.map_dim))
    all_thetas = np.zeros(n_agents)
    all_etypes = np.zeros(n_agents)

    for e_type, e_slice in entity_slices.items():
        all_positions[slice(*e_slice), :] = populations[e_type].positions
        all_thetas[slice(*e_slice)] = populations[e_type].thetas
        all_etypes[slice(*e_slice)] = populations[e_type].entity_type

    return entity_slices, jnp.array(all_positions), jnp.array(all_thetas), jnp.array(all_etypes)


def sim_state_to_populations(sim_state, entity_slices):
    pop_dict = {}
    for e_type, e_slice in entity_slices.items():
        pop_dict[e_type] = Population(sim_state.positions[slice(*e_slice), :], sim_state.thetas[slice(*e_slice)], sim_state.entity_type[slice(*e_slice)][0])

    return pop_dict


def behavior_map_to_jax(entity_slices, n_agents, behavior_map):
    res = None
    for e_type, e_slice in entity_slices.items():
        if res is None:
            res = np.zeros((n_agents, behavior_map[e_type].shape[0]))
        res[slice(*e_slice), :] = behavior_map[e_type]
    return res



class Simulator():
    def __init__(self, sim_config, pop_config, beh_config, proxs_dist_max, proxs_cos_min, num_steps_lax = 50, num_lax_loops = 1, freq=100., to_jit=True):

        self.sim_config = sim_config
        self.proxs_dist_max = proxs_dist_max
        self.proxs_cos_min = proxs_cos_min
        self.populations = sim_config.generate_entities(pop_config)

        self.entity_slices, all_positions, all_thetas, all_etypes = populations_to_jax(self.populations, sim_config, pop_config)

        n_agents = all_positions.shape[0]

        behavior_set, behavior_map = beh_config

        behavior_map = behavior_map_to_jax(self.entity_slices, n_agents, behavior_map)

        self.beh_config = behavior_set, behavior_map

        self.state = Population(positions=all_positions, thetas= all_thetas, entity_type=all_etypes)

        self.displacement, self.shift = space.periodic(sim_config.box_size)
        neighbor_radius = sim_config.box_size
        self.neighbor_fn = partition.neighbor_list(self.displacement,
                                                   sim_config.box_size,
                                                   r_cutoff=neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   format=partition.Sparse)
        # global neighbors
        self.neighbors = self.neighbor_fn.allocate(all_positions)

        self.update_fn = dynamics(sim_config, pop_config, self.beh_config, self.entity_slices, self.shift, self.displacement, sim_config.map_dim,
                                  sim_config.base_length, sim_config.wheel_diameter,
                                  self.proxs_dist_max, self.proxs_cos_min,
                                  sim_config.speed_mul, sim_config.theta_mul, 1e-1)

        self.beh_config = beh_config

        self.to_jit = to_jit
        if self.to_jit:
            self.update_fn = jit(self.update_fn)

        self.num_steps_lax = num_steps_lax
        self.num_lax_loops = num_lax_loops
        self.boids_buffer = []

        self.is_started = False

        self.freq = freq

        #self.main_thread_queue = queue.Queue()

        # self_is_running = False

    def run(self, threaded=False):
        if self.is_started:
            raise Exception("Simulator is already started")
        if threaded:
            threading.Thread(target=self._run).start()
            #self.main_thread_queue.put(lambda: self._run())
        else:
            return self._run()

    def _run(self):

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
                #assert False, "not good, modifies self.state"
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

    behavior_set = [[[1., 0., 0.],  # Fear
                     [0., 1., 0.]],

                    [[0., 1., 0.],  # Aggression
                     [1., 0., 0.]],

                    [[-1., -0., 1.],  # Love
                     [0., -1., 1.]],

                    [[0., -1., 1.],  # Shy
                     [-1., -0., 1.]]]

    behavior_set = jnp.array(behavior_set)

    behavior_map = {EntityType.PREY.name: jnp.array([0., 1, 0., 0.]), EntityType.PREDATOR.name: jnp.array([0., 1, 0., 0.]), EntityType.OBSTACLE.name: jnp.array([0., 0, 0., 0.])}


    beh_config = behavior_set, behavior_map

    sim_config = SimulatorConfig()

    print("WARNING: to_jit=False")
    simulator = Simulator(sim_config, pop_config, beh_config, proxs_dist_max=sim_config.box_size, proxs_cos_min=0., to_jit=False)

    simulator.run()
