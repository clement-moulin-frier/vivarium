import jax.numpy as jnp
from jax import jit
from jax import lax
import jax
import numpy as np
from jax_md import space, partition

import vivarium.simulator.behaviors as behaviors

from functools import partial

from vivarium.simulator.sim_computation import dynamics, Population

from vivarium.simulator import config

import time
import threading
import math
import param
import requests

def sim_state_to_populations(sim_state, entity_slices):
    pop_dict = {}
    for e_type, e_slice in entity_slices.items():
        pop_dict[e_type] = Population(sim_state.positions[slice(*e_slice), :], sim_state.thetas[slice(*e_slice)], sim_state.entity_type[slice(*e_slice)][0])

    return pop_dict

def generate_population(n_agents, box_size):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    positions = box_size * jax.random.uniform(subkey, (n_agents, 2))
    key, subkey = jax.random.split(key)
    thetas = jax.random.uniform(subkey, (n_agents,), maxval=2 * np.pi)
    proxs = jnp.zeros((n_agents, 2))
    motors = jnp.zeros((n_agents, 2))
    return Population(positions=positions, thetas=thetas, proxs=proxs, motors=motors, entity_type=0)


class Simulator():
    # simulation_config = param.ClassSelector(config.SimulatorConfig, instantiate=False)
    # agent_config = param.ClassSelector(config.AgentConfig, instantiate=False)
    # # behavior_config = param.ClassSelector(config.BehaviorConfig, instantiate=False)
    # # population_config = param.ClassSelector(config.PopulationConfig, instantiate=False)
    # engine_config = param.ClassSelector(config.EngineConfig, instantiate=False)
    # is_started = param.Boolean(False)
    # # engine_config = param.ClassSelector(config.EngineConfig)

    def __init__(self, simulation_config, agent_config, engine_config):
        # super().__init__(**params)
        self.simulation_config = simulation_config
        self.agent_config = agent_config
        self.engine_config = engine_config
        self.engine_config.behavior_name_map['manual'] = len(self.engine_config.behavior_bank) - 1
        self.is_started = False
        # self.simulation_config.param.watch_values(self._record_change, self.simulation_config.export_fields, queued=True)
        # self._recorded_change_dict = {}

    # def _record_change(self, **kwargs):
    #     self._recorded_change_dict.update(kwargs)
    #
    # def get_recorded_changes(self):
    #     d = dict(self._recorded_change_dict)
    #     self._recorded_change_dict = {}
    #     return d
    #
    # @param.depends('simulation_config.displacement', 'simulation_config.box_size', 'agent_config.neighbor_radius',
    #                watch=True, on_init=True)
    def update_neighbor_fn(self):
        print('_update_neighbor_fn')
        self.neighbor_fn = partition.neighbor_list(self.simulation_config.displacement,
                                                   self.simulation_config.box_size,
                                                   r_cutoff=self.agent_config.neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   format=partition.Sparse)

    # @param.depends('simulation_config.n_agents', 'simulation_config.box_size', watch=True, on_init=True)
    def update_state_neighbors(self):
        print('_update_state_neighbors')
        # self.is_started = False
        self.state = generate_population(self.simulation_config.n_agents, self.simulation_config.box_size)
        # self.state = Population(positions=self.population_config.positions, thetas=self.population_config.thetas,
        #                         proxs=self.population_config.proxs, motors=self.population_config.motors,
        #                         entity_type=0)

        self.neighbors = self.neighbor_fn.allocate(self.state.positions)
        # self.run(threaded=True)

    # @param.depends('simulation_config.displacement', 'simulation_config.shift', 'simulation_config.map_dim',
    #                'simulation_config.dt', 'agent_config.speed_mul', 'agent_config.theta_mul',
    #                'agent_config.proxs_dist_max', 'agent_config.proxs_cos_min', 'agent_config.base_length',
    #                'agent_config.wheel_diameter', 'simulation_config.entity_behaviors', 'engine_config.behavior_bank',
    #                watch=True, on_init=True)
    def update_function_update(self):
        print("_update_function_update")
        self.update_fn = dynamics(self.engine_config, self.simulation_config, self.agent_config)

        if self.simulation_config.to_jit:
            self.update_fn = jit(self.update_fn)

    # @param.depends('simulation_config.n_agents', watch=True, on_init=True)
    def update_behaviors(self):
        print('_update_behaviors')
        self.simulation_config.entity_behaviors = np.zeros(self.simulation_config.n_agents, dtype=int)

    def set_behavior(self, e_idx, behavior_name):
        self.simulation_config.entity_behaviors[e_idx] = self.engine_config.behavior_name_map[behavior_name]  # self.behavior_config.entity_behaviors.at[e_idx].set(self.behavior_config.behavior_name_map[behavior_name])

    def set_motors(self, e_idx, motors):
        if self.behavior_config.entity_behaviors[e_idx] != self.behavior_config.behavior_name_map['manual']:
            self.set_behavior(e_idx, 'manual')
        self.state = Population(positions=self.state.positions,
                                thetas=self.state.thetas,
                                proxs=self.state.proxs,
                                motors=self.state.motors.at[e_idx, :].set(jnp.array(motors)),
                                entity_type=self.state.entity_type)

    def run(self, threaded=False):
        if self.is_started:
            raise Exception("Simulator is already started")
        if threaded:
            threading.Thread(target=self._run).start()
        else:
            return self._run()

    def _run(self, num_loops=math.inf):
        self.is_started = True
        print('Run starts')
        loop_count = 0
        while loop_count < num_loops:
            # print(self.simulation_config.entity_behaviors)
            if self.simulation_config.freq is not None:
                time.sleep(1. / self.simulation_config.freq)

            if not self.is_started:
                break
            if self.simulation_config.to_jit:
                new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_steps_lax, self.update_fn,
                                                    (self.state, self.neighbors))
            else:
                #assert False, "not good, modifies self.state"
                #val = (self.state, self.neighbors)
                for i in range(0, self.simulation_config.num_steps_lax):
                    self.state, self.neighbors = self.update_fn(i, (self.state, self.neighbors))
                new_state = self.state
                #new_state, neighbors = val

            # If the neighbor list can't fit in the allocation, rebuild it but bigger.
            if neighbors.did_buffer_overflow:
                print('REBUILDING')
                neighbors = self.simulation_config.neighbor_fn.allocate(self.state.positions)
                new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_lax_loops, self.update_fn, (self.state, neighbors))
                assert not neighbors.did_buffer_overflow

            self.state = new_state

            loop_count += 1

        print('Run stops')

    def stop(self):
        self.is_started = False


if __name__ == "__main__":

    agent_config = config.AgentConfig()
    simulation_config = config.SimulatorConfig()
    engine_config = config.EngineConfig()

    simulator = Simulator(simulation_config=simulation_config, agent_config=agent_config,
                          engine_config=engine_config)


    #simulator.set_motors(0, jnp.array([0., 0.]))
    simulator.run()

