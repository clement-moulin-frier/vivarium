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




class Simulator(param.Parameterized):
    simulation_config = param.ClassSelector(config.SimulatorConfig, instantiate=False)
    agent_config = param.ClassSelector(config.AgentConfig, instantiate=False)
    # behavior_config = param.ClassSelector(config.BehaviorConfig, instantiate=False)
    population_config = param.ClassSelector(config.PopulationConfig, instantiate=False)
    is_started = param.Boolean(False)
    engine_config = param.ClassSelector(config.EngineConfig)


    def __init__(self, **params):
        super().__init__(**params)
        self.engine_config.behavior_name_map['manual'] = len(self.engine_config.behavior_bank) - 1
    #     self._update_space()
    #     self.param.watch_values(self._update_space, ['simulation_config'])
    #     self.param.watch_values(self._update_neighbor_fn, ['displacement', 'simulation_config', 'agent_config.neighbor_radius'])
    #     self.param.watch_values(self._update_state_neighbors,  ['population_config', 'simulation_config.box_size'])
    #     self.param.watch_values(self._update_function_update,
    #                      ['simulation_config.displacement', 'simulation_config.shift',
    #                       'simulation_config.map_dim', 'simulation_config.dt',
    #                       'agent_config.speed_mul', 'agent_config.theta_mul',
    #                       'agent_config.proxs_dist_max', 'agent_config.proxs_cos_min',
    #                       'agent_config.base_length', 'agent_config.wheel_diameter',
    #                       'behavior_config.entity_behaviors', 'behavior_config.behavior_bank'])


    @param.depends('simulation_config.box_size', watch=True, on_init=True)
    def _update_space(self):
        displacement, shift = space.periodic(self.simulation_config.box_size)
        self.engine_config = config.EngineConfig(displacement=displacement, shift=shift,
                                                 entity_behaviors=np.zeros(self.population_config.n_agents))

    @param.depends('engine_config.displacement', 'simulation_config.box_size', 'agent_config.neighbor_radius',
                   watch=True, on_init=True)
    def _update_neighbor_fn(self):
        self.neighbor_fn = partition.neighbor_list(self.engine_config.displacement,
                                                   self.simulation_config.box_size,
                                                   r_cutoff=self.agent_config.neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   format=partition.Sparse)

    @param.depends('population_config', 'simulation_config.box_size', watch=True, on_init=True)
    def _update_state_neighbors(self):
        self.state = generate_population(self.population_config.n_agents, self.simulation_config.box_size)
        # self.state = Population(positions=self.population_config.positions, thetas=self.population_config.thetas,
        #                         proxs=self.population_config.proxs, motors=self.population_config.motors,
        #                         entity_type=0)

        self.neighbors = self.neighbor_fn.allocate(self.state.positions)

    @param.depends('engine_config.displacement', 'engine_config.shift', 'simulation_config.map_dim',
                   'simulation_config.dt', 'agent_config.speed_mul', 'agent_config.theta_mul',
                   'agent_config.proxs_dist_max', 'agent_config.proxs_cos_min', 'agent_config.base_length',
                   'agent_config.wheel_diameter', 'engine_config.entity_behaviors', 'engine_config.behavior_bank',
                   watch=True, on_init=True)
    def _update_function_update(self):
        print("Update update function due to watched param dependencies")
        self.update_fn = dynamics(self.engine_config, self.simulation_config, self.agent_config)

        if self.simulation_config.to_jit:
            self.update_fn = jit(self.update_fn)
    @param.depends('population_config.n_agents', watch=True, on_init=True)
    def _update_behaviors(self):
        #self.behavior_bank = self.predefined_behaviors + [behaviors.noop] * self.population_config.n_agents
        self.engine_config.entity_behaviors = np.zeros(self.population_config.n_agents, dtype=int)

    def set_behavior(self, e_idx, behavior_name):
        #self.behavior_config.behavior_bank[-e_idx - 1] = behaviors.apply_motors
        self.behavior_config.entity_behaviors = self.behavior_config.entity_behaviors.at[e_idx].set(self.behavior_config.behavior_name_map[behavior_name])
        #self.update_fn = dynamics(self.simulation_config, self.agent_config, self.behavior_config)

    def set_motors(self, e_idx, motors):
        if self.behavior_config.entity_behaviors[e_idx] != self.behavior_config.behavior_name_map['manual']:
            self.set_behavior(e_idx, 'manual')
        # self.behavior_config.entity_behaviors = self.behavior_config.entity_behaviors.at[e_idx].set(self.behavior_config.behavior_name_map['manual'])
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

    def _run(self):

        self.is_started = True
        print('Run starts')
        while True:

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

        print('Run stops')

    def stop(self):
        self.is_started = False



if __name__ == "__main__":

    agent_config = config.AgentConfig()
    simulation_config = config.SimulatorConfig()
    population_config = config.PopulationConfig()
    # behavior_config = config.BehaviorConfig(population_config=population_config)

    simulator = Simulator(simulation_config=simulation_config, agent_config=agent_config,
                          population_config=population_config)

    #simulator.set_motors(0, jnp.array([0., 0.]))
    simulator.run()
